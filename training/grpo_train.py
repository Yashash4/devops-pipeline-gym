"""GRPO training for DevOps Pipeline Gym (Round 2).

Unsloth + TRL GRPO fine-tunes a small LLM against the running OpenEnv
HTTP endpoint. For the Phase 6 dry-run we run 5 steps on Kaggle T4 with
Qwen 1.5B 4-bit just to prove the pipeline wires up end to end.

Not importable on CPU-only machines: Unsloth + bitsandbytes require CUDA.
All heavy imports are inside main() so this module can be syntax-checked
without the deps.

Typical invocation (Kaggle T4 dry-run):
    python training/grpo_train.py \\
        --model unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit \\
        --env-url http://localhost:8000 \\
        --max-steps 5 \\
        --num-generations 2 \\
        --output-dir ./dryrun

Saturday H100 real training:
    python training/grpo_train.py \\
        --model unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit \\
        --env-url http://localhost:8000 \\
        --max-steps 100 \\
        --num-generations 8 \\
        --use-vllm \\
        --output-dir ./outputs/run1

Phase 6.5 SFT-adapter-fix test procedure (run on Kaggle T4):
    # 1. Restart kernel, cd /kaggle/working/gym, git pull
    # 2. SFT warmup:
    python training/sft_warmup.py \\
        --model unsloth/Qwen3-0.6B-bnb-4bit \\
        --trajectories data/sft_trajectories.jsonl \\
        --output-dir outputs/sft_warmup \\
        --epochs 2
    # 3. GRPO with SFT adapter (must NOT crash with CUDA assert / NaN):
    python training/grpo_train.py \\
        --model unsloth/Qwen3-0.6B-bnb-4bit \\
        --sft-adapter-path outputs/sft_warmup/final \\
        --env-url http://127.0.0.1:8000 \\
        --max-steps 3 --num-generations 4 --max-episode-steps 8 \\
        --output-dir outputs/grpo_sft_test
    # 4. Verify: no CUDA assert errors, reward > 0 on step 1,
    #    frac_reward_zero_std < 0.5, adapter log lines show two adapters:
    #    "sft_warmup" (frozen) + "default"/"grpo_training" (trainable).
    # If this still NaN-fails: drop --sft-adapter-path to restore pre-fix
    # behaviour while we investigate further upstream.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import textwrap
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger("grpo_train")


# ─── Prompt / completion helpers (dep-free, can be imported on CPU) ──────────

_ROLE_DESCRIPTIONS = {
    "dev": "You are a Developer. You write configs and propose fixes. Actions: view_config, edit_config, run_migration.",
    "sre": "You are an SRE. You investigate and diagnose. Actions: view_logs, view_pipeline.",
    "ops": "You are an Ops engineer. You manage production deployments. Actions: deploy, rollback, approve, abort.",
}

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an autonomous DevOps agent operating a CI/CD pipeline.
    Respond with EXACTLY ONE JSON object describing the next action.
    Do not include markdown code fences, prose, or explanation.

    Fields:
      action_type  (one of view_pipeline, view_logs, view_config, edit_config,
                   run_migration, deploy, rollback, approve, abort)
      role         (dev / sre / ops — must match the role assigned to you)
      service_name, target_version, config_edits, migration_name, reason
    """
).strip()


def build_prompt(obs: Dict[str, Any], role: str) -> str:
    """Render a role-specific prompt from an observation dict.

    `obs` is expected to be the `observation` field of an env reset/step
    response (a plain dict produced by PipelineObservation.model_dump()).
    """
    role_desc = _ROLE_DESCRIPTIONS.get(role, _ROLE_DESCRIPTIONS["sre"])
    services = obs.get("services", []) or []
    service_lines = "\n".join(
        f"  - {s.get('name')} | health={s.get('health')} | "
        f"latency={s.get('request_latency_ms', 0):.0f}ms | "
        f"err={s.get('error_rate', 0):.1f}/s"
        for s in services
    )
    available = obs.get("available_actions", [])
    last_result = obs.get("last_action_result") or "none"

    actions_str = ', '.join(available) if available else '(none)'
    # Build flush-left to match SFT trajectory user-prompt format byte-exactly.
    # Avoid textwrap.dedent — it misbehaves when an interpolated multi-line
    # value (service_lines) has different leading whitespace than the f-string
    # template, producing mixed indents.
    # ONE-LINE schema hint added (proof-run experiment): force config_edits
    # shape because parser-coercion alone wasn't enough — clipped completions
    # with malformed config_edits poisoned reward signal in the prior tiny run.
    # Cost: ~30 chars OOD vs SFT (still ~98% match on prompt content).
    user = (
        f"ROLE: {role_desc}\n"
        f"\n"
        f"TASK: {obs.get('task_description', '')}\n"
        f"GOAL: {obs.get('goal', '')}\n"
        f"\n"
        f"CURRENT SERVICES:\n"
        f"{service_lines}\n"
        f"\n"
        f"LAST ACTION RESULT: {last_result}\n"
        f"PREVIOUS HANDOFF NOTES: none\n"
        f"\n"
        f"AVAILABLE ACTIONS: {actions_str}\n"
        f"\n"
        f"FORMAT: config_edits (when present) must be a list of {{key, value}} objects.\n"
        f"\n"
        f"Respond with ONE JSON action."
    )
    return user


def poll_curriculum_progress(env_url: str, output_path: "Path", step: int) -> None:
    """Phase J.7 — GET /curriculum_progress and append to a JSONL log.

    Best-effort: any failure is logged as a warning print so training never
    crashes from telemetry. Uses urllib (stdlib) — no new dependency.
    """
    import json as _json
    import time as _time
    import urllib.error
    import urllib.request

    url = f"{env_url.rstrip('/')}/curriculum_progress"
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status != 200:
                print(f"[curriculum_poll] step={step} HTTP {resp.status} — skipping",
                      flush=True)
                return
            payload = _json.loads(resp.read().decode("utf-8"))
        payload["step"] = step
        payload["timestamp"] = _time.time()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(_json.dumps(payload) + "\n")
    except (urllib.error.URLError, OSError, ValueError) as e:
        print(f"[curriculum_poll] step={step} failed: {type(e).__name__}: {e} — continuing",
              flush=True)


def parse_completion(text: str) -> Dict[str, Any]:
    """Extract an action dict from a model completion.

    Falls back to a safe view_pipeline action when parsing fails — this
    means invalid JSON gets a small investigative reward rather than
    crashing the training loop. `role=sre` is the safest default since
    SRE is always allowed to view_pipeline.
    """
    fallback = {"action_type": "view_pipeline", "role": "sre"}
    if not text:
        return fallback
    text = text.strip()
    if text.startswith("```"):
        text = text[3:]
        if text.startswith("json"):
            text = text[4:]
        text = text.lstrip("\n")
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    # Trim any prose before / after the JSON object.
    first = text.find("{")
    last = text.rfind("}")
    if first == -1 or last <= first:
        return fallback
    try:
        data = json.loads(text[first : last + 1])
        if not isinstance(data, dict) or "action_type" not in data:
            return fallback
        # Aggressively coerce config_edits to a valid List[ConfigEdit] or drop it.
        # Pydantic List[ConfigEdit] rejects everything except list-of-{key,value}.
        # Hardened to handle every shape we've observed the model emit:
        #   list of {key,value}                -> pass through
        #   list of single-key dicts {x: y}    -> [{key: x, value: y}, ...]
        #   list of "x=y" strings              -> [{key: x, value: y}, ...]
        #   {key, value} dict                  -> wrap in list
        #   single-key dict {x: y}             -> [{key: x, value: y}]
        #   multi-key dict {a: b, c: d}        -> [{key: a, value: b}, {key: c, value: d}]
        #   "x=y" string                       -> [{key: x, value: y}]
        #   bool, int, None, "", "none", etc.  -> drop the field
        ce = data.get("config_edits")
        coerced = None
        if isinstance(ce, list):
            new_items = []
            for item in ce:
                if isinstance(item, dict):
                    if "key" in item and "value" in item:
                        new_items.append({"key": str(item["key"]), "value": str(item["value"])})
                    elif len(item) == 1:
                        k, v = next(iter(item.items()))
                        new_items.append({"key": str(k), "value": str(v)})
                elif isinstance(item, str) and "=" in item:
                    k, _, v = item.partition("=")
                    new_items.append({"key": k.strip(), "value": v.strip()})
            coerced = new_items if new_items else None
        elif isinstance(ce, dict):
            if "key" in ce and "value" in ce:
                coerced = [{"key": str(ce["key"]), "value": str(ce["value"])}]
            elif len(ce) == 1:
                k, v = next(iter(ce.items()))
                coerced = [{"key": str(k), "value": str(v)}]
            elif len(ce) > 1:
                coerced = [{"key": str(k), "value": str(v)} for k, v in ce.items()]
        elif isinstance(ce, str) and "=" in ce:
            k, _, v = ce.partition("=")
            coerced = [{"key": k.strip(), "value": v.strip()}]
        if coerced is not None:
            data["config_edits"] = coerced
        else:
            data.pop("config_edits", None)
        # Normalise enum-valued fields to lowercase. Kaggle dry-run showed the
        # base model likes to emit "role": "SRE" / "DEV" / "OPS" (uppercase),
        # which Pydantic rejects — the Role enum values are lowercase ("sre",
        # "dev", "ops"). Do the same for action_type since models also tend
        # toward "DEPLOY" / "VIEW_LOGS". Any non-string value is left as-is
        # so Pydantic's own validation surfaces the real error.
        for key in ("role", "action_type"):
            if isinstance(data.get(key), str):
                data[key] = data[key].lower()
        return data
    except Exception:
        return fallback


# ─── CLI ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="GRPO training for DevOps Pipeline Gym")
    p.add_argument("--model", default="unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
                   help="HF model id for the base model")
    p.add_argument("--env-url", default="http://localhost:8000",
                   help="URL of the DevOps Pipeline Gym env server")
    p.add_argument("--max-steps", type=int, default=5,
                   help="GRPO max_steps (total optimisation steps)")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--num-generations", type=int, default=2,
                   help="Completions per prompt (GRPO group size)")
    p.add_argument("--learning-rate", type=float, default=5e-6)
    p.add_argument("--max-completion-length", type=int, default=256)
    p.add_argument("--curriculum-mode", choices=["on", "off"], default="on")
    p.add_argument("--output-dir", default="./outputs/grpo_run")
    p.add_argument("--use-vllm", action="store_true",
                   help="Enable vLLM generation backend (H100 recommended)")
    p.add_argument("--prompts-per-task", type=int, default=10,
                   help="Number of initial-observation prompts sampled per task")
    # --- Phase 6.5 flags ---
    p.add_argument("--sft-adapter-path", default=None,
                   help="Path to an SFT adapter produced by training/sft_warmup.py; "
                        "merged into the base before the new GRPO LoRA goes on top")
    p.add_argument("--max-episode-steps", type=int, default=12,
                   help="Max steps per episode in the multi-step reward rollout")
    p.add_argument("--continuation-temperature", type=float, default=0.7,
                   help="Sampling temperature for continuation actions (steps 2..N)")
    return p


# ─── Env rollout → reward ─────────────────────────────────────────────────────

def _make_sync_client(env_url: str):
    """Return a sync EnvClient context manager. Late import so the module
    remains importable without openenv-core installed in sidecar tooling."""
    from devops_pipeline_gym.client import DevopsPipelineEnv
    return DevopsPipelineEnv(base_url=env_url).sync()


def build_dataset(env_url: str, prompts_per_task: int = 10) -> List[Dict[str, str]]:
    """Sample initial-observation prompts from the env for the GRPO dataset.

    Each dataset entry looks like {"prompt": "<system>\\n<user>"}. The
    reward function later resets the env again to score each completion
    against a matching initial state — GRPO generates multiple completions
    per prompt and compares their rewards.
    """
    tasks = ["clean_deploy", "broken_pipeline", "judgment_call",
             "cascading_failure", "capacity_crisis", "random_incident"]
    prompts: List[Dict[str, str]] = []
    for task in tasks:
        for seed_offset in range(prompts_per_task):
            os.environ["DEVOPS_TASK"] = task
            if task == "random_incident":
                os.environ["DEVOPS_SEED"] = str(6000 + seed_offset)
            # Fresh client per reset — OpenEnv server side creates a session
            with _make_sync_client(env_url) as client:
                result = client.reset()
                obs = result.observation.model_dump()
                role = obs.get("current_role", "sre")
                user = build_prompt(obs, role)
                # Stash task+seed in the prompt metadata so reward_function can
                # reproduce the same state. Simple encoded sentinel line.
                prompt_text = f"{SYSTEM_PROMPT}\n\n<<TASK={task};SEED={seed_offset}>>\n\n{user}"
                prompts.append({"prompt": prompt_text})
    os.environ.pop("DEVOPS_TASK", None)
    os.environ.pop("DEVOPS_SEED", None)
    return prompts


def _extract_sentinel(prompt: str) -> Dict[str, Any]:
    """Pull the <<TASK=...;SEED=...>> sentinel out of a training prompt."""
    import re
    m = re.search(r"<<TASK=([a-z_]+);SEED=(-?\d+)>>", prompt)
    if not m:
        return {"task": "clean_deploy", "seed": 0}
    return {"task": m.group(1), "seed": int(m.group(2))}


def _generate_continuation_action(
    model,
    tokenizer,
    system_prompt: str,
    user_prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
) -> str:
    """Generate ONE action string from current model state given observation.

    Used inside reward_function for multi-step episode rollouts (Phase 6.5
    Option B: the first action in the episode is the GRPO training
    completion; subsequent actions come from this helper using the same
    model via closure).

    Returns the raw completion string (parse_completion will turn it into
    a PipelineAction dict).
    """
    import torch  # late import so module stays importable on CPU hosts

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    prompt_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    completion_ids = output[0][prompt_ids.shape[1]:]
    return tokenizer.decode(completion_ids, skip_special_tokens=True)


def make_reward_function(
    env_url: str,
    model_ref=None,
    tokenizer_ref=None,
    system_prompt: str = "",
    max_episode_steps: int = 12,
    continuation_temperature: float = 0.7,
    continuation_max_tokens: int = 256,
):
    """Multi-step episode reward via closure over model + tokenizer.

    Phase 6.5 Option B: the GRPO training completion is executed as the
    FIRST action (gradients target this). Remaining steps (up to
    max_episode_steps - 1) continue using the SAME model via
    `_generate_continuation_action`, running until env.done or the cap.
    Reward returned is the episode sum — ranges roughly [-3, +5] once
    the terminal bonus/penalty in pipeline_environment.step() fires.

    Backward compatibility: when model_ref/tokenizer_ref are None, falls
    back to single-step bandit behaviour (the original Phase 6 semantics).
    This is only used by CPU-host sanity checks; real training paths
    always pass model and tokenizer.

    Reward is purely outcome-based from env.step() — no LLM judge in the
    loop (Groq judge_client cut for kube-sre-gym overlap; deterministic
    grading is required for plagiarism-check transparency and reproducible
    GRPO gradients).
    """
    from devops_pipeline_gym.models import PipelineAction

    multi_step = model_ref is not None and tokenizer_ref is not None

    def reward_function(completions: List[str], prompts: List[str], **_kwargs) -> List[float]:
        rewards: List[float] = []
        for prompt_text, first_completion in zip(prompts, completions):
            meta = _extract_sentinel(prompt_text)
            task = meta["task"]
            seed = meta["seed"]
            try:
                os.environ["DEVOPS_TASK"] = task
                if task == "random_incident":
                    os.environ["DEVOPS_SEED"] = str(6000 + seed)

                # Parse the GRPO-generated first action.
                try:
                    action_data = parse_completion(first_completion)
                    action1 = PipelineAction(**action_data)
                except Exception as e:
                    logger.warning("step 1 parse failed: %s", str(e)[:80])
                    rewards.append(-1.0)
                    continue

                with _make_sync_client(env_url) as client:
                    client.reset()

                    # STEP 1 — the training completion (GRPO gradient target).
                    try:
                        result = client.step(action1)
                    except Exception as e:
                        logger.warning("step 1 env error: %s", str(e)[:80])
                        rewards.append(-1.0)
                        continue
                    episode_reward = float(getattr(result, "reward", 0.0) or 0.0)
                    done = bool(getattr(result, "done", False))
                    current_obs = getattr(result, "observation", None)

                    # STEPS 2..N — continue with the same model (only when
                    # multi_step; otherwise degrade to single-step bandit).
                    obs_dict: Dict[str, Any] = {}
                    step = 1
                    while multi_step and not done and step < max_episode_steps:
                        if current_obs is None:
                            break
                        if hasattr(current_obs, "model_dump"):
                            obs_dict = current_obs.model_dump()
                        elif isinstance(current_obs, dict):
                            obs_dict = current_obs
                        else:
                            break

                        # Role for the next action's prompt
                        role = "sre"
                        cr = obs_dict.get("current_role")
                        if cr is None:
                            cr = getattr(current_obs, "current_role", None)
                        if cr:
                            role = cr.value if hasattr(cr, "value") else str(cr).lower()

                        user = build_prompt(obs_dict, role)
                        try:
                            cont = _generate_continuation_action(
                                model_ref, tokenizer_ref,
                                system_prompt, user,
                                max_new_tokens=continuation_max_tokens,
                                temperature=continuation_temperature,
                            )
                            action_data_n = parse_completion(cont)
                            action_n = PipelineAction(**action_data_n)
                            step_result = client.step(action_n)
                            step_reward = float(getattr(step_result, "reward", 0.0) or 0.0)
                            episode_reward += step_reward
                            done = bool(getattr(step_result, "done", False))
                            current_obs = getattr(step_result, "observation", None)
                        except Exception as e:
                            logger.warning("step %d failed: %s", step + 1, str(e)[:80])
                            episode_reward -= 0.1  # small penalty; keep the loop going
                        step += 1

                rewards.append(episode_reward)
            except Exception as e:
                logger.error("episode rollout failed: %s: %s", type(e).__name__, str(e)[:150])
                rewards.append(-1.0)
        return rewards

    return reward_function


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "final").mkdir(parents=True, exist_ok=True)

    # Heavy imports here so CPU-only checkers can still import the module.
    try:
        import torch  # noqa: F401
        from datasets import Dataset
        from unsloth import FastLanguageModel
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as e:
        logger.error(
            "Phase 6 training requires GPU deps (torch+CUDA, unsloth, trl, datasets, bitsandbytes). "
            "Install: pip install -e '.[training]'. Import error: %s", e,
        )
        raise SystemExit(2)

    # Phase 6.5 informational — does TRL expose its experimental OpenEnv
    # helper? Option A path. We stay on Option B (closure) regardless.
    try:
        from trl.experimental.openenv import generate_rollout_completions  # noqa: F401
        logger.info("TRL experimental.openenv AVAILABLE (Option A viable, not used this run)")
    except ImportError:
        logger.info("TRL experimental.openenv NOT available; using Option B (closure-based multi-step)")

    no_unsloth = os.environ.get("NO_UNSLOTH", "").lower() in ("1", "true", "yes")
    if no_unsloth:
        # Pure HF + PEFT + TRL path (kube-sre-gym 1st place pattern).
        # Avoids Unsloth's xformers attention which has no backward kernel
        # for Qwen3's 5D BMGHK shape (xformers/ops/fmha/dispatch.py:83
        # NotImplementedError).
        logger.info("Loading base model via PURE HF (NO_UNSLOTH=1): %s", args.model)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",   # NOT xformers — works for Qwen3 GQA
            device_map="cuda:0",
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        # Prepare for k-bit training — required for QLoRA gradient flow
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
        logger.info("Pure HF model loaded with SDPA attention + 4-bit quant + grad checkpointing")
    else:
        logger.info("Loading base model via Unsloth: %s", args.model)
        fpt_kwargs = dict(
            model_name=args.model,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )
        if args.use_vllm:
            fpt_kwargs.update(
                fast_inference=True,
                max_lora_rank=16,
                gpu_memory_utilization=0.5,
                enforce_eager=True,
            )
            logger.info("vLLM colocate enabled (fast_inference=True, max_lora_rank=16, gpu_mem=0.5, enforce_eager=True)")
        model, tokenizer = FastLanguageModel.from_pretrained(**fpt_kwargs)

    # Phase 6.5 hotfix — load the SFT adapter as a FROZEN named prior and
    # let GRPO stack a new trainable adapter on top. Do NOT merge.
    #
    # Why not merge: `merge_and_unload()` on a LoRA attached to a 4-bit
    # bitsandbytes base runs a dequantize -> add -> requantize cycle.
    # Rounding error from re-quantisation corrupts enough weights to produce
    # NaN in the logits during torch.multinomial sampling the first time the
    # model generates (Kaggle Qwen3-0.6B run 2026-04-24). Confirmed upstream:
    # PEFT issue #2321 — "Merge LoRA module to 4-bit linear may get different
    # generations due to rounding errors" (the warning emitted right before
    # the CUDA assert).
    #
    # Stacking via named adapters sidesteps the quantisation round-trip:
    #   - "sft_warmup"   — frozen, applied during forward pass (the prior)
    #   - "default"      — trainable, GRPO gradients flow here (the new LoRA)
    if args.sft_adapter_path and args.sft_adapter_path.lower() != "none":
        from peft import PeftModel
        logger.info(
            "Loading SFT adapter as frozen prior: %s", args.sft_adapter_path
        )
        model = PeftModel.from_pretrained(
            model,
            args.sft_adapter_path,
            adapter_name="sft_warmup",
            is_trainable=False,
        )
        # Defense in depth — explicitly freeze SFT params regardless of what
        # downstream adapter stacking does.
        for name, param in model.named_parameters():
            if "sft_warmup" in name:
                param.requires_grad = False
        logger.info(
            "SFT adapter loaded as 'sft_warmup' (frozen prior). "
            "GRPO LoRA will stack on top."
        )
    else:
        logger.info(
            "No SFT adapter — training GRPO from raw base "
            "(vLLM-compatible single LoRA path)"
        )

    # Use standard PyTorch gradient checkpointing instead of Unsloth's
    # "smart" smart-offload when ANY of: --use-vllm OR SFT adapter loaded.
    # Smart-offload breaks autograd graph through stacked PEFT adapters
    # (frozen sft_warmup + trainable default), causing
    # "RuntimeError: element 0 of tensors does not require grad and does
    # not have a grad_fn" at first backward(). Standard checkpointing
    # costs ~2GB more VRAM but autograd works correctly.
    sft_loaded = bool(args.sft_adapter_path and args.sft_adapter_path.lower() != "none")
    grad_ckpt_mode = True if (args.use_vllm or sft_loaded) else "unsloth"
    logger.info(
        "Gradient checkpointing mode: %s (use_vllm=%s, sft_loaded=%s)",
        grad_ckpt_mode, args.use_vllm, sft_loaded,
    )
    if no_unsloth:
        # Pure PEFT path — no Unsloth wrapping. SFT adapter (if loaded) is
        # already in 'sft_warmup' (frozen). Now add 'default' as trainable.
        from peft import LoraConfig
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        )
        if sft_loaded:
            # Already wrapped in PeftModel from SFT load — use add_adapter
            model.add_adapter("default", lora_config)
            model.set_adapter("default")
            logger.info("GRPO LoRA added via PEFT add_adapter on top of frozen SFT prior")
        else:
            from peft import get_peft_model
            model = get_peft_model(model, lora_config)
            logger.info("GRPO LoRA added via pure PEFT get_peft_model (no SFT prior)")
        # Skip the Unsloth try/except block entirely — pure PEFT done
        try:
            pass  # placeholder to keep the try/except structure below valid
        except Exception:
            pass
    else:
        try:
            # lora_alpha = 2 * r is Daniel Han (Unsloth) and kube-sre-gym (sid-rp)
            # standard. lora_alpha=16 (1× rank) was a leftover from earlier dry-runs.
            model = FastLanguageModel.get_peft_model(
                model,
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                use_gradient_checkpointing=grad_ckpt_mode,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"],
            )
            logger.info(
                "GRPO LoRA added via Unsloth. Active adapter: %s",
                getattr(model, "active_adapter", "default"),
            )
        except Exception as e:
            # Unsloth's get_peft_model isn't always happy being handed a model
            # that's already wrapped in PeftModel. Fall back to stock PEFT so
            # we at least get the adapter we need on top of the SFT prior.
            logger.warning(
                "Unsloth get_peft_model failed on pre-wrapped PeftModel (%s: %s); "
                "falling back to standard peft.LoraConfig + add_adapter",
                type(e).__name__, e,
            )
            from peft import LoraConfig
            grpo_lora_config = LoraConfig(
                r=16,
                lora_alpha=16,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"],
                lora_dropout=0.0,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model.add_adapter("grpo_training", grpo_lora_config)
            model.set_adapter("grpo_training")
            logger.info(
                "GRPO adapter added via PEFT fallback; active adapter: grpo_training"
            )

    # Adapter state sanity check — blows up loudly if we ended up with a
    # model that can't train (no trainable params) or has suspiciously many
    # (LoRA should be well under 5% of total parameters).
    if hasattr(model, "peft_config") and model.peft_config:
        adapter_names = list(model.peft_config.keys())
        logger.info("Active adapter(s): %s", adapter_names)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        pct = 100.0 * trainable_params / max(1, total_params)
        logger.info(
            "Trainable params: %s / %s (%.2f%%)",
            f"{trainable_params:,}", f"{total_params:,}", pct,
        )
        if trainable_params == 0:
            raise RuntimeError(
                "No trainable parameters! GRPO cannot train. Check adapter setup."
            )
        if pct > 5.0:
            logger.warning(
                "Trainable params suspiciously high (%.1f%%). LoRA should be <5%%.",
                pct,
            )

    # Belt-and-suspenders for stacked-adapter setups where Unsloth doesn't
    # always wire up requires_grad / adapter activation correctly.
    # 1) train mode (vs inference)
    # 2) explicitly enable adapter layers (in case enable_adapters was disabled)
    # 3) forward-pass smoke test — fail FAST if loss has no grad_fn
    model.train()
    if hasattr(model, "enable_adapter_layers"):
        try:
            model.enable_adapter_layers()
        except Exception as e:
            logger.warning("enable_adapter_layers() failed: %s", e)

    # Compat shim for Unsloth-patched GRPOTrainer.
    # When `from unsloth import ...` runs anywhere, Unsloth monkey-patches
    # TRL's GRPOTrainer into its own UnslothGRPOTrainer (saved to
    # unsloth_compiled_cache/). The patched trainer calls
    # model.for_training() and model.for_inference() — Unsloth-specific
    # methods that pure-HF models don't have. Add no-op shims so the
    # patched trainer doesn't crash with AttributeError.
    if no_unsloth:
        import types as _types

        def _ufm_for_training(self, *args, **kwargs):
            self.train()
            return self

        def _ufm_for_inference(self, *args, **kwargs):
            self.eval()
            return self

        # Patch the outermost model and walk down to be safe (PEFT's
        # __getattr__ falls through to base_model.model, but explicit
        # patching at multiple levels avoids surprises).
        targets = [model]
        if hasattr(model, "base_model"):
            targets.append(model.base_model)
            if hasattr(model.base_model, "model"):
                targets.append(model.base_model.model)
        for t in targets:
            if not hasattr(t, "for_training"):
                t.for_training = _types.MethodType(_ufm_for_training, t)
            if not hasattr(t, "for_inference"):
                t.for_inference = _types.MethodType(_ufm_for_inference, t)
        logger.info("Patched %d model levels with for_training/for_inference shims", len(targets))

    # Forward-pass smoke test: catch the "no grad_fn" bug BEFORE trainer.train()
    # so we crash with a clear message rather than 2 minutes into the loop.
    try:
        import torch
        smoke_input = tokenizer("test", return_tensors="pt").to(model.device)
        smoke_out = model(**smoke_input, labels=smoke_input["input_ids"])
        if not getattr(smoke_out.loss, "requires_grad", False) or smoke_out.loss.grad_fn is None:
            logger.error(
                "FORWARD SMOKE TEST FAILED: loss has no grad_fn. "
                "Stacked-adapter autograd is broken. "
                "requires_grad=%s, grad_fn=%s",
                getattr(smoke_out.loss, "requires_grad", "?"),
                smoke_out.loss.grad_fn,
            )
            # Try one more rescue: explicitly mark all LoRA "default" params trainable
            n_fixed = 0
            for name, param in model.named_parameters():
                if "lora_" in name and "default" in name:
                    param.requires_grad = True
                    n_fixed += 1
            logger.warning("Forced requires_grad=True on %d LoRA params", n_fixed)
            # Re-test
            smoke_out = model(**smoke_input, labels=smoke_input["input_ids"])
            if smoke_out.loss.grad_fn is None:
                raise RuntimeError(
                    "Stacked PEFT autograd broken. Loss has no grad_fn even after "
                    "manual requires_grad fix. Cannot train. Drop --sft-adapter-path "
                    "or remove Unsloth wrapping."
                )
        logger.info(
            "Forward smoke test PASSED: loss=%.4f has grad_fn (autograd OK)",
            float(smoke_out.loss.item()),
        )
    except RuntimeError:
        raise
    except Exception as e:
        logger.warning("Forward smoke test inconclusive (%s): proceeding anyway", e)

    logger.info("Building dataset from env rollouts (prompts_per_task=%d)", args.prompts_per_task)
    prompts = build_dataset(args.env_url, prompts_per_task=args.prompts_per_task)
    logger.info("Dataset size: %d prompts", len(prompts))
    dataset = Dataset.from_list(prompts)

    reward_fn = make_reward_function(
        env_url=args.env_url,
        model_ref=model,
        tokenizer_ref=tokenizer,
        system_prompt=SYSTEM_PROMPT,
        max_episode_steps=args.max_episode_steps,
        continuation_temperature=args.continuation_temperature,
        continuation_max_tokens=args.max_completion_length,
    )

    cfg_kwargs = dict(
        output_dir=str(out_dir),
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=2,
        max_grad_norm=1.0,
        logging_steps=1,
        save_steps=20,
        max_steps=args.max_steps,
        beta=0.01,
        push_to_hub=False,
        report_to=["trackio"],
        # kube-sre-gym pattern — modern PyTorch checkpointing (the older
        # use_reentrant=True is deprecated and has known issues with PEFT).
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # Explicit GRPO temperature; TRL default may vary across versions.
        # 1.0 maximizes exploration (GRPO needs diverse completions for
        # meaningful advantage estimation). kube-sre-gym uses 1.0.
        temperature=1.0,
    )
    # Newer TRL versions support loss_type + mask_truncated_completions +
    # vLLM colocate plumbing. Pass best-effort; GRPOConfig raises TypeError
    # on unknown kwargs.
    extras = [
        ("loss_type", "dapo"),
        ("mask_truncated_completions", True),
        ("use_vllm", bool(args.use_vllm)),
    ]
    if args.use_vllm:
        # In-process vLLM (no external server). Memory fraction must align
        # with FastLanguageModel.from_pretrained gpu_memory_utilization above.
        extras.append(("vllm_mode", "colocate"))
        extras.append(("vllm_gpu_memory_utilization", 0.5))
    for extra_key, extra_val in extras:
        try:
            _probe = GRPOConfig(**{**cfg_kwargs, extra_key: extra_val})
            cfg_kwargs[extra_key] = extra_val
        except TypeError:
            logger.warning("GRPOConfig does not accept %s; skipping", extra_key)

    grpo_config = GRPOConfig(**cfg_kwargs)

    # Phase J.7 — poll /curriculum_progress every 10 GRPO steps and append
    # the snapshot to outputs/<run>/curriculum_progress.jsonl. Best-effort:
    # endpoint failures log a warning, never crash training.
    from transformers import TrainerCallback

    class _CurriculumProgressCallback(TrainerCallback):
        def __init__(self, env_url: str, output_path: Path, every_n_steps: int = 10):
            self.env_url = env_url
            self.output_path = output_path
            self.every_n_steps = max(1, int(every_n_steps))

        def on_step_end(self, args_, state, control, **_kwargs):
            step = int(getattr(state, "global_step", 0))
            if step > 0 and step % self.every_n_steps == 0:
                poll_curriculum_progress(self.env_url, self.output_path, step)

    curriculum_callback = _CurriculumProgressCallback(
        env_url=args.env_url,
        output_path=out_dir / "curriculum_progress.jsonl",
        every_n_steps=10,
    )

    # Per-step reward CSV logger — writes one row per logging_step with
    # whatever metrics TRL exposes (loss, reward, reward_std, kl, lr, etc.)
    # Mirrors kube-sre-gym's reward_log.csv pattern (Help Guide #19: judges
    # value visible per-episode tabular evidence on the trained-adapter Hub).
    import csv

    class _RewardCSVCallback(TrainerCallback):
        def __init__(self, csv_path: Path):
            self.csv_path = csv_path
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(self.csv_path, "w", newline="", encoding="utf-8")
            self._writer = None
            self._fieldnames: list = []

        def on_log(self, args_, state, control, logs=None, **_kwargs):
            if not logs:
                return
            row = {"step": int(getattr(state, "global_step", 0)), **logs}
            # Initialize header on first row using whatever keys TRL emitted
            if self._writer is None:
                self._fieldnames = list(row.keys())
                self._writer = csv.DictWriter(self._fh, fieldnames=self._fieldnames)
                self._writer.writeheader()
            else:
                # If TRL emits new keys later, extend gracefully
                for k in row.keys():
                    if k not in self._fieldnames:
                        self._fieldnames.append(k)
            # Write only known fields; ignore unexpected nested types
            safe_row = {k: row.get(k, "") for k in self._fieldnames}
            self._writer.writerow(safe_row)
            self._fh.flush()

        def on_train_end(self, args_, state, control, **_kwargs):
            try:
                self._fh.close()
            except Exception:
                pass

    reward_csv_callback = _RewardCSVCallback(out_dir / "grpo_log.csv")

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_fn],
        args=grpo_config,
        train_dataset=dataset,
        callbacks=[curriculum_callback, reward_csv_callback],
    )

    # TRL 0.29.0 / vLLM 0.11.x logprobs format mismatch — TRL expects
    # list-of-lists (top-k per token), vLLM 0.11.x returns plain floats.
    # Patch from kube-sre-gym (sid-rp) train.py. See TRL issue #4159.
    # Only fires when use_vllm is on; harmless no-op otherwise.
    if args.use_vllm and hasattr(trainer, "vllm_generation"):
        _orig_vllm_gen = trainer.vllm_generation.generate

        def _patched_vllm_generate(**kwargs):
            result = _orig_vllm_gen(**kwargs)
            prompt_ids, completion_ids, logprobs, *rest = result
            if logprobs and logprobs[0] and isinstance(logprobs[0][0], float):
                logprobs = [[[lp] for lp in seq] for seq in logprobs]
            return (prompt_ids, completion_ids, logprobs, *rest)

        trainer.vllm_generation.generate = _patched_vllm_generate
        logger.info("Applied TRL/vLLM logprobs compat patch (kube-sre-gym pattern)")

    logger.info("Starting GRPO training: max_steps=%d", args.max_steps)
    # Capture the t=0 snapshot before any training step.
    poll_curriculum_progress(args.env_url, out_dir / "curriculum_progress.jsonl", step=0)
    trainer.train()

    # Save adapter.
    final_dir = out_dir / "final"
    logger.info("Saving adapter to %s", final_dir)
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    # Reward curve PNG.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        history = trainer.state.log_history if hasattr(trainer, "state") else []
        rewards_over_time = [
            entry.get("reward") for entry in history if "reward" in entry
        ]
        if rewards_over_time:
            plt.figure(figsize=(8, 4))
            plt.plot(range(1, len(rewards_over_time) + 1), rewards_over_time, marker="o")
            plt.xlabel("logging step")
            plt.ylabel("mean reward")
            plt.title(f"GRPO reward — {args.model} — {args.max_steps} steps")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(out_dir / "reward_curve.png", dpi=120)
            logger.info("Reward curve saved: %s", out_dir / "reward_curve.png")
        else:
            logger.warning("No reward history found in trainer state — skipping reward_curve.png")
    except Exception as e:
        logger.warning("reward_curve.png generation failed (%s): %s", type(e).__name__, e)

    logger.info("Done. Adapter at %s", final_dir)


if __name__ == "__main__":
    main()
