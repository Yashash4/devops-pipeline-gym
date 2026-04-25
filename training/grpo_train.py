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
    "dev": "You are a Developer. You write configs and propose fixes. Actions you may take: view_config, edit_config, run_migration.",
    "sre": "You are an SRE. You investigate and diagnose problems. Actions you may take: view_logs, view_pipeline.",
    "ops": "You are an Ops engineer. You manage production deployments. Actions you may take: deploy, rollback, approve, abort.",
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

    user = textwrap.dedent(
        f"""
        ROLE: {role_desc}

        TASK: {obs.get('task_description', '')}
        GOAL: {obs.get('goal', '')}

        CURRENT SERVICES:
        {service_lines}

        LAST ACTION RESULT: {last_result}

        AVAILABLE ACTIONS: {', '.join(available) if available else '(none)'}

        OUTPUT SCHEMA (strict — must match Pydantic PipelineAction):
          role values:         sre | dev | ops          (lowercase)
          action_type values:  view_pipeline | view_logs | view_config | edit_config |
                               run_migration | deploy | rollback | approve | abort
          service_name:        api-gateway | auth-service | cache-service |
                               database-primary | web-frontend
          config_edits:        LIST of objects, each with exactly two keys "key" and "value",
                               both strings. Key uses dot-notation
                               (e.g. "database.pool_size", "redis.host"). NOT a dict
                               of named sections like {{"environment": ..., "packages": ...}}.

        EXAMPLES (valid JSON bodies):
          {{"action_type": "view_pipeline", "role": "sre"}}
          {{"action_type": "view_logs", "service_name": "cache-service", "role": "sre"}}
          {{"action_type": "edit_config", "service_name": "cache-service",
            "config_edits": [{{"key": "redis.host", "value": "redis-prod.internal:6379"}}],
            "role": "dev"}}
          {{"action_type": "deploy", "service_name": "api-gateway",
            "target_version": "v2.3.1", "role": "ops"}}

        Respond with ONE JSON action matching the schema above. No prose, no code fences.
        """
    ).strip()
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

    logger.info("Loading base model: %s", args.model)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )

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
    if args.sft_adapter_path:
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

    try:
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            lora_alpha=16,
            lora_dropout=0.0,
            bias="none",
            use_gradient_checkpointing="unsloth",
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
        warmup_steps=1,
        max_grad_norm=1.0,
        logging_steps=1,
        save_steps=20,
        max_steps=args.max_steps,
        beta=0.01,
        push_to_hub=False,
    )
    # Newer TRL versions support loss_type + mask_truncated_completions.
    # Pass them best-effort; GRPOConfig raises TypeError on unknown kwargs.
    for extra_key, extra_val in (
        ("loss_type", "dapo"),
        ("mask_truncated_completions", True),
        ("use_vllm", bool(args.use_vllm)),
    ):
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

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_fn],
        args=grpo_config,
        train_dataset=dataset,
        callbacks=[curriculum_callback],
    )

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
