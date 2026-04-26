"""GRPO training for DevOps Pipeline Gym (clean kube-sre-gym pattern).

Pure HF Transformers + PEFT + TRL. No Unsloth wrapping (xformers BMGHK
bug on Qwen3 GQA), no vLLM (multi-step rollout incompatible with vLLM
colocate's lock model). Same path that won 1st place last hackathon.

Architecture:
  base = AutoModelForCausalLM (4-bit BNB NF4 + SDPA attention)
  + sft_warmup adapter (frozen prior, optional)
  + default adapter (trainable GRPO LoRA)
  -> stock trl.GRPOTrainer with closure-based multi-step rollout reward

Usage:
  python training/grpo_train.py \\
    --model unsloth/Qwen3-1.7B-bnb-4bit \\
    --env-url http://localhost:8000 \\
    --sft-adapter-path /workspace/sft_adapter/final \\
    --max-steps 20 --num-generations 2 --batch-size 1 --grad-accum 4 \\
    --max-completion-length 128 --max-episode-steps 6 \\
    --learning-rate 5e-6 --output-dir /workspace/grpo_output
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import textwrap
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger("grpo_train")


# ─── Prompt template (matches SFT trajectories byte-for-byte) ─────────────────

_ROLE_DESCRIPTIONS = {
    "dev": "You are a Developer. You write configs and propose fixes. Actions: view_config, edit_config, run_migration.",
    "sre": "You are an SRE. You investigate and diagnose. Actions: view_logs, view_pipeline.",
    "ops": "You are an Ops engineer. You manage production deployments. Actions: deploy, rollback, approve, abort.",
}

SYSTEM_PROMPT = textwrap.dedent("""
    You are an autonomous DevOps agent operating a CI/CD pipeline.
    Respond with EXACTLY ONE JSON object describing the next action.
    Do not include markdown code fences, prose, or explanation.

    Fields:
      action_type  (one of view_pipeline, view_logs, view_config, edit_config,
                   run_migration, deploy, rollback, approve, abort)
      role         (dev / sre / ops — must match the role assigned to you)
      service_name, target_version, config_edits, migration_name, reason
""").strip()


def build_prompt(obs: Dict[str, Any], role: str) -> str:
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
    actions_str = ", ".join(available) if available else "(none)"
    return (
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


# ─── Action parsing + sanitization ─────────────────────────────────────────────

def parse_completion(text: str) -> Dict[str, Any]:
    """Extract a PipelineAction-compatible dict from a model completion.
    Aggressively coerces config_edits to a valid List[ConfigEdit] or drops it.
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
    first = text.find("{")
    last = text.rfind("}")
    if first == -1 or last <= first:
        return fallback
    try:
        data = json.loads(text[first : last + 1])
        if not isinstance(data, dict) or "action_type" not in data:
            return fallback
        # Coerce config_edits to List[{key,value}] or drop entirely
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
        # Lowercase enum-valued fields
        for key in ("role", "action_type"):
            if isinstance(data.get(key), str):
                data[key] = data[key].lower()
        return data
    except Exception:
        return fallback


# ─── CLI ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="GRPO training for DevOps Pipeline Gym (clean)")
    p.add_argument("--model", default="unsloth/Qwen3-1.7B-bnb-4bit",
                   help="HF model id for the base model")
    p.add_argument("--env-url", default="http://localhost:8000")
    p.add_argument("--max-steps", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--num-generations", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=5e-6)
    p.add_argument("--max-completion-length", type=int, default=128)
    p.add_argument("--output-dir", default="./outputs/grpo_run")
    p.add_argument("--prompts-per-task", type=int, default=2)
    p.add_argument("--sft-adapter-path", default=None,
                   help="Path to SFT adapter (loaded as frozen prior); "
                        "pass 'none' or omit to train from raw base.")
    p.add_argument("--max-episode-steps", type=int, default=6)
    p.add_argument("--continuation-temperature", type=float, default=0.7)
    return p


# ─── Env client + dataset ─────────────────────────────────────────────────────

def _make_sync_client(env_url: str):
    """Return a sync EnvClient context manager."""
    from devops_pipeline_gym.client import DevopsPipelineEnv
    return DevopsPipelineEnv(base_url=env_url).sync()


def build_dataset(env_url: str, prompts_per_task: int = 2) -> List[Dict[str, str]]:
    """Sample initial-observation prompts from the env for the GRPO dataset."""
    tasks = ["clean_deploy", "broken_pipeline", "judgment_call",
             "cascading_failure", "capacity_crisis", "random_incident"]
    prompts: List[Dict[str, str]] = []
    for task in tasks:
        for seed_offset in range(prompts_per_task):
            os.environ["DEVOPS_TASK"] = task
            if task == "random_incident":
                os.environ["DEVOPS_SEED"] = str(6000 + seed_offset)
            with _make_sync_client(env_url) as client:
                result = client.reset()
                obs = result.observation.model_dump()
                role = obs.get("current_role", "sre")
                user = build_prompt(obs, role)
                # Stash task+seed in prompt so reward_function can reproduce
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


# ─── Multi-step rollout reward ────────────────────────────────────────────────

def _generate_continuation_action(model, tokenizer, system_prompt, user_prompt,
                                   max_new_tokens=128, temperature=0.7):
    """Generate ONE action string for steps 2..N of an episode rollout."""
    import torch
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    prompt_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt",
    ).to(model.device)
    with torch.no_grad():
        output = model.generate(
            prompt_ids, max_new_tokens=max_new_tokens, do_sample=True,
            temperature=temperature, top_p=0.9,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    completion_ids = output[0][prompt_ids.shape[1]:]
    return tokenizer.decode(completion_ids, skip_special_tokens=True)


def make_reward_function(env_url, model_ref, tokenizer_ref, system_prompt,
                          max_episode_steps=6, continuation_temperature=0.7,
                          continuation_max_tokens=128):
    """Closure-based multi-step reward: GRPO completion = step 1, model.generate for steps 2..N."""
    from devops_pipeline_gym.models import PipelineAction

    def reward_function(completions: List[str], prompts: List[str], **_kwargs) -> List[float]:
        rewards: List[float] = []
        for prompt_text, first_completion in zip(prompts, completions):
            meta = _extract_sentinel(prompt_text)
            task, seed = meta["task"], meta["seed"]
            try:
                os.environ["DEVOPS_TASK"] = task
                if task == "random_incident":
                    os.environ["DEVOPS_SEED"] = str(6000 + seed)

                try:
                    action1 = PipelineAction(**parse_completion(first_completion))
                except Exception as e:
                    logger.warning("step 1 parse failed: %s", str(e)[:80])
                    rewards.append(-1.0)
                    continue

                with _make_sync_client(env_url) as client:
                    client.reset()
                    try:
                        result = client.step(action1)
                    except Exception as e:
                        logger.warning("step 1 env error: %s", str(e)[:80])
                        rewards.append(-1.0)
                        continue
                    episode_reward = float(getattr(result, "reward", 0.0) or 0.0)
                    done = bool(getattr(result, "done", False))
                    current_obs = getattr(result, "observation", None)

                    step = 1
                    while not done and step < max_episode_steps:
                        if current_obs is None:
                            break
                        obs_dict = (current_obs.model_dump() if hasattr(current_obs, "model_dump")
                                    else current_obs if isinstance(current_obs, dict) else {})
                        if not obs_dict:
                            break
                        role = obs_dict.get("current_role") or "sre"
                        if hasattr(role, "value"):
                            role = role.value
                        user = build_prompt(obs_dict, str(role).lower())
                        try:
                            cont = _generate_continuation_action(
                                model_ref, tokenizer_ref, system_prompt, user,
                                max_new_tokens=continuation_max_tokens,
                                temperature=continuation_temperature,
                            )
                            action_n = PipelineAction(**parse_completion(cont))
                            step_result = client.step(action_n)
                            episode_reward += float(getattr(step_result, "reward", 0.0) or 0.0)
                            done = bool(getattr(step_result, "done", False))
                            current_obs = getattr(step_result, "observation", None)
                        except Exception as e:
                            logger.warning("step %d failed: %s", step + 1, str(e)[:80])
                            episode_reward -= 0.1
                        step += 1

                rewards.append(episode_reward)
            except Exception as e:
                logger.error("episode failed: %s: %s", type(e).__name__, str(e)[:150])
                rewards.append(-1.0)
        return rewards

    return reward_function


# ─── Curriculum telemetry (Phase J.7) ─────────────────────────────────────────

def poll_curriculum_progress(env_url: str, output_path: Path, step: int) -> None:
    """GET /curriculum_progress and append to JSONL log. Best-effort."""
    import time
    url = f"{env_url.rstrip('/')}/curriculum_progress"
    try:
        with urllib.request.urlopen(
            urllib.request.Request(url, headers={"Accept": "application/json"}),
            timeout=5,
        ) as resp:
            if resp.status != 200:
                return
            payload = json.loads(resp.read().decode("utf-8"))
        payload["step"] = step
        payload["timestamp"] = time.time()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
    except (urllib.error.URLError, OSError, ValueError) as e:
        print(f"[curriculum_poll] step={step} failed: {type(e).__name__}: {e}", flush=True)


# ─── Main: clean kube-sre-gym pattern ─────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    args = build_parser().parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "final").mkdir(parents=True, exist_ok=True)

    # GPU deps
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
    from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
    from trl import GRPOConfig, GRPOTrainer

    sft_loaded = bool(args.sft_adapter_path and args.sft_adapter_path.lower() != "none")

    # ── Model: pure HF + 4-bit BNB + SDPA attention ──────────────────────────
    logger.info("Loading base model: %s", args.model)
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
        attn_implementation="sdpa",
        device_map="cuda:0",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info("Model + tokenizer loaded with SDPA attention + 4-bit NF4")

    # ── k-bit prep + gradient checkpointing ──────────────────────────────────
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    logger.info("Model prepared for k-bit training (grad ckpt, use_reentrant=False)")

    # ── LoRA setup ───────────────────────────────────────────────────────────
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
        # Stack: load SFT first as 'sft_warmup' (frozen), then add 'default' (trainable)
        logger.info("Loading SFT adapter as frozen prior: %s", args.sft_adapter_path)
        model = PeftModel.from_pretrained(
            model, args.sft_adapter_path,
            adapter_name="sft_warmup", is_trainable=False,
        )
        model.add_adapter("default", lora_config)
        model.set_adapter("default")
        # Defense in depth: explicitly freeze sft_warmup, ensure default is trainable
        for name, param in model.named_parameters():
            if "sft_warmup" in name:
                param.requires_grad = False
            elif "lora_" in name and "default" in name:
                param.requires_grad = True
        logger.info("Stacked: sft_warmup (frozen) + default (trainable GRPO LoRA)")
    else:
        from peft import get_peft_model
        model = get_peft_model(model, lora_config)
        logger.info("Single trainable LoRA on raw base (no SFT prior)")

    # ── Trainable params sanity check ────────────────────────────────────────
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info("Trainable params: %s / %s (%.2f%%)",
                f"{trainable:,}", f"{total:,}", 100.0 * trainable / max(1, total))
    if trainable == 0:
        raise RuntimeError("No trainable parameters! Cannot run GRPO.")

    # ── Dataset: sample initial-observation prompts from env ─────────────────
    logger.info("Building dataset from env (prompts_per_task=%d)", args.prompts_per_task)
    prompts = build_dataset(args.env_url, prompts_per_task=args.prompts_per_task)
    logger.info("Dataset size: %d prompts", len(prompts))
    dataset = Dataset.from_list(prompts)

    # ── Reward function: closure over model + tokenizer ──────────────────────
    reward_fn = make_reward_function(
        env_url=args.env_url,
        model_ref=model,
        tokenizer_ref=tokenizer,
        system_prompt=SYSTEM_PROMPT,
        max_episode_steps=args.max_episode_steps,
        continuation_temperature=args.continuation_temperature,
        continuation_max_tokens=args.max_completion_length,
    )

    # ── GRPO config (kube-sre-gym pattern + DAPO settings) ──────────────────
    grpo_config = GRPOConfig(
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
        save_steps=10,
        max_steps=args.max_steps,
        beta=0.01,
        push_to_hub=False,
        report_to=["trackio"],
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        temperature=1.0,
        loss_type="dapo",
        mask_truncated_completions=True,
    )

    # ── Curriculum + reward CSV callbacks ───────────────────────────────────
    class _CurriculumProgressCallback(TrainerCallback):
        def __init__(self, env_url, output_path, every_n_steps=10):
            self.env_url = env_url
            self.output_path = output_path
            self.every_n_steps = max(1, int(every_n_steps))

        def on_step_end(self, args_, state, control, **_kwargs):
            step = int(getattr(state, "global_step", 0))
            if step > 0 and step % self.every_n_steps == 0:
                poll_curriculum_progress(self.env_url, self.output_path, step)

    import csv

    class _RewardCSVCallback(TrainerCallback):
        def __init__(self, csv_path):
            self.csv_path = csv_path
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(self.csv_path, "w", newline="", encoding="utf-8")
            self._writer = None
            self._fieldnames = []

        def on_log(self, args_, state, control, logs=None, **_kwargs):
            if not logs:
                return
            row = {"step": int(getattr(state, "global_step", 0)), **logs}
            if self._writer is None:
                self._fieldnames = list(row.keys())
                self._writer = csv.DictWriter(self._fh, fieldnames=self._fieldnames)
                self._writer.writeheader()
            else:
                for k in row.keys():
                    if k not in self._fieldnames:
                        self._fieldnames.append(k)
            self._writer.writerow({k: row.get(k, "") for k in self._fieldnames})
            self._fh.flush()

        def on_train_end(self, args_, state, control, **_kwargs):
            try:
                self._fh.close()
            except Exception:
                pass

    callbacks = [
        _CurriculumProgressCallback(args.env_url, out_dir / "curriculum_progress.jsonl"),
        _RewardCSVCallback(out_dir / "grpo_log.csv"),
    ]

    # ── Stock TRL GRPOTrainer ───────────────────────────────────────────────
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_fn],
        args=grpo_config,
        train_dataset=dataset,
        callbacks=callbacks,
    )

    logger.info("Starting GRPO training: max_steps=%d", args.max_steps)
    poll_curriculum_progress(args.env_url, out_dir / "curriculum_progress.jsonl", step=0)
    trainer.train()

    # ── Save final adapter ──────────────────────────────────────────────────
    final_dir = out_dir / "final"
    logger.info("Saving adapter to %s", final_dir)
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    # ── Reward curve PNG ────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        history = trainer.state.log_history if hasattr(trainer, "state") else []
        rewards = [e.get("reward") for e in history if "reward" in e]
        if rewards:
            plt.figure(figsize=(8, 4))
            plt.plot(range(1, len(rewards) + 1), rewards, marker="o")
            plt.xlabel("logging step")
            plt.ylabel("mean reward")
            plt.title(f"GRPO reward — {args.model} — {args.max_steps} steps")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(out_dir / "reward_curve.png", dpi=120)
            logger.info("Reward curve saved: %s", out_dir / "reward_curve.png")
    except Exception as e:
        logger.warning("reward_curve.png failed: %s", e)

    logger.info("Done. Adapter at %s", final_dir)


if __name__ == "__main__":
    main()
