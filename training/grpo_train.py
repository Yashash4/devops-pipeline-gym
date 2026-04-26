"""GRPO training for DevOps Pipeline Gym (clean single-step kube-sre-gym pattern).

Pure HF Transformers + PEFT + TRL. No Unsloth, no vLLM, no multi-step
rollout inside the reward function (calling model.generate during a
training forward pass is structurally broken: PEFT attribute delegation,
grad-ckpt-vs-use-cache conflict, autograd graph contamination).

Architecture:
  base = AutoModelForCausalLM (4-bit BNB NF4 + SDPA attention)
  + SFT adapter loaded as the trainable LoRA (is_trainable=True) — SFT
    weights are active during generation AND continue-trainable by GRPO
  -> stock trl.GRPOTrainer with single-step env-reward function:
       completion -> parse action -> env.reset -> env.step -> reward

Usage:
  python training/grpo_train.py \\
    --model unsloth/Qwen3-1.7B-bnb-4bit \\
    --env-url http://localhost:8000 \\
    --sft-adapter-path /workspace/sft_adapter/final \\
    --max-steps 20 --num-generations 2 --batch-size 1 --grad-accum 4 \\
    --max-completion-length 128 --learning-rate 5e-6 \\
    --output-dir /workspace/grpo_output
"""

from __future__ import annotations

# ─── mergekit stub (MUST run before TRL import) ─────────────────────────────
# TRL 0.29's trainer/callbacks.py top-level imports mergekit_utils, which
# top-level imports `from mergekit.config import MergeConfiguration`. The
# real mergekit package pins pydantic<=2.10, but openenv-core[core]>=0.2.2
# requires pydantic>=2.11.7 (via fastmcp), so the two can't co-install.
# We never use mergekit features (no model merging in our training loop),
# so a stub that satisfies any attribute access is safe. __getattr__ on the
# module returns a no-op class for any name.
import sys as _sys
import types as _types


class _StubMergekitModule(_types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        cls = type(name, (), {"__init__": lambda self, *a, **kw: None})
        setattr(self, name, cls)
        return cls


for _mod_name in (
    "mergekit",
    "mergekit.config",
    "mergekit.merge",
    "mergekit.architecture",
    "mergekit.io",
    "mergekit.options",
):
    if _mod_name not in _sys.modules:
        _sys.modules[_mod_name] = _StubMergekitModule(_mod_name)

# ────────────────────────────────────────────────────────────────────────────

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
                   help="Path to SFT adapter (loaded as the trainable LoRA so "
                        "SFT weights are both active during generation and "
                        "further trained by GRPO); pass 'none' or omit to "
                        "train from raw base.")
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

def make_reward_function(env_url):
    """Single-step reward function.

    For each (prompt, completion) pair: parse the completion as a
    PipelineAction, reset the env to the prompt's task/seed, take ONE
    env.step, and return that step's reward. No model.generate calls
    inside this closure — that's the structural mismatch that broke
    every multi-step variant. GRPO sees clean (prompt, completion,
    reward) triples and learns to maximize the per-step env reward.
    """
    from devops_pipeline_gym.models import PipelineAction

    def reward_function(completions: List[str], prompts: List[str], **_kwargs) -> List[float]:
        rewards: List[float] = []
        for prompt_text, completion in zip(prompts, completions):
            meta = _extract_sentinel(prompt_text)
            task, seed = meta["task"], meta["seed"]
            try:
                os.environ["DEVOPS_TASK"] = task
                if task == "random_incident":
                    os.environ["DEVOPS_SEED"] = str(6000 + seed)

                try:
                    action = PipelineAction(**parse_completion(completion))
                except Exception as e:
                    logger.warning("parse failed: %s: %r", type(e).__name__, e)
                    rewards.append(-1.0)
                    continue

                with _make_sync_client(env_url) as client:
                    client.reset()
                    try:
                        result = client.step(action)
                    except Exception as e:
                        logger.warning("env step error: %s: %r", type(e).__name__, e)
                        rewards.append(-1.0)
                        continue
                    rewards.append(float(getattr(result, "reward", 0.0) or 0.0))
            except Exception as e:
                logger.error("episode failed: %s: %r", type(e).__name__, e)
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

    from peft import get_peft_model
    if sft_loaded:
        # Load SFT adapter as THE trainable LoRA. PEFT activates one adapter
        # at a time, so the SFT weights are both active during generation
        # AND further trained by GRPO. No merge (lossy on 4-bit base, the
        # rounding-error warning destroys the SFT prior), no stacking (only
        # one adapter active during forward, which left fresh-init GRPO LoRA
        # in the active path producing 128-token garbage).
        logger.info("Loading SFT adapter as the trainable GRPO LoRA: %s",
                    args.sft_adapter_path)
        model = PeftModel.from_pretrained(
            model, args.sft_adapter_path, is_trainable=True,
        )
    else:
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

    # ── Reward function: single-step env reward ──────────────────────────────
    reward_fn = make_reward_function(env_url=args.env_url)

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
        # mask_truncated_completions=False: SFT was trained on JSON-only
        # trajectories (no EOS marker), so the model produces valid actions
        # but can't stop, padding to max_completion_length. With masking
        # enabled, every clipped completion is dropped from the loss
        # (loss=0, grad_norm=0). The completions DO contain parseable
        # actions and the per-step env reward differentiates them — we
        # need gradient to flow on those rewards.
        mask_truncated_completions=False,
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
