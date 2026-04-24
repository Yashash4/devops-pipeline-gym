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
      service_name, target_version, config_edits, migration_name, reason,
      handoff_notes (optional; a short diagnosis or hand-off note)
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
    prev_handoff = obs.get("previous_handoff") or "none"
    last_result = obs.get("last_action_result") or "none"

    user = textwrap.dedent(
        f"""
        ROLE: {role_desc}

        TASK: {obs.get('task_description', '')}
        GOAL: {obs.get('goal', '')}

        CURRENT SERVICES:
        {service_lines}

        LAST ACTION RESULT: {last_result}
        PREVIOUS HANDOFF NOTES: {prev_handoff}

        AVAILABLE ACTIONS: {', '.join(available) if available else '(none)'}

        Respond with ONE JSON action.
        """
    ).strip()
    return user


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


def make_reward_function(env_url: str):
    """Return a GRPO-compatible reward_function bound to a server URL.

    Signature matches trl.GRPOTrainer's reward_funcs contract:
        reward_function(completions, prompts, **kwargs) -> List[float]
    """
    from devops_pipeline_gym.models import PipelineAction

    def reward_function(completions: List[str], prompts: List[str], **_kwargs) -> List[float]:
        rewards: List[float] = []
        for prompt_text, completion in zip(prompts, completions):
            meta = _extract_sentinel(prompt_text)
            task = meta["task"]
            seed = meta["seed"]
            action_data = parse_completion(completion)
            try:
                action = PipelineAction(**action_data)
            except Exception as e:
                logger.warning("parse_completion produced invalid action: %s; err=%s",
                               action_data, e)
                rewards.append(-0.10)
                continue
            try:
                os.environ["DEVOPS_TASK"] = task
                if task == "random_incident":
                    os.environ["DEVOPS_SEED"] = str(6000 + seed)
                with _make_sync_client(env_url) as client:
                    client.reset()
                    result = client.step(action)
                    r = float(result.reward or 0.0)
                    rewards.append(r)
            except Exception as e:
                logger.error("env rollout failed (%s): %s", type(e).__name__, e)
                rewards.append(-0.10)
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

    logger.info("Loading base model: %s", args.model)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
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

    logger.info("Building dataset from env rollouts (prompts_per_task=%d)", args.prompts_per_task)
    prompts = build_dataset(args.env_url, prompts_per_task=args.prompts_per_task)
    logger.info("Dataset size: %d prompts", len(prompts))
    dataset = Dataset.from_list(prompts)

    reward_fn = make_reward_function(args.env_url)

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
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_fn],
        args=grpo_config,
        train_dataset=dataset,
    )

    logger.info("Starting GRPO training: max_steps=%d", args.max_steps)
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
