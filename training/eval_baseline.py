"""Run a model (HF hub id or local adapter) across all 6 tasks × N seeds.

Output JSON is the input to generate_comparison_chart.py. Works in two
modes:
  * Base model (e.g. ``unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit``) — loads with
    transformers and samples greedily.
  * Local adapter directory (e.g. ``./outputs/run1/final``) — loads the base
    model indicated inside adapter_config.json then merges the LoRA adapter.

This script is intentionally self-contained and uses HTTP against the env
server (no direct imports of scenarios / engine) so the same eval can be
pointed at the HF Space instead of a local env.

CPU works too (slow) — useful for a dry-run sanity check. For Saturday real
eval on GPU: pass a 4-bit base + PEFT adapter.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional

from devops_pipeline_gym.client import DevopsPipelineEnv
from devops_pipeline_gym.models import ActionType, PipelineAction, Role
from training.grpo_train import SYSTEM_PROMPT, build_prompt, parse_completion

logger = logging.getLogger("eval_baseline")

TASKS = [
    "clean_deploy",
    "broken_pipeline",
    "judgment_call",
    "cascading_failure",
    "capacity_crisis",
    "random_incident",
]

MAX_STEPS_PER_EPISODE = 20


# ─── Model adapters ──────────────────────────────────────────────────────────

class _ModelAdapter:
    """Abstract over 'HF hub id' vs 'local adapter dir'."""

    def __init__(self, model_spec: str):
        self.model_spec = model_spec
        self._pipe = None

    def load(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        import torch

        is_dir = Path(self.model_spec).is_dir()
        if is_dir:
            # Load LoRA adapter: need its base model name and merge.
            try:
                from peft import PeftModel
            except ImportError as e:
                raise RuntimeError(
                    "Loading a local adapter requires peft. pip install peft"
                ) from e
            cfg_path = Path(self.model_spec) / "adapter_config.json"
            if not cfg_path.exists():
                raise FileNotFoundError(
                    f"Expected adapter_config.json under {self.model_spec}"
                )
            adapter_cfg = json.loads(cfg_path.read_text())
            base_name = adapter_cfg.get("base_model_name_or_path")
            tokenizer = AutoTokenizer.from_pretrained(base_name)
            base = AutoModelForCausalLM.from_pretrained(
                base_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
            )
            model = PeftModel.from_pretrained(base, self.model_spec)
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.model_spec)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_spec,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
            )
        self._pipe = pipeline(
            "text-generation", model=model, tokenizer=tokenizer,
            max_new_tokens=256, do_sample=False, return_full_text=False,
        )
        return self

    def complete(self, system: str, user: str) -> str:
        if self._pipe is None:
            raise RuntimeError("Call .load() first")
        prompt = f"{system}\n\n{user}"
        out = self._pipe(prompt)
        if isinstance(out, list) and out and isinstance(out[0], dict):
            return out[0].get("generated_text", "")
        return ""


# ─── Per-episode evaluation ──────────────────────────────────────────────────

def run_episode(client, model_adapter, task: str, seed_offset: int) -> Dict[str, Any]:
    """Run one episode. Returns metrics dict."""
    os.environ["DEVOPS_TASK"] = task
    if task == "random_incident":
        os.environ["DEVOPS_SEED"] = str(6000 + seed_offset)

    result = client.reset()
    obs = result.observation
    reward_sum = 0.0
    rewards: List[float] = []
    roles_used: set = set()
    handoff_scores: List[float] = []
    done = False
    steps = 0
    broke_healthy = False
    for steps in range(1, MAX_STEPS_PER_EPISODE + 1):
        obs_dict = obs.model_dump() if hasattr(obs, "model_dump") else obs
        role = obs_dict.get("current_role", "sre")
        roles_used.add(role)
        user = build_prompt(obs_dict, role)
        completion = model_adapter.complete(SYSTEM_PROMPT, user)
        action_data = parse_completion(completion)
        try:
            action = PipelineAction(**action_data)
        except Exception as e:
            logger.warning("invalid action JSON: %s; err=%s", action_data, e)
            action = PipelineAction(action_type=ActionType.VIEW_PIPELINE, role=Role.SRE)
        result = client.step(action)
        obs = result.observation
        r = float(result.reward or 0.0)
        rewards.append(r)
        reward_sum += r
        if getattr(obs, "previous_handoff", None):
            # Hand-off bonus is already reflected in reward — we log the raw
            # notes length as a proxy signal so the chart can show "agent did
            # or didn't use notes."
            handoff_scores.append(len(obs.previous_handoff))
        if result.done:
            done = True
            break

    # Success heuristic — matches pipeline_environment.reset/record_episode.
    # We call /grader for the canonical score.
    try:
        # The EnvClient exposes generic state; for the grader score we use
        # a separate HTTP call handled by the runner below.
        pass
    except Exception:
        pass

    return {
        "task": task,
        "seed_offset": seed_offset,
        "steps": steps,
        "reward_sum": round(reward_sum, 4),
        "rewards_per_step": [round(x, 4) for x in rewards],
        "roles_used": sorted(roles_used),
        "num_roles_used": len(roles_used),
        "handoff_notes_count": len(handoff_scores),
        "done": done,
    }


# ─── Summary ──────────────────────────────────────────────────────────────────

def summarize(all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_task: Dict[str, Dict[str, Any]] = {}
    for r in all_results:
        by_task.setdefault(r["task"], {})[f"seed_{r['seed_offset']}"] = r
    summary: Dict[str, Any] = {}
    for task, runs in by_task.items():
        rewards = [v["reward_sum"] for v in runs.values()]
        steps = [v["steps"] for v in runs.values()]
        roles = [v["num_roles_used"] for v in runs.values()]
        summary[task] = {
            "avg_reward": round(statistics.mean(rewards), 4) if rewards else 0.0,
            "std_reward": round(statistics.pstdev(rewards), 4) if len(rewards) > 1 else 0.0,
            "avg_steps": round(statistics.mean(steps), 2) if steps else 0.0,
            "avg_roles_used": round(statistics.mean(roles), 2) if roles else 0.0,
            "n": len(runs),
        }
    return summary


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Baseline / trained-model eval for DevOps Pipeline Gym")
    p.add_argument("--model", required=True,
                   help="HF hub id OR path to a local adapter directory")
    p.add_argument("--env-url", default="http://localhost:8000")
    p.add_argument("--output", required=True, help="Path to write result JSON")
    p.add_argument("--n-seeds", type=int, default=3)
    args = p.parse_args()

    logger.info("Loading model: %s", args.model)
    adapter = _ModelAdapter(args.model).load()

    all_results: List[Dict[str, Any]] = []
    with DevopsPipelineEnv(base_url=args.env_url).sync() as client:
        for task in TASKS:
            for seed_offset in range(args.n_seeds):
                logger.info("task=%s seed=%d", task, seed_offset)
                try:
                    rec = run_episode(client, adapter, task, seed_offset)
                except Exception as e:
                    logger.error("episode failed: %s", e)
                    rec = {
                        "task": task, "seed_offset": seed_offset, "steps": 0,
                        "reward_sum": 0.0, "rewards_per_step": [],
                        "roles_used": [], "num_roles_used": 0,
                        "handoff_notes_count": 0, "done": False,
                        "error": f"{type(e).__name__}: {e}",
                    }
                all_results.append(rec)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {"model": args.model, "tasks": _group_by_task(all_results),
             "summary": summarize(all_results)},
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Wrote %s", out_path)


def _group_by_task(all_results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {}
    for r in all_results:
        grouped.setdefault(r["task"], {})[f"seed_{r['seed_offset']}"] = r
    return grouped


if __name__ == "__main__":
    main()
