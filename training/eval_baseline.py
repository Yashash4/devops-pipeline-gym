"""Run a model (HF hub id or local adapter) across all 6 tasks × N seeds.

Output JSON feeds generate_comparison_chart.py. Three execution modes:
  * ``--use-hf-router`` — send prompts to https://router.huggingface.co/v1
    via the openai-compatible client. No GPU required. Same HF_TOKEN as
    inference.py. Good for capturing baselines on a CPU-only host.
  * HF hub id (e.g. ``unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit``) — loads
    locally with transformers. Requires GPU for 4-bit; CPU works slowly.
  * Local adapter directory (e.g. ``./outputs/run1/final``) — loads the
    base model referenced in ``adapter_config.json`` then wraps with
    the LoRA adapter via PEFT.

This script never imports scenarios / engine directly, so the same eval
can be pointed at the HF Space by setting --env-url to the Space URL.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Allow running directly (python training/eval_baseline.py) OR as a module
# (python -m training.eval_baseline). When launched as a script, the
# parent dir isn't on sys.path → add it so `from training....` works.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

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


class _HFRouterAdapter:
    """HF Inference Router (OpenAI-compatible). No GPU required.

    Mirrors inference.py's pattern so baseline + trained eval share the
    same client wiring. Reads HF_TOKEN from env, defaults API_BASE_URL
    to HuggingFace's router.
    """

    def __init__(self, model_spec: str,
                 api_base_url: Optional[str] = None,
                 temperature: float = 0.0,
                 max_tokens: int = 300):
        self.model_spec = model_spec
        self.api_base_url = api_base_url or os.environ.get("API_BASE_URL") or "https://router.huggingface.co/v1"
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None

    def load(self):
        from openai import OpenAI  # late import
        api_key = os.environ.get("HF_TOKEN") or os.environ.get("API_KEY")
        if not api_key:
            raise RuntimeError(
                "--use-hf-router requires HF_TOKEN (or API_KEY) in env / .env"
            )
        self._client = OpenAI(base_url=self.api_base_url, api_key=api_key)
        return self

    def complete(self, system: str, user: str) -> str:
        if self._client is None:
            raise RuntimeError("Call .load() first")
        try:
            resp = self._client.chat.completions.create(
                model=self.model_spec,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=False,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            logger.error("HF router call failed (%s): %s", type(e).__name__, e)
            return ""


def _load_env_file(path: Path) -> None:
    """Tiny .env loader — no python-dotenv dep."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


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
        if result.done:
            done = True
            break

    # Success heuristic — from the last observation, count services still
    # healthy. pipeline_environment records success = final_health >= 50 and
    # no broke_healthy, but broke_healthy lives only in episode_history on
    # the server side; the EnvClient doesn't expose it. "Majority-healthy"
    # is a reasonable client-side proxy.
    final_obs_dict = obs.model_dump() if hasattr(obs, "model_dump") else obs
    final_services = final_obs_dict.get("services", []) or []
    healthy_count = sum(
        1 for s in final_services
        if (s.get("health") if isinstance(s, dict) else getattr(s, "health", None)) == "healthy"
    )
    total_services = max(len(final_services), 1)
    healthy_ratio = healthy_count / total_services
    success = healthy_ratio >= 0.6 and reward_sum > 0.0
    all_3_modes = len(roles_used) >= 3

    return {
        "task": task,
        "seed_offset": seed_offset,
        "steps": steps,
        "reward_sum": round(reward_sum, 4),
        "rewards_per_step": [round(x, 4) for x in rewards],
        "roles_used": sorted(roles_used),
        "num_roles_used": len(roles_used),
        "all_3_modes_used": all_3_modes,
        "done": done,
        "success": success,
        "healthy_ratio": round(healthy_ratio, 3),
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
        successes = [1 if v.get("success") else 0 for v in runs.values()]
        all3 = [1 if v.get("all_3_modes_used") else 0 for v in runs.values()]
        summary[task] = {
            "avg_reward": round(statistics.mean(rewards), 4) if rewards else 0.0,
            "std_reward": round(statistics.pstdev(rewards), 4) if len(rewards) > 1 else 0.0,
            "avg_steps": round(statistics.mean(steps), 2) if steps else 0.0,
            "avg_roles_used": round(statistics.mean(roles), 2) if roles else 0.0,
            "success_rate": round(statistics.mean(successes), 3) if successes else 0.0,
            "all_3_modes_hit_rate": round(statistics.mean(all3), 3) if all3 else 0.0,
            "n": len(runs),
        }
    return summary


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Baseline / trained-model eval for DevOps Pipeline Gym")
    p.add_argument("--model", required=True,
                   help="HF hub id OR path to a local adapter directory OR HF Router model name (with --use-hf-router)")
    p.add_argument("--env-url", default="http://localhost:8000")
    p.add_argument("--output", required=True, help="Path to write result JSON")
    p.add_argument("--n-seeds", type=int, default=3)
    p.add_argument("--use-hf-router", action="store_true",
                   help="Call model via HF Inference Router (openai-compatible). "
                        "Uses HF_TOKEN from env/.env. No GPU required.")
    p.add_argument("--api-base-url", default=None,
                   help="Override HF router URL (default https://router.huggingface.co/v1)")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Sampling temperature for --use-hf-router (default 0.0 greedy)")
    p.add_argument("--max-tokens", type=int, default=300,
                   help="Max completion tokens for --use-hf-router")
    args = p.parse_args()

    # Load .env for HF_TOKEN if present alongside the script tree.
    for candidate in (Path(__file__).resolve().parent.parent / ".env",
                      Path.cwd() / ".env"):
        _load_env_file(candidate)

    logger.info("Loading model: %s (hf_router=%s)", args.model, args.use_hf_router)
    if args.use_hf_router:
        adapter = _HFRouterAdapter(
            args.model,
            api_base_url=args.api_base_url,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        ).load()
    else:
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
                        "all_3_modes_used": False,
                        "done": False,
                        "success": False, "healthy_ratio": 0.0,
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
