"""Export ONE episode's per-step decisions to JSON for video production.

Phase J.6 (win-1st add): produces an input file for `render_replay.py`,
which draws the dependency graph at each step colored by service health.
The before/after side-by-side video in Phase O is built from one
baseline replay + one trained replay of the same task+seed.

Reuses model-loading + observation-health logic from `eval_baseline.py`
(no refactor — same imports, same helpers).

Example invocations:
    # Untrained baseline replay:
    python training/export_replay.py \\
      --model unsloth/Qwen3-1.7B-bnb-4bit \\
      --task judgment_call --seed 3003 \\
      --output-json outputs/replay_baseline_judgment.json

    # Trained replay (same seed → same env state, fair comparison):
    python training/export_replay.py \\
      --model unsloth/Qwen3-1.7B-bnb-4bit \\
      --adapter-path outputs/run1/final \\
      --task judgment_call --seed 3003 \\
      --output-json outputs/replay_trained_judgment.json

    # HF Router (no GPU required — works from CPU host):
    python training/export_replay.py \\
      --model Qwen/Qwen2.5-72B-Instruct --use-hf-router \\
      --task judgment_call --seed 3003 \\
      --output-json outputs/replay_hfrouter_judgment.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Allow running directly OR as a module (mirrors eval_baseline.py).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from devops_pipeline_gym.client import DevopsPipelineEnv
from devops_pipeline_gym.models import ActionType, PipelineAction, Role
from training.eval_baseline import (
    MAX_STEPS_PER_EPISODE,
    _HFRouterAdapter,
    _ModelAdapter,
    _extract_system_health,
    _load_env_file,
)
from training.grpo_train import SYSTEM_PROMPT, build_prompt, parse_completion

logger = logging.getLogger("export_replay")


def _capture_service_snapshot(obs) -> List[Dict[str, Any]]:
    """Pull per-service status from observation as a list of plain dicts.

    Tolerant of both Pydantic ServiceStatus models and dict-form services.
    Health enum is normalized to its string value so render_replay can
    bucket without re-importing the enum class.
    """
    services = obs.services if hasattr(obs, "services") else obs.get("services", [])
    out: List[Dict[str, Any]] = []
    for s in services:
        get = (lambda k, default=None: getattr(s, k, default)) if not isinstance(s, dict) else s.get
        h = get("health")
        h_str = h.value if hasattr(h, "value") else (str(h).lower() if h is not None else "unknown")
        out.append({
            "name": get("name"),
            "health": h_str,
            "cpu_percent": get("cpu_percent", 0.0),
            "memory_percent": get("memory_percent", 0.0),
            "request_latency_ms": get("request_latency_ms", 0.0),
            "error_rate": get("error_rate", 0.0),
            "current_version": get("current_version", ""),
        })
    return out


def run_replay(client, model_adapter, task: str, seed: int) -> Dict[str, Any]:
    """Run one episode and capture decisions for replay rendering."""
    # Pin task + seed via reset kwargs so the replay is reproducible.
    # (eval_baseline uses os.environ; we use kwargs because the WS client
    # forwards them to the env's reset() and curriculum is bypassed.)
    result = client.reset(task=task, seed=seed)
    obs = result.observation

    steps: List[Dict[str, Any]] = []
    total_reward = 0.0
    done = False

    for step_num in range(1, MAX_STEPS_PER_EPISODE + 1):
        obs_dict = obs.model_dump() if hasattr(obs, "model_dump") else obs
        role = obs_dict.get("current_role", "sre")
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
        total_reward += r
        done = bool(result.done)

        steps.append({
            "step": step_num,
            "action": action.model_dump(),
            "reward": round(r, 4),
            "system_health": round(_extract_system_health(obs), 1),
            "services": _capture_service_snapshot(obs),
            "current_role": (
                obs.current_role.value if hasattr(getattr(obs, "current_role", None), "value")
                else (obs_dict.get("current_role") if isinstance(obs_dict, dict) else None)
            ),
            "last_action_result": getattr(obs, "last_action_result", None),
            "last_action_error": getattr(obs, "last_action_error", None),
            "done": done,
        })

        if done:
            break

    final_health = _extract_system_health(obs)
    last_reward = steps[-1]["reward"] if steps else 0.0
    succeeded = bool(done and last_reward > 0)

    return {
        "model": model_adapter.model_spec,
        "adapter_path": getattr(model_adapter, "_adapter_path", None),
        "task": task,
        "seed": seed,
        "trained": getattr(model_adapter, "_adapter_path", None) is not None,
        "total_reward": round(total_reward, 4),
        "final_health": round(final_health, 1),
        "succeeded": succeeded,
        "steps": steps,
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Export one episode's decisions to JSON for replay rendering.")
    p.add_argument("--model", required=True,
                   help="HF hub id OR local adapter dir OR HF Router model name (with --use-hf-router)")
    p.add_argument("--adapter-path", default=None,
                   help="Optional LoRA adapter dir; baseline replays leave this unset")
    p.add_argument("--task", required=True,
                   help="Task name (clean_deploy, broken_pipeline, judgment_call, "
                        "cascading_failure, capacity_crisis, random_incident)")
    p.add_argument("--seed", type=int, required=True,
                   help="Episode seed for reproducibility (matched between baseline & trained)")
    p.add_argument("--env-url", default="http://localhost:8000")
    p.add_argument("--output-json", required=True, help="Path to write replay JSON")
    p.add_argument("--use-hf-router", action="store_true",
                   help="Call model via HF Inference Router (no GPU). Uses HF_TOKEN.")
    p.add_argument("--api-base-url", default=None)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=300)
    args = p.parse_args()

    # Load .env for HF_TOKEN.
    for candidate in (Path(__file__).resolve().parent.parent / ".env",
                      Path.cwd() / ".env"):
        _load_env_file(candidate)

    logger.info("Loading model: %s (adapter=%s, hf_router=%s)",
                args.model, args.adapter_path, args.use_hf_router)
    if args.use_hf_router:
        adapter = _HFRouterAdapter(
            args.model, api_base_url=args.api_base_url,
            temperature=args.temperature, max_tokens=args.max_tokens,
        ).load()
    else:
        # _ModelAdapter takes the spec; if --adapter-path is given, we treat
        # IT as the spec (eval_baseline auto-detects adapter dirs by the
        # presence of adapter_config.json). Stash the original --model on
        # the adapter so trained-vs-baseline labelling is preserved.
        spec = args.adapter_path or args.model
        adapter = _ModelAdapter(spec).load()
        adapter._adapter_path = args.adapter_path  # type: ignore[attr-defined]
        adapter._base_model = args.model           # type: ignore[attr-defined]

    logger.info("Running replay: task=%s seed=%d", args.task, args.seed)
    with DevopsPipelineEnv(base_url=args.env_url).sync() as client:
        replay = run_replay(client, adapter, args.task, args.seed)

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(replay, indent=2), encoding="utf-8")
    logger.info("Wrote %s (%d steps, total_reward=%.3f, succeeded=%s)",
                out_path, len(replay["steps"]), replay["total_reward"], replay["succeeded"])


if __name__ == "__main__":
    main()
