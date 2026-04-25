"""Generate a before/after PNG from two eval_baseline.py JSON outputs.

Two side-by-side subplots in one PNG:
  Left  — avg_reward per task (higher is better). Error bars: std across seeds.
  Right — avg_steps_to_recovery per task (lower is better; Phase J.5 metric).
          Tasks that always start healthy (e.g. clean_deploy) report 0 with a
          0-recovery-episode count and are dropped from the right subplot.

Side-by-side baseline (untrained) vs trained. Saved at --output.

Designed to be non-interactive and Agg-backend so it runs on headless CI.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger("chart")

CANONICAL_TASK_ORDER = [
    "clean_deploy",
    "broken_pipeline",
    "judgment_call",
    "cascading_failure",
    "capacity_crisis",
    "random_incident",
]


def _load(path: str) -> Dict[str, Any]:
    data = json.loads(Path(path).read_text())
    if "summary" not in data:
        raise ValueError(f"{path}: missing 'summary' key (not an eval_baseline JSON?)")
    return data


def _series(summary: Dict[str, Dict[str, float]]):
    avgs: List[float] = []
    stds: List[float] = []
    tasks: List[str] = []
    for task in CANONICAL_TASK_ORDER:
        if task not in summary:
            continue
        tasks.append(task)
        avgs.append(float(summary[task].get("avg_reward", 0.0)))
        stds.append(float(summary[task].get("std_reward", 0.0)))
    return tasks, avgs, stds


def _recovery_series(summary: Dict[str, Dict[str, float]]):
    """Per-task avg_steps_to_recovery, dropping tasks with 0 recovery episodes
    (i.e. tasks whose every episode started already-healthy — clean_deploy)."""
    tasks: List[str] = []
    avgs: List[float] = []
    for task in CANONICAL_TASK_ORDER:
        if task not in summary:
            continue
        ep_count = int(summary[task].get("recovery_episodes_count", 0))
        if ep_count == 0:
            continue  # task starts healthy in every seed; skip to avoid 0-bars
        tasks.append(task)
        avgs.append(float(summary[task].get("avg_steps_to_recovery", 0.0)))
    return tasks, avgs


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    p = argparse.ArgumentParser(description="Before/after comparison chart")
    p.add_argument("--baseline", required=True, help="JSON from untrained model eval")
    p.add_argument("--trained", required=True, help="JSON from trained adapter eval")
    p.add_argument("--output", required=True, help="PNG output path")
    p.add_argument("--title", default="Baseline vs Trained — average reward per task")
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    baseline_json = _load(args.baseline)
    trained_json = _load(args.trained)

    base_tasks, base_avgs, base_stds = _series(baseline_json["summary"])
    trained_tasks, trained_avgs, trained_stds = _series(trained_json["summary"])

    # Align x-axis to the union of both models' tasks (canonical order).
    tasks = [t for t in CANONICAL_TASK_ORDER if t in base_tasks or t in trained_tasks]
    base_map = dict(zip(base_tasks, zip(base_avgs, base_stds)))
    trained_map = dict(zip(trained_tasks, zip(trained_avgs, trained_stds)))

    b_y = [base_map.get(t, (0.0, 0.0))[0] for t in tasks]
    b_e = [base_map.get(t, (0.0, 0.0))[1] for t in tasks]
    t_y = [trained_map.get(t, (0.0, 0.0))[0] for t in tasks]
    t_e = [trained_map.get(t, (0.0, 0.0))[1] for t in tasks]

    x = np.arange(len(tasks))
    width = 0.38

    base_label = f"baseline: {Path(baseline_json.get('model', 'base')).name or baseline_json.get('model', 'base')}"
    trained_label = f"trained: {Path(trained_json.get('model', 'trained')).name or trained_json.get('model', 'trained')}"

    fig, (ax, ax_rec) = plt.subplots(1, 2, figsize=(15, 5))

    # ── Left subplot: avg reward per task (higher is better) ──────────────────
    ax.bar(x - width / 2, b_y, width, yerr=b_e, capsize=3,
           label=base_label, color="#9aa6b2")
    ax.bar(x + width / 2, t_y, width, yerr=t_e, capsize=3,
           label=trained_label, color="#3a86ff")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=20, ha="right")
    ax.set_ylabel("avg reward (summed over episode)")
    ax.set_title(args.title)
    ax.legend(loc="best", fontsize=9)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(axis="y", alpha=0.3)

    # ── Right subplot: steps to recovery per task (lower is better) ──────────
    base_rec_tasks, base_rec_avgs = _recovery_series(baseline_json["summary"])
    trained_rec_tasks, trained_rec_avgs = _recovery_series(trained_json["summary"])
    rec_tasks = [t for t in CANONICAL_TASK_ORDER if t in base_rec_tasks or t in trained_rec_tasks]
    base_rec_map = dict(zip(base_rec_tasks, base_rec_avgs))
    trained_rec_map = dict(zip(trained_rec_tasks, trained_rec_avgs))
    b_r = [base_rec_map.get(t, 0.0) for t in rec_tasks]
    t_r = [trained_rec_map.get(t, 0.0) for t in rec_tasks]
    xr = np.arange(len(rec_tasks))
    if rec_tasks:
        ax_rec.bar(xr - width / 2, b_r, width, label=base_label, color="#9aa6b2")
        ax_rec.bar(xr + width / 2, t_r, width, label=trained_label, color="#3a86ff")
        ax_rec.set_xticks(xr)
        ax_rec.set_xticklabels(rec_tasks, rotation=20, ha="right")
        ax_rec.set_ylabel("avg steps to recovery (lower is better)")
        ax_rec.set_title("Steps to Recovery — first step where mean health ≥ 80")
        ax_rec.legend(loc="best", fontsize=9)
        ax_rec.grid(axis="y", alpha=0.3)
    else:
        ax_rec.axis("off")
        ax_rec.text(0.5, 0.5, "No degraded-start episodes recorded\n(all tasks began healthy in this run)",
                    ha="center", va="center", transform=ax_rec.transAxes, fontsize=10, color="#666")

    fig.tight_layout()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    sys.exit(main())
