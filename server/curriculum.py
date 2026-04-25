# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Round 2 — Curriculum controller for adaptive task selection.

Tracks mastery at two granularities:
  - Per-task        (clean_deploy, broken_pipeline, judgment_call, ...)
  - Per-failure-type (config_error, capacity_limit, memory_leak, ...)

Picks the next scenario based on overall per-task mastery:
  overall < 0.3  -> easy candidates   (clean_deploy, broken_pipeline),   seed 1
  0.3 <= < 0.6  -> medium candidates (broken_pipeline, cascading,         seed 20+fixed_offset
                                       capacity_crisis)
  overall >= 0.6 -> hard candidates   (judgment_call, random_incident),    seed 60+fixed_offset

Signals plateau (last 10 rewards std < 0.05) -> returns ("adversarial", None),
which pipeline_environment.reset() will route to the adversarial designer
in Phase 5. On designer failure the caller is expected to fall back to
`random_incident` with a hard seed.

This module is STANDALONE: no imports from pipeline_environment / engine /
graders / rewards. Integration happens in Phase 5.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Deterministic per-task seed offsets within each tier. Required because
# CLAUDE.md forbids hash() for seeding (Python's hash() is randomized per
# process, breaking reproducibility). Offsets must stay within [0, 39] so
# medium seeds land in [20, 60) and hard seeds in [60, 100).
_TASK_SEED_OFFSET: Dict[str, int] = {
    "clean_deploy": 0,
    "broken_pipeline": 5,
    "judgment_call": 10,
    "cascading_failure": 15,
    "capacity_crisis": 20,
    "random_incident": 25,
}

_DEFAULT_ALL_TASKS: List[str] = [
    "clean_deploy",
    "broken_pipeline",
    "judgment_call",
    "cascading_failure",
    "capacity_crisis",
    "random_incident",
]

_DEFAULT_FAILURE_TYPES: List[str] = [
    "config_error",
    "degraded_performance",
    "capacity_limit",
    "memory_leak",
    "certificate_expiry",
]


@dataclass
class MasteryTracker:
    """Accumulates per-task and per-failure success ratios + reward history."""

    # task_name -> (successes, total_attempts)
    per_task: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    # failure_type -> (successes, total_attempts)
    per_failure: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    # Ring buffer of final episode rewards (plateau signal source).
    recent_rewards: List[float] = field(default_factory=list)
    # Cap on ring buffer length.
    max_recent: int = 20
    # Window size + variance threshold for plateau detection.
    plateau_window: int = 10
    plateau_var_threshold: float = 0.0025  # std < 0.05

    # --- mutation -------------------------------------------------------------

    def record_episode(
        self,
        task: str,
        failure_type: Optional[str],
        success: bool,
        final_reward: float,
    ) -> None:
        """Record the outcome of one episode."""
        s, t = self.per_task.get(task, (0, 0))
        self.per_task[task] = (s + (1 if success else 0), t + 1)

        if failure_type:
            fs, ft = self.per_failure.get(failure_type, (0, 0))
            self.per_failure[failure_type] = (fs + (1 if success else 0), ft + 1)

        self.recent_rewards.append(float(final_reward))
        if len(self.recent_rewards) > self.max_recent:
            self.recent_rewards.pop(0)

    # --- queries --------------------------------------------------------------

    def task_mastery(self, task: str) -> float:
        s, t = self.per_task.get(task, (0, 0))
        return s / t if t > 0 else 0.0

    def failure_mastery(self, failure_type: str) -> float:
        s, t = self.per_failure.get(failure_type, (0, 0))
        return s / t if t > 0 else 0.0

    def is_plateau(self) -> bool:
        """True iff last `plateau_window` rewards have variance < threshold."""
        if len(self.recent_rewards) < self.plateau_window:
            return False
        window = self.recent_rewards[-self.plateau_window:]
        avg = sum(window) / len(window)
        var = sum((r - avg) ** 2 for r in window) / len(window)
        return var < self.plateau_var_threshold


class CurriculumController:
    """Picks the next task / seed hint from current mastery."""

    # Tier boundaries (tunable).
    EASY_CUTOFF = 0.3
    MEDIUM_CUTOFF = 0.6

    def __init__(
        self,
        all_tasks: Optional[List[str]] = None,
        failure_types: Optional[List[str]] = None,
    ) -> None:
        self.tracker = MasteryTracker()
        self.all_tasks = list(all_tasks) if all_tasks is not None else list(_DEFAULT_ALL_TASKS)
        self.failure_types = list(failure_types) if failure_types is not None else list(_DEFAULT_FAILURE_TYPES)

    # --- primary API ----------------------------------------------------------

    def pick_task(self) -> Tuple[str, Optional[int]]:
        """Return (task_name, seed_hint).

        Special: returns ("adversarial", None) when the agent has plateaued —
        the environment is expected to ask the adversarial designer for a
        novel scenario targeting weak failure types.
        """
        if self.tracker.is_plateau():
            return ("adversarial", None)

        overall = self._overall_mastery()
        if overall < self.EASY_CUTOFF:
            return self._pick_easy()
        if overall < self.MEDIUM_CUTOFF:
            return self._pick_medium()
        return self._pick_hard()

    def dump_progress(self) -> Dict[str, object]:
        """Read-only mastery snapshot for external observation.

        Used by Phase J.7 /curriculum_progress endpoint and Phase M's GRPO
        polling callback. Does NOT mutate curriculum state. Returns valid
        empty dicts when no episode has been recorded yet.
        """
        per_task: Dict[str, Dict[str, float]] = {}
        for t, (s, a) in self.tracker.per_task.items():
            per_task[t] = {
                "successes": s,
                "attempts": a,
                "rate": (s / a) if a > 0 else 0.0,
            }
        per_failure: Dict[str, Dict[str, float]] = {}
        for f, (s, a) in self.tracker.per_failure.items():
            per_failure[f] = {
                "successes": s,
                "attempts": a,
                "rate": (s / a) if a > 0 else 0.0,
            }
        recent = list(self.tracker.recent_rewards)
        return {
            "per_task": per_task,
            "per_failure": per_failure,
            "recent_rewards_mean": (sum(recent) / len(recent)) if recent else 0.0,
            "recent_rewards_count": len(recent),
            "overall_mastery": self._overall_mastery(),
            "is_plateau": self.tracker.is_plateau(),
        }

    def get_weak_failure_types(self, top_n: int = 2) -> List[str]:
        """Return failure types with lowest mastery, lowest first.

        Untried failure types count as 0.0 mastery and therefore rank among
        the weakest — correct behaviour since the agent has no data there.
        Feeds the adversarial designer in Phase 5.
        """
        if top_n <= 0:
            return []
        scored = [
            (ft, self.tracker.failure_mastery(ft)) for ft in self.failure_types
        ]
        scored.sort(key=lambda x: x[1])
        return [ft for ft, _ in scored[:top_n]]

    # --- helpers --------------------------------------------------------------

    def _overall_mastery(self) -> float:
        if not self.all_tasks:
            return 0.0
        return sum(self.tracker.task_mastery(t) for t in self.all_tasks) / len(self.all_tasks)

    def _seed_offset(self, task: str) -> int:
        """Deterministic 0-39 offset for a task (never hash()).
        Unknown task -> 0, keeping behaviour stable for adversarial-only sessions.
        """
        return _TASK_SEED_OFFSET.get(task, 0) % 40

    def _pick_easy(self) -> Tuple[str, Optional[int]]:
        candidates = [t for t in ("clean_deploy", "broken_pipeline") if t in self.all_tasks] or list(self.all_tasks)
        task = min(candidates, key=self.tracker.task_mastery)
        return (task, 1)

    def _pick_medium(self) -> Tuple[str, Optional[int]]:
        candidates = [
            t for t in ("broken_pipeline", "cascading_failure", "capacity_crisis")
            if t in self.all_tasks
        ] or list(self.all_tasks)
        task = min(candidates, key=self.tracker.task_mastery)
        return (task, 20 + self._seed_offset(task))

    def _pick_hard(self) -> Tuple[str, Optional[int]]:
        candidates = [
            t for t in ("judgment_call", "random_incident")
            if t in self.all_tasks
        ] or list(self.all_tasks)
        task = min(candidates, key=self.tracker.task_mastery)
        return (task, 60 + self._seed_offset(task))
