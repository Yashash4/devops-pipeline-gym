# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Round 2 — hand-off quality metrics between role transitions.

When the active role changes mid-episode (SRE -> DEV -> OPS, etc.), the
outgoing agent writes `handoff_notes` on its action. This module scores
those notes for whether they actually carry useful context to the next
role.

Rubric (each contribution is independent, max total = 1.0):
  context          0.30  — notes mention at least one current service by name
  diagnosis        0.40  — SRE hand-offs need diagnosis keywords;
                          non-SRE hand-offs auto-grant 0.4 (their job isn't
                          to diagnose)
  target_action    0.30  — notes suggest what the next role should do

Reward pipeline (Phase 5 integration) multiplies `quality_score` by 0.02
per transition and caps cumulative bonus at +0.08 per episode, so even a
perfect-score spam strategy contributes at most ~6% of an episode's
total reward.

This module is STANDALONE for Phase 4 — no imports from pipeline_environment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from devops_pipeline_gym.models import Role, ServiceStatus

# Keywords. Kept as module-level constants so tests can reason about them.
_DIAGNOSIS_KEYWORDS = (
    "root cause",
    "config",
    "error",
    "logs show",
    "symptom",
    "cpu",
    "memory",
    "latency",
    "cascading",
    "investigating",
)

_ACTION_KEYWORDS = (
    "deploy",
    "rollback",
    "edit",
    "fix",
    "migrate",
    "approve",
    "abort",
    "update",
    "restart",
)

# Scoring weights (must sum to 1.0).
_W_CONTEXT = 0.3
_W_DIAGNOSIS = 0.4
_W_ACTION = 0.3


@dataclass
class HandoffQuality:
    from_role: Role
    to_role: Role
    notes: str
    has_context: bool
    has_diagnosis: bool
    has_target_action: bool
    quality_score: float  # in [0.0, 1.0]


class HandoffTracker:
    """Scores and logs hand-off quality between role transitions."""

    def __init__(self) -> None:
        self.handoffs: List[HandoffQuality] = []

    def score_handoff(
        self,
        from_role: Role,
        to_role: Role,
        notes: Optional[str],
        last_action: Any = None,  # accepted for parity with BATTLEPLAN signature; unused here
        current_services: Optional[List[ServiceStatus]] = None,
    ) -> HandoffQuality:
        """Score one hand-off. Empty/None notes score 0.0. Appends to `handoffs`."""
        notes_str = notes or ""
        notes_lower = notes_str.lower()

        # 1. Context — does the note mention a real service by name?
        service_names: List[str] = [s.name for s in (current_services or [])]
        has_context = any(sn and sn.lower() in notes_lower for sn in service_names)

        # 2. Diagnosis — for SRE hand-offs, must contain a diagnosis keyword.
        #    For non-SRE hand-offs, diagnosis credit is auto-granted (their job
        #    is doing, not diagnosing).
        has_diagnosis_keyword = any(kw in notes_lower for kw in _DIAGNOSIS_KEYWORDS)
        has_diagnosis = (from_role == Role.SRE) and has_diagnosis_keyword

        # 3. Target action — does the note suggest what the next role should do?
        has_target_action = any(kw in notes_lower for kw in _ACTION_KEYWORDS)

        score = 0.0
        if has_context:
            score += _W_CONTEXT
        if from_role != Role.SRE or has_diagnosis:
            score += _W_DIAGNOSIS
        if has_target_action:
            score += _W_ACTION
        # Clamp for safety (float round-off).
        if score > 1.0:
            score = 1.0
        elif score < 0.0:
            score = 0.0

        handoff = HandoffQuality(
            from_role=from_role,
            to_role=to_role,
            notes=notes_str,
            has_context=has_context,
            has_diagnosis=has_diagnosis,
            has_target_action=has_target_action,
            quality_score=score,
        )
        self.handoffs.append(handoff)
        return handoff

    def average_quality(self) -> float:
        if not self.handoffs:
            return 0.0
        return sum(h.quality_score for h in self.handoffs) / len(self.handoffs)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_handoffs": len(self.handoffs),
            "avg_quality": round(self.average_quality(), 3),
            "handoffs": [
                {
                    "from": h.from_role.value,
                    "to": h.to_role.value,
                    "notes": h.notes,
                    "has_context": h.has_context,
                    "has_diagnosis": h.has_diagnosis,
                    "has_target_action": h.has_target_action,
                    "score": round(h.quality_score, 3),
                }
                for h in self.handoffs
            ],
        }
