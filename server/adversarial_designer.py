# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Round 2 — Adversarial scenario designer.

When CurriculumController.pick_task() returns ("adversarial", None), the
environment asks this designer for a novel incident that targets the
agent's weak spots. The generated scenario is a structured spec — NOT a
full engine state; Phase 5 translates it into PipelineEngine state.

Hard rules (per CLAUDE.md):
  - No LLM calls inside pipeline_engine.py / scenarios.py / graders.py.
    This file is only called from pipeline_environment.reset() (pre-episode),
    never from step().
  - Graceful degradation: returns None on any failure. Caller falls back
    to random_incident.

Model is configurable via DESIGNER_MODEL env var (default gpt-oss:120b).
A small in-memory cache avoids repeat LLM calls for the same weak_spots.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from server.ollama_client import OllamaClient

logger = logging.getLogger(__name__)


DESIGNER_PROMPT = """You are designing a realistic DevOps incident scenario for training an AI SRE agent.

The agent has shown weakness in these failure types: {weak_spots}

Available services: database-primary, auth-service, api-gateway, cache-service, web-frontend
Dependencies: database-primary -> {{auth, cache}} -> api-gateway -> web-frontend

Generate ONE novel scenario that:
1. Targets the listed weak spot(s)
2. Has a clear root cause + symptoms
3. Requires multi-step diagnosis (3-8 actions to resolve)
4. Is realistic (something that happens in production)
5. Cannot be solved by just deploying or rolling back

Return JSON only (no markdown, no prose):
{{
  "scenario_id": "adv_<short_name>",
  "description": "1-sentence task description",
  "goal": "What success looks like",
  "root_cause": "What actually broke",
  "initial_failures": [
    {{"service": "<name>", "failure_type": "<type>", "severity": "moderate|severe"}}
  ],
  "misleading_signals": ["logs or alerts that could distract agent"],
  "expected_diagnosis_steps": ["list of view_* actions agent should take"],
  "expected_fix_actions": ["list of edit_config/deploy/rollback to resolve"],
  "max_steps": 12,
  "difficulty": "hard"
}}
"""

# Fields that MUST be present in the LLM's JSON output for the scenario
# to be usable. Anything else is best-effort.
_REQUIRED_FIELDS = ("description", "goal", "root_cause", "initial_failures")


@dataclass
class GeneratedScenario:
    scenario_id: str
    description: str
    goal: str
    root_cause: str
    initial_failures: List[Dict[str, Any]]
    misleading_signals: List[str] = field(default_factory=list)
    expected_diagnosis_steps: List[str] = field(default_factory=list)
    expected_fix_actions: List[str] = field(default_factory=list)
    max_steps: int = 12
    difficulty: str = "hard"
    raw_json: Dict[str, Any] = field(default_factory=dict)

    def to_scenario_spec(self) -> Dict[str, Any]:
        """Shape expected by Phase 5 PipelineEngine setup."""
        return {
            "task_id": self.scenario_id,
            "description": self.description,
            "goal": self.goal,
            "max_steps": self.max_steps,
            "initial_failures": list(self.initial_failures),
        }


class AdversarialDesigner:
    """Generates novel incident scenarios via Ollama Cloud."""

    def __init__(self, model: Optional[str] = None) -> None:
        self.model = model or os.environ.get("DESIGNER_MODEL") or "gpt-oss:120b"
        self.client = OllamaClient(model=self.model)
        self.cache: Dict[str, GeneratedScenario] = {}

    # --- public API -----------------------------------------------------------

    def generate(
        self,
        weak_spots: List[str],
        use_cache: bool = True,
    ) -> Optional[GeneratedScenario]:
        """Ask the LLM for a scenario targeting these weak_spots.

        Returns a `GeneratedScenario` on success, or None on:
          - missing OLLAMA_API_KEY (client degrades to None)
          - network / HTTP error
          - LLM returns non-JSON or JSON missing required fields
        """
        if not weak_spots:
            logger.warning("AdversarialDesigner.generate: empty weak_spots, returning None")
            return None

        cache_key = self._cache_key(weak_spots)
        if use_cache and cache_key in self.cache:
            logger.info("AdversarialDesigner: cache hit for weak_spots=%s", weak_spots)
            return self.cache[cache_key]

        prompt = DESIGNER_PROMPT.format(weak_spots=", ".join(weak_spots))
        response = self.client.generate_json(
            system="You are a DevOps incident scenario designer. Return valid JSON only.",
            user=prompt,
            temperature=0.7,
        )
        if response is None:
            logger.warning("AdversarialDesigner.generate: client returned None (no key or failed)")
            return None

        scenario = self._parse(response, weak_spots)
        if scenario is not None:
            self.cache[cache_key] = scenario
        return scenario

    def clear_cache(self) -> None:
        self.cache.clear()

    # --- private --------------------------------------------------------------

    def _parse(
        self,
        response: Dict[str, Any],
        weak_spots: List[str],
    ) -> Optional[GeneratedScenario]:
        missing = [f for f in _REQUIRED_FIELDS if f not in response]
        if missing:
            logger.error(
                "AdversarialDesigner._parse: missing required fields %s — got keys=%s",
                missing, sorted(response.keys()),
            )
            return None

        initial = response.get("initial_failures") or []
        if not isinstance(initial, list) or not initial:
            logger.error("AdversarialDesigner._parse: initial_failures must be a non-empty list")
            return None

        scenario_id = response.get("scenario_id") or ("adv_" + "_".join(sorted(weak_spots))[:40])

        try:
            return GeneratedScenario(
                scenario_id=str(scenario_id),
                description=str(response["description"]),
                goal=str(response["goal"]),
                root_cause=str(response["root_cause"]),
                initial_failures=list(initial),
                misleading_signals=list(response.get("misleading_signals") or []),
                expected_diagnosis_steps=list(response.get("expected_diagnosis_steps") or []),
                expected_fix_actions=list(response.get("expected_fix_actions") or []),
                max_steps=int(response.get("max_steps") or 12),
                difficulty=str(response.get("difficulty") or "hard"),
                raw_json=response,
            )
        except (TypeError, ValueError) as e:
            logger.error("AdversarialDesigner._parse: coercion error: %s", e)
            return None

    @staticmethod
    def _cache_key(weak_spots: List[str]) -> str:
        return "|".join(sorted(weak_spots))
