# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DevOps Pipeline Environment Implementation."""

import os
import random
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from devops_pipeline_env.models import (
    ActionType,
    PipelineAction,
    PipelineObservation,
    ServiceHealth,
    ServiceStatus,
)
from server.graders import grade_task
from server.pipeline_engine import PipelineEngine
from server.rewards import calculate_reward
from server.scenarios import load_scenario

# Deterministic seeds per task
TASK_SEEDS = {
    "clean_deploy": 1001,
    "broken_pipeline": 2002,
    "judgment_call": 3003,
    "cascading_failure": 4004,
    "capacity_crisis": 5005,
}

TASK_MAX_STEPS = {
    "clean_deploy": 15,
    "broken_pipeline": 20,
    "judgment_call": 12,
    "cascading_failure": 15,
    "capacity_crisis": 15,
}

# Goal suffixes that hint at investigation without giving away answers
_INVESTIGATION_HINTS = {
    "clean_deploy": " Use view_logs and view_config to inspect services before deploying.",
    "broken_pipeline": " Investigate service logs and configs to diagnose issues before acting.",
    "judgment_call": " Check service logs and configs to understand the incident before deciding.",
    "capacity_crisis": " Inspect database-primary logs and config to find the bottleneck.",
}


class PipelineEnvironment(Environment):
    """CI/CD Pipeline environment — manages microservice deployments."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = False
    _register_callback = None  # Set by app.py to register active env for /grader

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._engine = None
        self._task_name = "clean_deploy"
        self._max_steps = 15
        self._episode_history = []
        self._viewed_actions = set()
        self._investigated_services = set()  # e.g. "logs:api-gateway", "config:cache-service"

    def reset(self) -> PipelineObservation:
        """Initialize a new episode. Task selected via DEVOPS_TASK env var."""
        self._task_name = os.environ.get("DEVOPS_TASK", "clean_deploy")
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._episode_history = []
        self._viewed_actions = set()
        self._investigated_services = set()
        if PipelineEnvironment._register_callback:
            PipelineEnvironment._register_callback(self)

        seed = TASK_SEEDS.get(self._task_name, 9999)
        scenario = load_scenario(self._task_name, seed)
        self._engine = PipelineEngine(scenario, seed)
        self._max_steps = TASK_MAX_STEPS.get(self._task_name, 15)

        return self._build_observation(
            last_action_result="Environment reset. Deployment pipeline initialized.",
            last_action_error=None,
            done=False,
            reward=0.0,
        )

    def step(self, action: PipelineAction) -> PipelineObservation:
        """Execute action, return observation with reward and done."""
        self._state.step_count += 1

        prev_state = self._engine.snapshot()

        # Validate action
        error = self._validate_action(action)
        if error:
            self._episode_history.append({
                "step": self._state.step_count,
                "action": action.model_dump(),
                "reward": -0.05,
                "error": error,
            })
            done = self._state.step_count >= self._max_steps
            return self._build_observation(
                last_action_result=None,
                last_action_error=error,
                done=done,
                reward=-0.05,
            )

        # Track investigation actions BEFORE executing (so the observation
        # returned from this step already includes the revealed info)
        if action.action_type == ActionType.VIEW_LOGS and action.service_name:
            self._investigated_services.add(f"logs:{action.service_name}")
        elif action.action_type == ActionType.VIEW_CONFIG and action.service_name:
            self._investigated_services.add(f"config:{action.service_name}")

        # Execute action
        result_text = self._engine.execute(action)

        current_state = self._engine.snapshot()

        # Calculate outcome-based reward
        reward = calculate_reward(prev_state, current_state, action, self._viewed_actions)

        # Check episode termination
        done = self._check_done(action)

        # Track if we broke a healthy service (for grader)
        broke_healthy = False
        for name, curr_svc in current_state["services"].items():
            prev_svc = prev_state["services"].get(name, {})
            if prev_svc.get("health") == "healthy" and curr_svc["health"] in ("degraded", "down"):
                broke_healthy = True

        self._episode_history.append({
            "step": self._state.step_count,
            "action": action.model_dump(),
            "reward": reward,
            "error": None,
            "broke_healthy": broke_healthy,
        })

        # Include config_snapshot if viewing/editing config
        config_snapshot = None
        if action.action_type in (ActionType.VIEW_CONFIG, ActionType.EDIT_CONFIG):
            svc = self._engine.services.get(action.service_name)
            if svc:
                config_snapshot = svc.get_config_snapshot()

        return self._build_observation(
            last_action_result=result_text,
            last_action_error=None,
            done=done,
            reward=reward,
            config_snapshot=config_snapshot,
        )

    @property
    def state(self) -> State:
        return self._state

    def get_episode_history(self):
        return self._episode_history

    def get_engine(self):
        return self._engine

    def get_task_name(self):
        return self._task_name

    def _build_observation(self, last_action_result, last_action_error,
                           done, reward, config_snapshot=None):
        """Build observation from current engine state.

        Partial observability: services show only high-level metrics by default.
        CPU, memory are hidden until the agent runs view_logs for that service.
        Config is hidden until the agent runs view_config for that service.
        """
        scenario = self._engine.scenario

        # Build service statuses with partial observability
        raw_statuses = self._engine.get_service_statuses()
        filtered_statuses = []
        for svc in raw_statuses:
            has_logs = f"logs:{svc.name}" in self._investigated_services
            filtered_statuses.append(ServiceStatus(
                name=svc.name,
                health=svc.health,
                current_version=svc.current_version,
                # Always visible: high-level numbers so agent sees something is wrong
                error_rate=svc.error_rate,
                request_latency_ms=svc.request_latency_ms,
                active_connections=svc.active_connections,
                last_deploy_timestamp=svc.last_deploy_timestamp,
                # Hidden until view_logs: detailed resource usage
                cpu_percent=svc.cpu_percent if has_logs else 0.0,
                memory_percent=svc.memory_percent if has_logs else 0.0,
            ))

        # Append investigation hint to goal
        goal = scenario.goal
        hint = _INVESTIGATION_HINTS.get(self._task_name, "")
        if hint and not self._investigated_services:
            goal = goal + hint

        # Build summary from raw engine state (not filtered)
        alerts = []
        for name, svc_state in self._engine.services.items():
            if svc_state.health == ServiceHealth.DOWN:
                alerts.append(f"CRITICAL: {name} is DOWN")
            elif svc_state.health == ServiceHealth.DEGRADED:
                alerts.append(
                    f"WARNING: {name} degraded "
                    f"(lat={svc_state.latency_ms:.0f}ms, err={svc_state.error_rate:.1f}/s)"
                )
            elif svc_state.cpu_percent > 80:
                alerts.append(f"CAUTION: {name} CPU high ({svc_state.cpu_percent:.0f}%)")
        summary = "; ".join(alerts) if alerts else "All services nominal."

        return PipelineObservation(
            task_description=scenario.task_description,
            goal=goal,
            step_number=self._state.step_count,
            max_steps=self._max_steps,
            services=filtered_statuses,
            pipeline=self._engine.get_pipeline_status(),
            migrations=self._engine.get_migration_status(),
            active_alerts=self._engine.get_alerts(),
            available_actions=self._get_available_actions(),
            last_action_result=last_action_result,
            last_action_error=last_action_error,
            config_snapshot=config_snapshot,
            done=done,
            reward=reward,
            summary=summary,
        )

    def _get_available_actions(self):
        """Context-sensitive: only show valid actions."""
        actions = ["view_pipeline", "view_logs", "approve", "abort"]
        if self._engine.has_services():
            actions.extend(["view_config", "edit_config", "deploy", "rollback"])
        if self._engine.has_pending_migrations():
            actions.append("run_migration")
        return actions

    def _validate_action(self, action):
        """Return error string if action is invalid, None if valid."""
        if action.action_type in (
            ActionType.VIEW_LOGS, ActionType.VIEW_CONFIG,
            ActionType.EDIT_CONFIG, ActionType.DEPLOY,
            ActionType.ROLLBACK,
        ):
            if not action.service_name:
                return f"action_type '{action.action_type.value}' requires service_name"
            if action.service_name not in self._engine.get_service_names():
                return (
                    f"Unknown service '{action.service_name}'. "
                    f"Available: {self._engine.get_service_names()}"
                )
        if action.action_type == ActionType.DEPLOY and not action.target_version:
            return "deploy requires target_version"
        if action.action_type == ActionType.EDIT_CONFIG and not action.config_edits:
            return "edit_config requires config_edits"
        if action.action_type == ActionType.RUN_MIGRATION and not action.migration_name:
            return "run_migration requires migration_name"
        return None

    def _check_done(self, action):
        """Episode ends on approve, abort, max steps, or catastrophic failure."""
        if action.action_type == ActionType.APPROVE:
            return True
        if action.action_type == ActionType.ABORT:
            return True
        if self._state.step_count >= self._max_steps:
            return True
        if self._engine.get_system_health() < 20.0:
            return True
        return False
