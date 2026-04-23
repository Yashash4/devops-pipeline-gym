# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DevOps Pipeline Environment Implementation."""

import os
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from devops_pipeline_gym.models import (
    ActionType,
    PipelineAction,
    PipelineObservation,
    Role,
    RoleHistoryEntry,
    ROLE_ACTIONS,
    ServiceHealth,
    ServiceStatus,
)
from server.adversarial_designer import AdversarialDesigner
from server.curriculum import CurriculumController
from server.handoff_metrics import HandoffTracker
from server.pipeline_engine import PipelineEngine
from server.rewards import (
    bound_step_reward,
    calculate_reward,
    handoff_quality_reward,
    role_alignment_reward,
    role_specialization_bonus,
)
from server.roles import RoleRouter
from server.scenarios import load_scenario

# Deterministic seeds per task
TASK_SEEDS = {
    "clean_deploy": 1001,
    "broken_pipeline": 2002,
    "judgment_call": 3003,
    "cascading_failure": 4004,
    "capacity_crisis": 5005,
    "random_incident": 6006,
}

TASK_MAX_STEPS = {
    "clean_deploy": 15,
    "broken_pipeline": 20,
    "judgment_call": 12,
    "cascading_failure": 15,
    "capacity_crisis": 15,
    "random_incident": 15,
}

# Map each task to its dominant failure type so curriculum can track mastery
# per-failure-type without having to introspect the scenario internals.
# Failure type names match CurriculumController._DEFAULT_FAILURE_TYPES.
_TASK_FAILURE_TYPE = {
    "clean_deploy": None,  # nothing broken → no failure-type signal
    "broken_pipeline": "config_error",
    "judgment_call": "degraded_performance",
    "cascading_failure": "degraded_performance",
    "capacity_crisis": "capacity_limit",
    "random_incident": None,  # varies per seed; curriculum treats as generic
}


# Goal suffixes that hint at investigation without giving away answers
_INVESTIGATION_HINTS = {
    "clean_deploy": " Use view_logs and view_config to inspect services before deploying.",
    "broken_pipeline": " Investigate service logs and configs to diagnose issues before acting.",
    "judgment_call": " Check service logs and configs to understand the incident before deciding.",
    "capacity_crisis": " Inspect database-primary logs and config to find the bottleneck.",
    "random_incident": " Investigate service logs and config to find the root cause.",
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
        self._last_action_key = None
        self._investigated_services = set()  # e.g. "logs:api-gateway", "config:cache-service"
        # Round 2 — role system + curriculum + adversarial designer + hand-off scoring.
        # All are lazy / graceful: AdversarialDesigner tolerates a missing
        # OLLAMA_API_KEY by returning None from generate(), callers fall back.
        self._role_router = RoleRouter()
        self._curriculum = CurriculumController()
        self._designer = AdversarialDesigner()
        self._handoff_tracker = HandoffTracker()
        self._current_role: Role = Role.SRE
        self._episode_roles: list = []               # one Role entry per executed step
        self._role_history: list = []                # list[RoleHistoryEntry] for observations
        self._previous_handoff: str | None = None    # notes from the last action w/ handoff
        self._current_task: str = "clean_deploy"     # last task selected (incl. "adversarial")
        self._coordination_bonus_accumulated: float = 0.0

    def reset(self, seed=None, episode_id=None, **kwargs) -> PipelineObservation:
        """Initialize a new episode.

        Task selection priority (Round 2 additions are additive):
          1. Explicit task in reset body (backward compat: kwargs['task'])
          2. DEVOPS_TASK env var (Round 1 integration tests use this)
          3. Curriculum pick (Round 2 autonomous mode)
        """
        explicit_task = kwargs.get("task") or os.environ.get("DEVOPS_TASK")
        chosen_seed = None
        if explicit_task:
            self._task_name = explicit_task
        else:
            # Round 2 — let the curriculum pick based on mastery/plateau signal.
            picked_task, seed_hint = self._curriculum.pick_task()
            if picked_task == "adversarial":
                # Curriculum signalled plateau. Try the designer; on None
                # (no API key / network / invalid JSON), fall back to a hard
                # random_incident seed. Phase 5.2 does NOT yet translate the
                # generated scenario into engine state — that wiring is deferred;
                # for now we preserve the plateau signal by jumping to a hard seed.
                weak_spots = self._curriculum.get_weak_failure_types()
                gen = self._designer.generate(weak_spots) if weak_spots else None
                if gen is not None:
                    # Stashed for future use (Phase 6+ can consume .to_scenario_spec()).
                    self._last_adversarial_scenario = gen
                self._task_name = "random_incident"
                chosen_seed = 85  # hard-tier seed (60 + random_incident offset 25)
            else:
                self._task_name = picked_task
                chosen_seed = seed_hint

        # Canonical seed — explicit / fallback / curriculum-supplied.
        if chosen_seed is None:
            chosen_seed = TASK_SEEDS.get(self._task_name, 9999)
        # random_incident honours DEVOPS_SEED for backward compat with Round 1 tests.
        if self._task_name == "random_incident":
            chosen_seed = int(os.environ.get("DEVOPS_SEED", str(chosen_seed)))

        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._episode_history = []
        self._viewed_actions = set()
        self._last_action_key = None
        self._investigated_services = set()
        # Round 2 per-episode reset. Curriculum + designer persist across
        # episodes (they learn / cache); router + handoff tracker are
        # per-episode state.
        self._role_router = RoleRouter()
        self._handoff_tracker = HandoffTracker()
        self._episode_roles = []
        self._role_history = []
        self._previous_handoff = None
        self._coordination_bonus_accumulated = 0.0
        self._current_role = Role.SRE  # default before next_role() below
        self._current_task = self._task_name

        if PipelineEnvironment._register_callback:
            PipelineEnvironment._register_callback(self)

        scenario = load_scenario(self._task_name, chosen_seed)
        self._engine = PipelineEngine(scenario, chosen_seed)
        self._max_steps = TASK_MAX_STEPS.get(self._task_name, 15)

        # First-pass observation to feed next_role(); then update and rebuild
        # so the returned obs carries the correct current_role.
        temp_obs = self._build_observation(
            last_action_result="Environment reset. Deployment pipeline initialized.",
            last_action_error=None,
            done=False,
            reward=0.0,
        )
        self._current_role = self._role_router.next_role(temp_obs)

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

        # Round 2 — role validation. Only enforced when the caller EXPLICITLY
        # set `role` on the action (Pydantic's model_fields_set tells us).
        # Round 1 clients that omit the field bypass this check — preserves
        # backward-compat without compromising Round 2 policy enforcement.
        role_was_explicit = "role" in action.model_fields_set
        role_error = None
        role_penalty = 0.0
        if role_was_explicit:
            if action.role != self._current_role:
                role_error = (
                    f"Role mismatch: action.role='{action.role.value}' but "
                    f"environment current_role='{self._current_role.value}'"
                )
                role_penalty = -0.15
            elif not self._role_router.validate_action(action.role, action.action_type):
                role_error = (
                    f"Action '{action.action_type.value}' is not permitted for "
                    f"role '{action.role.value}' — allowed: "
                    f"{[a.value for a in self._role_router.get_valid_actions(action.role)]}"
                )
                role_penalty = -0.10
        if role_error:
            self._episode_history.append({
                "step": self._state.step_count,
                "action": action.model_dump(),
                "reward": role_penalty,
                "error": role_error,
            })
            done = self._state.step_count >= self._max_steps
            return self._build_observation(
                last_action_result=None,
                last_action_error=role_error,
                done=done,
                reward=role_penalty,
            )

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

        # Round 1 base reward (bounded [-0.35, +0.30] by calculate_reward).
        round1_reward = calculate_reward(
            prev_state, current_state, action, self._viewed_actions,
            last_action_key=self._last_action_key, task_name=self._task_name,
        )
        self._last_action_key = f"{action.action_type.value}:{action.service_name or ''}"

        # Check episode termination
        done = self._check_done(action)

        # Include config_snapshot if viewing/editing config (needed before we
        # build any observation below).
        config_snapshot = None
        if action.action_type in (ActionType.VIEW_CONFIG, ActionType.EDIT_CONFIG):
            svc = self._engine.services.get(action.service_name)
            if svc:
                config_snapshot = svc.get_config_snapshot()

        # ─── Round 2 — role routing + hand-off scoring + reward additions ──
        # Score hand-off FIRST (before current_role is advanced) so the
        # transition pair lines up with tracker.handoffs[-1].
        interim_obs = self._build_observation(
            last_action_result=result_text,
            last_action_error=None,
            done=done,
            reward=round1_reward,       # placeholder; final obs below uses combined
            config_snapshot=config_snapshot,
        )
        next_role = self._role_router.next_role(interim_obs)

        if action.handoff_notes and next_role != action.role:
            self._handoff_tracker.score_handoff(
                from_role=action.role,
                to_role=next_role,
                notes=action.handoff_notes,
                last_action=action,
                current_services=self._engine.get_service_statuses(),
            )

        # Round 2 reward deltas — additive on top of Round 1 base.
        round2_delta = 0.0
        if role_was_explicit:
            round2_delta += role_alignment_reward(action, self._role_router)
        handoff_delta, self._coordination_bonus_accumulated = handoff_quality_reward(
            self._handoff_tracker,
            action.role,
            next_role,
            self._coordination_bonus_accumulated,
        )
        round2_delta += handoff_delta
        if done:
            # Include THIS action's role when computing specialisation since
            # self._episode_roles is appended below (after reward combination).
            round2_delta += role_specialization_bonus(
                self._episode_roles + [action.role]
            )

        # Final bounded reward for this step.
        reward = bound_step_reward(round1_reward + round2_delta)

        # ── Round 1 history entry (uses final combined reward) ────────────
        broke_healthy = False
        for name, curr_svc in current_state["services"].items():
            prev_svc = prev_state["services"].get(name, {})
            if prev_svc.get("health") == "healthy" and curr_svc["health"] in ("degraded", "down"):
                broke_healthy = True
        history_entry = {
            "step": self._state.step_count,
            "action": action.model_dump(),
            "reward": reward,
            "error": None,
            "broke_healthy": broke_healthy,
            "system_health": self._engine.get_system_health(),
        }
        if action.action_type == ActionType.DEPLOY and action.service_name == "api-gateway":
            cache_svc = self._engine.services.get("cache-service")
            if cache_svc:
                history_entry["cache_health_at_deploy"] = cache_svc.health.value
        self._episode_history.append(history_entry)

        # ── Advance Round 2 state for the NEXT step ────────────────────────
        self._role_router.record_role(action.role)
        self._episode_roles.append(action.role)
        self._role_history.append(RoleHistoryEntry(
            step=self._state.step_count,
            role=action.role,
            action_type=action.action_type,
        ))
        self._previous_handoff = action.handoff_notes
        self._current_role = next_role

        # ── Curriculum tracking — record the episode on the last step. ────
        if done:
            final_health = self._engine.get_system_health()
            any_broke = any(
                h.get("broke_healthy", False) for h in self._episode_history
            )
            success = final_health >= 50.0 and not any_broke
            final_reward_sum = sum(
                h.get("reward", 0.0) for h in self._episode_history
            )
            self._curriculum.tracker.record_episode(
                task=self._task_name,
                failure_type=_TASK_FAILURE_TYPE.get(self._task_name),
                success=success,
                final_reward=final_reward_sum,
            )

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
            investigated = (
                f"logs:{svc.name}" in self._investigated_services
                or f"config:{svc.name}" in self._investigated_services
            )
            # Get recovery status from engine state
            svc_state = self._engine.services.get(svc.name)
            recovery = "stable"
            if svc_state and hasattr(svc_state, '_recovery_steps_remaining') and svc_state._recovery_steps_remaining > 0:
                recovery = f"stabilizing ({svc_state._recovery_steps_remaining} steps remaining)"
            # Mask health for uninvestigated unhealthy services
            shown_health = svc.health
            if not investigated and svc.health.value != "healthy":
                shown_health = ServiceHealth.UNKNOWN
            filtered_statuses.append(ServiceStatus(
                name=svc.name,
                health=shown_health,
                current_version=svc.current_version,
                # Metrics visible only after investigation
                error_rate=svc.error_rate if investigated else 0.0,
                request_latency_ms=svc.request_latency_ms if investigated else 0.0,
                active_connections=svc.active_connections,
                last_deploy_timestamp=svc.last_deploy_timestamp,
                # Hidden until view_logs: detailed resource usage
                cpu_percent=svc.cpu_percent if investigated else 0.0,
                memory_percent=svc.memory_percent if investigated else 0.0,
                recovery_status=recovery,
            ))

        # Append investigation hint to goal
        goal = scenario.goal
        hint = _INVESTIGATION_HINTS.get(self._task_name, "")
        if hint and not self._investigated_services:
            goal = goal + hint

        # Build summary — only reveal details for investigated services
        alerts = []
        uninvestigated_alerts = 0
        for name, svc_state in self._engine.services.items():
            investigated = (
                f"logs:{name}" in self._investigated_services
                or f"config:{name}" in self._investigated_services
            )
            if svc_state.health == ServiceHealth.DOWN:
                if investigated:
                    alerts.append(f"CRITICAL: {name} is DOWN")
                else:
                    uninvestigated_alerts += 1
            elif svc_state.health == ServiceHealth.DEGRADED:
                if investigated:
                    alerts.append(
                        f"WARNING: {name} degraded "
                        f"(lat={svc_state.latency_ms:.0f}ms, err={svc_state.error_rate:.1f}/s)"
                    )
                else:
                    uninvestigated_alerts += 1
            elif investigated and svc_state.cpu_percent > 80:
                alerts.append(f"CAUTION: {name} CPU high ({svc_state.cpu_percent:.0f}%)")
            # Recovery status alert — inside the loop, for THIS service
            if hasattr(svc_state, '_recovery_steps_remaining') and svc_state._recovery_steps_remaining > 0:
                alerts.append(f"INFO: {name} recovering — stabilizing ({svc_state._recovery_steps_remaining} steps remaining)")
        if uninvestigated_alerts > 0:
            alerts.append(f"ALERT: {uninvestigated_alerts} service(s) may have issues — use view_logs to investigate")
        # Add dependency chain hints for investigated degraded services only
        for name, svc_state in self._engine.services.items():
            investigated = (
                f"logs:{name}" in self._investigated_services
                or f"config:{name}" in self._investigated_services
            )
            if investigated and svc_state.health in (ServiceHealth.DEGRADED, ServiceHealth.DOWN):
                upstream_issues = [
                    d for d in svc_state.dependencies
                    if d in self._engine.services
                    and self._engine.services[d].health in (ServiceHealth.DEGRADED, ServiceHealth.DOWN)
                ]
                if upstream_issues:
                    alerts.append(
                        f"HINT: {name} depends on {', '.join(upstream_issues)} "
                        f"(also unhealthy — root cause likely upstream)"
                    )
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
            # Round 2 fields — all backed by self state maintained in step()/reset().
            current_role=self._current_role,
            role_history=list(self._role_history),
            previous_handoff=self._previous_handoff,
        )

    def _get_available_actions(self):
        """Context-sensitive: only show valid actions, filtered to current role.

        Round 2: the returned list is the intersection of (a) actions
        valid in the current engine state and (b) actions permitted for
        self._current_role. Round 1 clients that ignore this field keep
        working; Round 2 clients use it to avoid submitting mismatched
        roles.
        """
        actions = ["view_pipeline", "view_logs", "approve", "abort"]
        if self._engine.has_services():
            actions.extend(["view_config", "edit_config", "deploy", "rollback"])
        if self._engine.has_pending_migrations():
            actions.append("run_migration")
        # Filter by current role's permitted action set.
        role_allowed = {a.value for a in ROLE_ACTIONS.get(self._current_role, [])}
        if role_allowed:
            actions = [a for a in actions if a in role_allowed]
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
