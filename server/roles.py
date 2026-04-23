# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Round 2 — state-driven role router for the DevOps Pipeline Environment.

Decides which role (DEV / SRE / OPS) should act next based on observed state.
Does NOT execute actions and does NOT mutate the engine — it only reads an
observation and returns a role. Integration into pipeline_environment.py
happens in Phase 5; this module stands alone for Phase 1.

Role responsibilities:
    DEV — writes configs and proposes fixes (VIEW_CONFIG, EDIT_CONFIG, RUN_MIGRATION)
    SRE — investigates and diagnoses                 (VIEW_LOGS, VIEW_PIPELINE)
    OPS — controls production                        (DEPLOY, ROLLBACK, APPROVE, ABORT)
"""

from typing import List

from devops_pipeline_gym.models import (
    ROLE_ACTIONS,
    ActionType,
    PipelineObservation,
    PipelineStage,
    Role,
    ServiceHealth,
)


class RoleRouter:
    """Decides which role acts next based on environment state.

    Priority order (first match wins):
      1. Unhealthy services AND no recent SRE investigation    -> SRE
      2. Config issue AND SRE investigated AND Dev not acted   -> DEV
      3. Pipeline in STAGING / DEPLOYING / APPROVAL            -> OPS
      4. All healthy AND past step 3                           -> OPS
      5. Default                                               -> SRE
    """

    # Window sizes used by the priority logic. Kept explicit so tests can
    # reason about them.
    SRE_LOOKBACK = 3
    DEV_LOOKBACK = 2
    HEALTHY_SETTLE_STEP = 3

    def __init__(self) -> None:
        self.history: List[Role] = []

    # --- primary API ---------------------------------------------------------

    def next_role(self, obs: PipelineObservation) -> Role:
        """Return the role that should act next given this observation."""
        has_unhealthy = any(
            s.health != ServiceHealth.HEALTHY for s in obs.services
        )
        investigation_done = self._has_recent_role(Role.SRE, self.SRE_LOOKBACK)
        config_fix_done = self._has_recent_role(Role.DEV, self.DEV_LOOKBACK)

        # 1. New incident, no investigation yet
        if has_unhealthy and not investigation_done:
            return Role.SRE

        # 2. Investigation done, config issue flagged, Dev hasn't fixed it
        if (
            self._config_issue_present(obs)
            and investigation_done
            and not config_fix_done
        ):
            return Role.DEV

        # 3. Pipeline is in a stage that requires Ops intervention
        if obs.pipeline and obs.pipeline.stage in (
            PipelineStage.STAGING,
            PipelineStage.DEPLOYING,
            PipelineStage.APPROVAL,
        ):
            return Role.OPS

        # 4. System stable past the initial settling window
        if not has_unhealthy and obs.step_number > self.HEALTHY_SETTLE_STEP:
            return Role.OPS

        # 5. Fallback: monitor via SRE
        return Role.SRE

    def record_role(self, role: Role) -> None:
        """Append a role transition to history. Call after each executed step."""
        self.history.append(role)

    def get_valid_actions(self, role: Role) -> List[ActionType]:
        """Return the ordered list of ActionTypes permitted for this role."""
        return list(ROLE_ACTIONS[role])

    def validate_action(self, role: Role, action_type: ActionType) -> bool:
        """True iff action_type is permitted for role per ROLE_ACTIONS."""
        return action_type in ROLE_ACTIONS[role]

    # --- helpers -------------------------------------------------------------

    def _has_recent_role(self, role: Role, lookback: int) -> bool:
        if not self.history or lookback <= 0:
            return False
        return role in self.history[-lookback:]

    def _config_issue_present(self, obs: PipelineObservation) -> bool:
        """True if the last error or any active alert hints at config trouble."""
        if obs.last_action_error:
            if "config" in obs.last_action_error.lower():
                return True
        for alert in obs.active_alerts:
            msg = alert.message.lower()
            if "config" in msg or "invalid" in msg or "misconfigured" in msg:
                return True
        return False
