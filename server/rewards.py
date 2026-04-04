# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Outcome-based reward calculator for the DevOps Pipeline Environment."""

from devops_pipeline_env.models import ActionType


def calculate_reward(prev_snapshot, current_snapshot, action, viewed_actions):
    """
    Outcome-based reward. No procedure bonuses.
    viewed_actions: a set owned by the environment instance, tracking first-time views.
    Returns a float (can be negative).
    """
    reward = 0.0

    # 1. System health delta (primary signal)
    health_delta = current_snapshot["system_health"] - prev_snapshot["system_health"]
    reward += health_delta * 0.005  # +0.005 per 1% health improvement

    # 2. Deployment progress
    for name, curr_svc in current_snapshot["services"].items():
        prev_svc = prev_snapshot["services"].get(name, {})
        if curr_svc["prod_deployed"] and not prev_svc.get("prod_deployed"):
            reward += 0.15  # Service successfully reached production
        if curr_svc["staging_verified"] and not prev_svc.get("staging_verified"):
            reward += 0.05  # Service verified in staging

    # 3. Broke something that was working
    for name, curr_svc in current_snapshot["services"].items():
        prev_svc = prev_snapshot["services"].get(name, {})
        if prev_svc.get("health") == "healthy" and curr_svc["health"] in ("degraded", "down"):
            reward -= 0.30  # Catastrophic penalty

    # 4. Investigation bonus (+0.02 for first-time view_* actions)
    if action.action_type in (ActionType.VIEW_PIPELINE, ActionType.VIEW_LOGS, ActionType.VIEW_CONFIG):
        action_key = f"{action.action_type.value}:{action.service_name or 'global'}"
        if action_key not in viewed_actions:
            viewed_actions.add(action_key)
            reward += 0.02
            return reward  # view actions don't trigger no-op penalty

    # 5. True no-op penalty (no state change and not an investigation action)
    if prev_snapshot == current_snapshot:
        if action.action_type not in (ActionType.VIEW_PIPELINE, ActionType.VIEW_LOGS, ActionType.VIEW_CONFIG):
            reward -= 0.01

    return reward
