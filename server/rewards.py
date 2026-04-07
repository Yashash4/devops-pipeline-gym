# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Outcome-based reward calculator for the DevOps Pipeline Environment."""

from devops_pipeline_env.models import ActionType


# Task urgency multipliers — harder tasks get steeper reward gradients
TASK_URGENCY = {
    "clean_deploy": 1.0,
    "broken_pipeline": 1.2,
    "judgment_call": 1.5,
    "cascading_failure": 1.3,
    "capacity_crisis": 1.4,
    "random_incident": 1.3,
}


def calculate_reward(prev_snapshot, current_snapshot, action, viewed_actions,
                     last_action_key=None, task_name="clean_deploy"):
    """
    Outcome-based reward. No procedure bonuses.
    viewed_actions: a set tracking first-time investigation keys (e.g. "view_logs:api-gateway")
    last_action_key: string key of previous action for repeat detection
    task_name: current task for urgency scaling
    Returns a float bounded to [-0.35, +0.20].
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

    # 4. Investigation bonus (info-gain aware)
    if action.action_type in (ActionType.VIEW_PIPELINE, ActionType.VIEW_LOGS, ActionType.VIEW_CONFIG):
        action_key = f"{action.action_type.value}:{action.service_name or 'global'}"
        if action_key not in viewed_actions:
            viewed_actions.add(action_key)
            # Higher reward for investigating unhealthy services (more info gain)
            if action.service_name:
                svc_data = current_snapshot["services"].get(action.service_name, {})
                if svc_data.get("health") in ("degraded", "down"):
                    reward += 0.04  # High info-gain
                else:
                    reward += 0.01  # Low info-gain (healthy service)
            else:
                reward += 0.02  # view_pipeline (global)
        else:
            reward -= 0.01  # Repeated investigation of same target

        # Apply urgency scaling and bound
        reward *= TASK_URGENCY.get(task_name, 1.0)
        return max(min(reward, 0.20), -0.35)

    # 5. Repeated exact action penalty
    current_action_key = f"{action.action_type.value}:{action.service_name or ''}"
    if last_action_key and current_action_key == last_action_key:
        reward -= 0.02  # Exact repeat penalty

    # 6. Apply task urgency scaling and bound
    reward *= TASK_URGENCY.get(task_name, 1.0)
    return max(min(reward, 0.20), -0.35)
