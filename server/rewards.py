# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Outcome-based reward calculator for the DevOps Pipeline Environment."""

from devops_pipeline_gym.models import ActionType


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
    Outcome-based reward. No procedure bonuses. No early returns.
    ALL actions go through the full reward pipeline.
    Returns a float bounded to [-0.35, +0.30].
    """
    reward = 0.0

    # 1. System health delta (ALL actions)
    health_delta = current_snapshot["system_health"] - prev_snapshot["system_health"]
    reward += health_delta * 0.005

    # 2. Deployment progress (ALL actions — recovery ticks can change state)
    for name, curr_svc in current_snapshot["services"].items():
        prev_svc = prev_snapshot["services"].get(name, {})
        if curr_svc["prod_deployed"] and not prev_svc.get("prod_deployed"):
            reward += 0.15
        if curr_svc["staging_verified"] and not prev_svc.get("staging_verified"):
            reward += 0.05

    # 3. Broke healthy service penalty (ALL actions)
    for name, curr_svc in current_snapshot["services"].items():
        prev_svc = prev_snapshot["services"].get(name, {})
        if prev_svc.get("health") == "healthy" and curr_svc["health"] in ("degraded", "down"):
            reward -= 0.30

    # 4. Sub-goal completion rewards (intermediate milestones for RL)
    # Config change detected — check if it fixed an error
    for name, curr_svc in current_snapshot["services"].items():
        prev_svc = prev_snapshot["services"].get(name, {})
        prev_config = prev_svc.get("config", {})
        curr_config = curr_svc.get("config", {})
        if prev_config and curr_config and prev_config != curr_config:
            # Config changed — reward if health improved on this service
            if prev_svc.get("health") in ("degraded", "down") and curr_svc["health"] == "healthy":
                reward += 0.08

    # Migration completed
    prev_pending = len(prev_snapshot.get("migrations_pending", []))
    curr_pending = len(current_snapshot.get("migrations_pending", []))
    if curr_pending < prev_pending:
        reward += 0.06

    # Alert resolved
    prev_alerts = len(prev_snapshot.get("alerts", []))
    curr_alerts = len(current_snapshot.get("alerts", []))
    if curr_alerts < prev_alerts:
        reward += 0.03

    # 5. Investigation bonus with diminishing returns (view_* actions only)
    if action.action_type in (ActionType.VIEW_PIPELINE, ActionType.VIEW_LOGS, ActionType.VIEW_CONFIG):
        action_key = f"{action.action_type.value}:{action.service_name or 'global'}"
        if action_key not in viewed_actions:
            viewed_actions.add(action_key)
            investigation_count = len(viewed_actions)
            decay_factor = 1.0 / (1 + (investigation_count - 1) * 0.3)
            if action.service_name:
                svc_data = current_snapshot["services"].get(action.service_name, {})
                if svc_data.get("health") in ("degraded", "down"):
                    reward += 0.04 * decay_factor
                else:
                    reward += 0.01 * decay_factor
            else:
                reward += 0.02 * decay_factor
        else:
            # Stronger penalty for consecutive repeat of same view action
            current_action_key = f"{action.action_type.value}:{action.service_name or 'global'}"
            if last_action_key and current_action_key == last_action_key:
                reward -= 0.03  # Consecutive spam = harsh penalty
            else:
                reward -= 0.01  # Non-consecutive repeat = mild penalty

    # 6. Repeated exact action penalty (non-view actions)
    if action.action_type not in (ActionType.VIEW_PIPELINE, ActionType.VIEW_LOGS, ActionType.VIEW_CONFIG):
        current_action_key = f"{action.action_type.value}:{action.service_name or ''}"
        if last_action_key and current_action_key == last_action_key:
            reward -= 0.02

    # 7. Apply task urgency scaling and bound
    reward *= TASK_URGENCY.get(task_name, 1.0)
    return max(min(reward, 0.30), -0.35)
