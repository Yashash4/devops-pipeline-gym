# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Deterministic graders for the DevOps Pipeline Environment.

Each grader produces a score in [0.0, 1.0].
Same actions -> same score. Always.
All criteria are outcome-based — no procedure bonuses.
"""


def grade_clean_deploy(episode_history, engine):
    """
    Task 1 grader:
    - 0.50 * (services at target version in prod / services with targets)
    - 0.30 * (final system_health / 100)
    - 0.20 * max(0, 1 - steps_used / (max_steps * 2))
    """
    target_services = [s for s in engine.services.values() if s.target_version]
    deployed_count = sum(
        1 for svc in target_services
        if svc.prod_deployed and svc.current_version == svc.target_version
    )

    deploy_ratio = deployed_count / len(target_services) if target_services else 0.0
    system_health = engine.get_system_health()

    steps_used = len(episode_history)
    max_steps = 15
    efficiency = max(0.0, 1.0 - steps_used / (max_steps * 2))

    score = 0.50 * deploy_ratio + 0.30 * (system_health / 100.0) + 0.20 * efficiency
    return min(max(score, 0.0), 1.0)


def grade_broken_pipeline(episode_history, engine):
    """
    Task 2 grader (all outcome-based):
    - 0.30 — cache-service config redis.host == redis-prod.internal:6379
    - 0.15 — migration applied (add_index_users_email in applied list)
    - 0.30 — (services at target in prod / 3)
    - 0.15 — (final system_health / 100)
    - 0.10 — step efficiency: max(0, 1 - steps_used / (max_steps * 2))
    """
    score = 0.0

    # Config fix outcome — is the config correct at end of episode?
    cache_svc = engine.services.get("cache-service")
    if cache_svc and cache_svc.config.get("redis.host") == "redis-prod.internal:6379":
        score += 0.30

    # Migration outcome — was the migration applied?
    if "add_index_users_email" in engine.migrations_applied:
        score += 0.15

    # Services at target in prod (only count services with deploy targets)
    target_services = [s for s in engine.services.values() if s.target_version]
    deployed_count = sum(
        1 for svc in target_services
        if svc.prod_deployed and svc.current_version == svc.target_version
    )
    if target_services:
        score += 0.30 * (deployed_count / len(target_services))

    # System health
    system_health = engine.get_system_health()
    score += 0.15 * (system_health / 100.0)

    # Step efficiency
    steps_used = len(episode_history)
    max_steps = 20
    efficiency = max(0.0, 1.0 - steps_used / (max_steps * 2))
    score += 0.10 * efficiency

    return min(max(score, 0.0), 1.0)


def grade_judgment_call(episode_history, engine):
    """
    Task 3 grader — 3 valid paths with different scores:
    - Expert path: deploy hotfix + fix web-frontend auth config -> best score
    - Hotfix only: deploy hotfix without fixing auth -> partial score
    - Rollback: safe but loses features -> moderate score

    Dimensions:
    - 0.30 — incident resolution (path-aware)
    - 0.25 — web-frontend collateral damage
    - 0.15 — time to resolution
    - 0.15 — no new issues introduced
    - 0.15 — final system health
    """
    score = 0.0
    api_gw = engine.services.get("api-gateway")
    web_fe = engine.services.get("web-frontend")

    # Detect which path the agent took
    deployed_hotfix = False
    did_rollback = False
    fixed_auth_config = False
    for entry in episode_history:
        action = entry.get("action", {})
        if action.get("action_type") == "deploy" and action.get("service_name") == "api-gateway":
            if action.get("target_version") == "v2.3.2":
                deployed_hotfix = True
        if action.get("action_type") == "rollback" and action.get("service_name") == "api-gateway":
            did_rollback = True
        if action.get("action_type") == "edit_config" and action.get("service_name") == "web-frontend":
            edits = action.get("config_edits", [])
            for edit in edits:
                if edit.get("key") == "api.auth_version" and edit.get("value") == "v2":
                    fixed_auth_config = True

    # Incident resolution (path-aware)
    resolved = False
    if api_gw:
        api_healthy = api_gw.health.value == "healthy" and api_gw.error_rate < 5.0
        api_fully_resolved = api_gw.latency_ms < 100 and api_gw.error_rate < 1.0
        if deployed_hotfix and fixed_auth_config and api_healthy:
            score += 0.30  # Expert path: full credit (root cause fixed + auth handled)
            resolved = True
        elif api_fully_resolved:
            resolved = True
            if did_rollback:
                score += 0.20  # Rollback: safe but lost features
            else:
                score += 0.25  # Some other resolution
        elif deployed_hotfix and api_healthy:
            score += 0.15  # Hotfix without auth fix: partial
            resolved = True
        elif api_gw.latency_ms < 500:
            score += 0.10  # Partial improvement

    # web-frontend collateral damage
    if web_fe:
        if web_fe.health.value == "healthy" and web_fe.error_rate < 2.0:
            score += 0.25
        elif web_fe.error_rate < 10.0:
            score += 0.10

    # Time to resolution
    resolution_step = len(episode_history)
    if resolved:
        for entry in episode_history:
            action = entry.get("action", {})
            if action.get("action_type") in ("deploy", "rollback"):
                resolution_step = entry.get("step", len(episode_history))
                break
    score += max(0.0, 1.0 - resolution_step / 10.0) * 0.15

    # No new issues introduced
    new_issues = 0
    for entry in episode_history:
        if entry.get("broke_healthy", False):
            new_issues += 1
    if new_issues == 0:
        score += 0.15
    elif new_issues == 1:
        score += 0.05

    # System health
    system_health = engine.get_system_health()
    score += 0.15 * (system_health / 100.0)

    return min(max(score, 0.0), 1.0)


def grade_cascading_failure(episode_history, engine):
    """
    Task 4 grader (all outcome-based):
    - 0.30 — root cause fixed (cache-service healthy AND max_connections != "5")
    - 0.25 — all services deployed to prod at target version
    - 0.20 — final system_health / 100 (only full marks if > 90%)
    - 0.15 — dependency health (cache-service healthy when api-gateway deployed)
    - 0.10 — step efficiency: max(0, 1 - steps_used / (max_steps * 2))
    """
    score = 0.0

    # Root cause fixed: cache-service healthy with correct config
    cache_svc = engine.services.get("cache-service")
    if cache_svc:
        cache_healthy = cache_svc.health.value == "healthy"
        config_fixed = cache_svc.config.get("redis.max_connections") != "5"
        if cache_healthy and config_fixed:
            score += 0.30

    # All services deployed to prod at target version (only those with targets)
    target_services = [s for s in engine.services.values() if s.target_version]
    deployed_count = sum(
        1 for svc in target_services
        if svc.prod_deployed and svc.current_version == svc.target_version
    )
    if target_services:
        score += 0.25 * (deployed_count / len(target_services))

    # System health
    system_health = engine.get_system_health()
    score += 0.20 * (system_health / 100.0)

    # Dependency health outcome: was cache-service healthy when api-gateway deployed?
    # This is outcome-based: checks the STATE at deploy time, not step ordering.
    api_deployed_with_healthy_dep = False
    for entry in episode_history:
        action = entry.get("action", {})
        if action.get("action_type") == "deploy" and action.get("service_name") == "api-gateway":
            # Check if cache-service was healthy at this point
            # We check the engine's final state for cache-service health
            if cache_svc and cache_svc.health.value == "healthy":
                api_deployed_with_healthy_dep = True
                break
    # Also award if api-gateway was never deployed (agent focused on root cause only)
    # and cache-service ended healthy
    if not api_deployed_with_healthy_dep:
        api_gw = engine.services.get("api-gateway")
        if api_gw and api_gw.prod_deployed and cache_svc and cache_svc.health.value == "healthy":
            api_deployed_with_healthy_dep = True
    if api_deployed_with_healthy_dep:
        score += 0.15

    # Step efficiency
    steps_used = len(episode_history)
    max_steps = 15
    efficiency = max(0.0, 1.0 - steps_used / (max_steps * 2))
    score += 0.10 * efficiency

    return min(max(score, 0.0), 1.0)


GRADERS = {
    "clean_deploy": grade_clean_deploy,
    "broken_pipeline": grade_broken_pipeline,
    "judgment_call": grade_judgment_call,
    "cascading_failure": grade_cascading_failure,
}


def grade_task(task_name, episode_history, engine):
    """Grade an episode. Returns score in [0.0, 1.0]."""
    grader = GRADERS.get(task_name)
    if grader is None:
        return 0.0
    return grader(episode_history, engine)
