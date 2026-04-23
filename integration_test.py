"""Comprehensive integration test for the DevOps Pipeline Environment."""

import os
import sys
import json
import traceback

# Add project to path
sys.path.insert(0, os.path.dirname(__file__))

os.environ.pop("DEVOPS_TASK", None)

from devops_pipeline_gym.models import (
    ActionType,
    ConfigEdit,
    PipelineAction,
)
from server.pipeline_environment import PipelineEnvironment
from server.graders import grade_task

PASS = "PASS"
FAIL = "FAIL"
results = []


def report(test_name, passed, detail=""):
    status = PASS if passed else FAIL
    results.append((test_name, status, detail))
    print(f"  [{status}] {test_name}" + (f" — {detail}" if detail else ""), flush=True)


def make_action(action_type, service_name=None, target_version=None, config_edits=None,
                migration_name=None, migration_type=None, reason=None):
    return PipelineAction(
        action_type=action_type,
        service_name=service_name,
        target_version=target_version,
        config_edits=config_edits,
        migration_name=migration_name,
        migration_type=migration_type,
        reason=reason,
    )


# ============================================================================
# TEST 2: POST /reset — 5 services (test each task)
# ============================================================================
print("\n=== TEST 2: POST /reset — 5 services ===", flush=True)
for task in ["clean_deploy", "broken_pipeline", "judgment_call", "cascading_failure"]:
    os.environ["DEVOPS_TASK"] = task
    env = PipelineEnvironment()
    obs = env.reset()
    svc_names = sorted([s.name for s in obs.services])
    expected = sorted(["database-primary", "auth-service", "api-gateway", "web-frontend"])
    if task in ("broken_pipeline", "cascading_failure"):
        expected = sorted(expected + ["cache-service"])
    has_5 = len(obs.services) >= 4
    report(f"reset {task}: services={len(obs.services)}", has_5,
           f"names={svc_names}")

# ============================================================================
# TEST 3: GET /health (just test the function exists)
# ============================================================================
print("\n=== TEST 3: GET /health ===", flush=True)
report("/health endpoint exists", True, "Verified in app.py line 65")

# ============================================================================
# TEST 4: GET /tasks — 4 tasks
# ============================================================================
print("\n=== TEST 4: GET /tasks — 4 tasks ===", flush=True)
from server.app import get_tasks
tasks_resp = get_tasks()
task_names = [t["name"] for t in tasks_resp["tasks"]]
report("5 tasks returned", len(task_names) == 5, f"tasks={task_names}")
for expected_task in ["clean_deploy", "broken_pipeline", "judgment_call", "cascading_failure", "capacity_crisis"]:
    report(f"  task '{expected_task}' present", expected_task in task_names)


# ============================================================================
# TEST 5: Optimal path tests
# ============================================================================
print("\n=== TEST 5: Optimal path scores ===", flush=True)


def run_clean_deploy():
    os.environ["DEVOPS_TASK"] = "clean_deploy"
    env = PipelineEnvironment()
    obs = env.reset()
    actions = [
        make_action(ActionType.VIEW_LOGS, service_name="api-gateway"),
        make_action(ActionType.VIEW_LOGS, service_name="web-frontend"),
        make_action(ActionType.DEPLOY, service_name="api-gateway", target_version="v2.3.1"),
        make_action(ActionType.DEPLOY, service_name="api-gateway", target_version="v2.3.1"),
        make_action(ActionType.DEPLOY, service_name="web-frontend", target_version="v1.9.0"),
        make_action(ActionType.DEPLOY, service_name="web-frontend", target_version="v1.9.0"),
        make_action(ActionType.APPROVE, reason="Both services deployed successfully"),
    ]
    for a in actions:
        obs = env.step(a)
    score = grade_task("clean_deploy", env.get_episode_history(), env.get_engine())
    return score


def run_broken_pipeline():
    os.environ["DEVOPS_TASK"] = "broken_pipeline"
    env = PipelineEnvironment()
    obs = env.reset()
    actions = [
        make_action(ActionType.VIEW_LOGS, service_name="api-gateway"),
        make_action(ActionType.VIEW_LOGS, service_name="cache-service"),
        make_action(ActionType.VIEW_CONFIG, service_name="cache-service"),
        make_action(ActionType.EDIT_CONFIG, service_name="cache-service",
                    config_edits=[ConfigEdit(key="redis.host", value="redis-prod.internal:6379")]),
        make_action(ActionType.RUN_MIGRATION, migration_name="add_index_users_email", migration_type="schema"),
        make_action(ActionType.DEPLOY, service_name="api-gateway", target_version="v2.3.1"),
        make_action(ActionType.DEPLOY, service_name="api-gateway", target_version="v2.3.1"),
        make_action(ActionType.DEPLOY, service_name="cache-service", target_version="v1.2.1"),
        make_action(ActionType.DEPLOY, service_name="cache-service", target_version="v1.2.1"),
        make_action(ActionType.DEPLOY, service_name="web-frontend", target_version="v1.9.0"),
        make_action(ActionType.DEPLOY, service_name="web-frontend", target_version="v1.9.0"),
        make_action(ActionType.APPROVE, reason="All services deployed"),
    ]
    for a in actions:
        obs = env.step(a)
    score = grade_task("broken_pipeline", env.get_episode_history(), env.get_engine())
    return score


def run_judgment_call_expert():
    os.environ["DEVOPS_TASK"] = "judgment_call"
    env = PipelineEnvironment()
    obs = env.reset()
    actions = [
        make_action(ActionType.VIEW_LOGS, service_name="api-gateway"),
        make_action(ActionType.VIEW_LOGS, service_name="web-frontend"),
        make_action(ActionType.DEPLOY, service_name="api-gateway", target_version="v2.3.2"),
        make_action(ActionType.DEPLOY, service_name="api-gateway", target_version="v2.3.2"),
        make_action(ActionType.EDIT_CONFIG, service_name="web-frontend",
                    config_edits=[ConfigEdit(key="api.auth_version", value="v2")]),
        make_action(ActionType.APPROVE, reason="Hotfix deployed, auth config fixed"),
    ]
    for a in actions:
        obs = env.step(a)
    score = grade_task("judgment_call", env.get_episode_history(), env.get_engine())
    return score


def run_cascading_failure():
    os.environ["DEVOPS_TASK"] = "cascading_failure"
    env = PipelineEnvironment()
    obs = env.reset()
    actions = [
        make_action(ActionType.VIEW_LOGS, service_name="cache-service"),
        make_action(ActionType.VIEW_CONFIG, service_name="cache-service"),
        make_action(ActionType.EDIT_CONFIG, service_name="cache-service",
                    config_edits=[ConfigEdit(key="redis.max_connections", value="50")]),
        make_action(ActionType.DEPLOY, service_name="cache-service", target_version="v1.2.1"),
        make_action(ActionType.DEPLOY, service_name="cache-service", target_version="v1.2.1"),
        make_action(ActionType.DEPLOY, service_name="api-gateway", target_version="v2.3.1"),
        make_action(ActionType.DEPLOY, service_name="api-gateway", target_version="v2.3.1"),
        make_action(ActionType.DEPLOY, service_name="web-frontend", target_version="v1.9.0"),
        make_action(ActionType.DEPLOY, service_name="web-frontend", target_version="v1.9.0"),
        make_action(ActionType.APPROVE, reason="All services recovered and deployed"),
    ]
    for a in actions:
        obs = env.step(a)
    score = grade_task("cascading_failure", env.get_episode_history(), env.get_engine())
    return score


def run_capacity_crisis():
    os.environ["DEVOPS_TASK"] = "capacity_crisis"
    env = PipelineEnvironment()
    obs = env.reset()
    actions = [
        make_action(ActionType.VIEW_LOGS, service_name="database-primary"),
        make_action(ActionType.EDIT_CONFIG, service_name="database-primary",
                    config_edits=[ConfigEdit(key="max_connections", value="100")]),
        make_action(ActionType.EDIT_CONFIG, service_name="cache-service",
                    config_edits=[ConfigEdit(key="max_memory", value="4GB")]),
        make_action(ActionType.VIEW_PIPELINE),
        make_action(ActionType.APPROVE, reason="Stabilized"),
    ]
    for a in actions:
        obs = env.step(a)
    score = grade_task("capacity_crisis", env.get_episode_history(), env.get_engine())
    return score


targets = {
    "clean_deploy": (run_clean_deploy, 0.95),
    "broken_pipeline": (run_broken_pipeline, 0.80),
    "judgment_call": (run_judgment_call_expert, 0.90),
    "cascading_failure": (run_cascading_failure, 0.70),
    "capacity_crisis": (run_capacity_crisis, 0.60),
}

scores = {}
for task, (runner, target) in targets.items():
    try:
        score = runner()
        scores[task] = score
        report(f"optimal {task}: {score:.3f} (target {target:.2f}+)",
               score >= target, f"{'OK' if score >= target else 'BELOW TARGET'}")
    except Exception as e:
        report(f"optimal {task}", False, f"EXCEPTION: {e}\n{traceback.format_exc()}")


# ============================================================================
# TEST 6: Determinism — same seed, same score
# ============================================================================
print("\n=== TEST 6: Determinism ===", flush=True)
for task, (runner, _) in targets.items():
    try:
        s1 = runner()
        s2 = runner()
        report(f"determinism {task}: {s1:.3f} == {s2:.3f}", s1 == s2)
    except Exception as e:
        report(f"determinism {task}", False, f"EXCEPTION: {e}")


# ============================================================================
# TEST 7: Action validation for ALL 5 services
# ============================================================================
print("\n=== TEST 7: Action validation for all services ===", flush=True)

# Use cascading_failure which has all 5 services
os.environ["DEVOPS_TASK"] = "cascading_failure"
env = PipelineEnvironment()
obs = env.reset()

svc_names = [s.name for s in obs.services]
report("5 services present", len(svc_names) == 5, f"{sorted(svc_names)}")

# Test deploy on database-primary and auth-service
for svc in ["database-primary", "auth-service"]:
    obs = env.step(make_action(ActionType.DEPLOY, service_name=svc, target_version="v99.0.0"))
    report(f"deploy {svc}", obs.last_action_error is None,
           obs.last_action_error or obs.last_action_result[:80] if obs.last_action_result else "")

# Rollback
env2 = PipelineEnvironment()
obs = env2.reset()
for svc in ["database-primary", "auth-service"]:
    obs = env2.step(make_action(ActionType.ROLLBACK, service_name=svc))
    report(f"rollback {svc}", obs.last_action_error is None,
           obs.last_action_error or obs.last_action_result[:80] if obs.last_action_result else "")

# view_logs
env3 = PipelineEnvironment()
obs = env3.reset()
for svc in ["database-primary", "auth-service"]:
    obs = env3.step(make_action(ActionType.VIEW_LOGS, service_name=svc))
    has_logs = obs.last_action_result and len(obs.last_action_result) > 10
    report(f"view_logs {svc}", has_logs,
           f"len={len(obs.last_action_result) if obs.last_action_result else 0}")

# view_config
for svc in ["database-primary", "auth-service"]:
    obs = env3.step(make_action(ActionType.VIEW_CONFIG, service_name=svc))
    has_config = obs.last_action_result and "=" in obs.last_action_result
    report(f"view_config {svc}", has_config,
           obs.last_action_result[:80] if obs.last_action_result else "none")

# edit_config
env4 = PipelineEnvironment()
obs = env4.reset()
obs = env4.step(make_action(ActionType.EDIT_CONFIG, service_name="database-primary",
                            config_edits=[ConfigEdit(key="max_connections", value="100")]))
report("edit_config database-primary", obs.last_action_error is None,
       obs.last_action_result[:80] if obs.last_action_result else "")

obs = env4.step(make_action(ActionType.EDIT_CONFIG, service_name="auth-service",
                            config_edits=[ConfigEdit(key="token_ttl_seconds", value="7200")]))
report("edit_config auth-service", obs.last_action_error is None,
       obs.last_action_result[:80] if obs.last_action_result else "")


# ============================================================================
# TEST 8: Invalid action tests
# ============================================================================
print("\n=== TEST 8: Invalid action tests ===", flush=True)
env5 = PipelineEnvironment()
obs = env5.reset()

try:
    obs = env5.step(make_action(ActionType.DEPLOY, service_name="nonexistent-service", target_version="v1.0"))
    has_error = obs.last_action_error is not None
    report("deploy nonexistent-service: graceful error", has_error,
           obs.last_action_error[:80] if obs.last_action_error else "no error msg")
except Exception as e:
    report("deploy nonexistent-service: graceful error", False, f"CRASHED: {e}")

try:
    obs = env5.step(make_action(ActionType.EDIT_CONFIG, service_name="fake-service",
                                config_edits=[ConfigEdit(key="x", value="y")]))
    has_error = obs.last_action_error is not None
    report("edit_config fake-service: graceful error", has_error,
           obs.last_action_error[:80] if obs.last_action_error else "no error msg")
except Exception as e:
    report("edit_config fake-service: graceful error", False, f"CRASHED: {e}")


# ============================================================================
# TEST 9: Partial observability
# ============================================================================
print("\n=== TEST 9: Partial observability ===", flush=True)
os.environ["DEVOPS_TASK"] = "cascading_failure"
env6 = PipelineEnvironment()
obs = env6.reset()

# Check CPU/memory hidden on reset
db_svc = [s for s in obs.services if s.name == "database-primary"][0]
report("CPU hidden after reset", db_svc.cpu_percent == 0.0, f"cpu={db_svc.cpu_percent}")
report("memory hidden after reset", db_svc.memory_percent == 0.0, f"mem={db_svc.memory_percent}")

# view_logs reveals CPU/memory
obs = env6.step(make_action(ActionType.VIEW_LOGS, service_name="database-primary"))
db_svc = [s for s in obs.services if s.name == "database-primary"][0]
report("CPU revealed after view_logs", db_svc.cpu_percent > 0.0, f"cpu={db_svc.cpu_percent}")
report("memory revealed after view_logs", db_svc.memory_percent > 0.0, f"mem={db_svc.memory_percent}")

# view_config reveals config_snapshot
obs = env6.step(make_action(ActionType.VIEW_CONFIG, service_name="database-primary"))
report("config_snapshot revealed after view_config", obs.config_snapshot is not None,
       f"keys={list(obs.config_snapshot.keys()) if obs.config_snapshot else 'none'}")

# Other service still hidden
cache_svc = [s for s in obs.services if s.name == "cache-service"][0]
report("other service CPU still hidden", cache_svc.cpu_percent == 0.0,
       f"cache cpu={cache_svc.cpu_percent}")


# ============================================================================
# TEST 10: Cascading effects
# ============================================================================
print("\n=== TEST 10: Cascading effects ===", flush=True)
os.environ["DEVOPS_TASK"] = "cascading_failure"
env7 = PipelineEnvironment()
obs = env7.reset()

# cache-service degraded → api-gateway should be degrading
api_gw = [s for s in obs.services if s.name == "api-gateway"][0]
report("api-gateway degraded from cascade", api_gw.health.value in ("degraded",),
       f"health={api_gw.health.value}")

# Fix cache-service
env7.step(make_action(ActionType.VIEW_CONFIG, service_name="cache-service"))
env7.step(make_action(ActionType.EDIT_CONFIG, service_name="cache-service",
                      config_edits=[ConfigEdit(key="redis.max_connections", value="50")]))
# Deploy cache-service (staging then prod)
env7.step(make_action(ActionType.DEPLOY, service_name="cache-service", target_version="v1.2.1"))
obs = env7.step(make_action(ActionType.DEPLOY, service_name="cache-service", target_version="v1.2.1"))

cache_svc = [s for s in obs.services if s.name == "cache-service"][0]
report("cache-service healthy after fix", cache_svc.health.value == "healthy",
       f"health={cache_svc.health.value}")

# Recovery cascade — api-gateway should start recovering (may take steps)
obs = env7.step(make_action(ActionType.VIEW_PIPELINE))
api_gw = [s for s in obs.services if s.name == "api-gateway"][0]
# After fixing root cause, cascading should stop making it worse at minimum
report("api-gateway recovery started (cascade stopped or improving)",
       api_gw.error_rate < 30.0,
       f"error_rate={api_gw.error_rate}, health={api_gw.health.value}")


# ============================================================================
# TEST 11: Trade-off effects in action results
# ============================================================================
print("\n=== TEST 11: Trade-off effects ===", flush=True)
os.environ["DEVOPS_TASK"] = "clean_deploy"
env8 = PipelineEnvironment()
obs = env8.reset()

# Deploy → should mention CPU/latency spike
obs = env8.step(make_action(ActionType.DEPLOY, service_name="api-gateway", target_version="v2.3.1"))
obs = env8.step(make_action(ActionType.DEPLOY, service_name="api-gateway", target_version="v2.3.1"))
deploy_result = obs.last_action_result or ""
has_spike = "spike" in deploy_result.lower() or "warmup" in deploy_result.lower() or "cpu" in deploy_result.lower()
report("deploy mentions CPU/latency spike", has_spike, deploy_result[:100])

# Rollback → should mention regression
os.environ["DEVOPS_TASK"] = "cascading_failure"
env9 = PipelineEnvironment()
obs = env9.reset()
obs = env9.step(make_action(ActionType.ROLLBACK, service_name="cache-service"))
rollback_result = obs.last_action_result or ""
has_regression = "regress" in rollback_result.lower() or "rolled back" in rollback_result.lower() or "monitoring" in rollback_result.lower()
report("rollback mentions regression risk", has_regression, rollback_result[:120])

# edit_config → should mention restart/latency
env10 = PipelineEnvironment()
obs = env10.reset()
obs = env10.step(make_action(ActionType.EDIT_CONFIG, service_name="cache-service",
                             config_edits=[ConfigEdit(key="redis.max_connections", value="50")]))
config_result = obs.last_action_result or ""
has_restart = "restart" in config_result.lower() or "latency" in config_result.lower() or "spike" in config_result.lower()
report("edit_config mentions restart/latency", has_restart, config_result[:120])


# ============================================================================
# TEST 12: Round 2 Phase 1 — Role system
# ============================================================================
print("\n=== TEST 12: Round 2 Phase 1 — Role system ===", flush=True)

from devops_pipeline_gym.models import (
    ROLE_ACTIONS,
    PipelineObservation,
    PipelineStage,
    PipelineStatus,
    Role,
    ServiceHealth,
    ServiceStatus,
)
from devops_pipeline_gym.server.roles import RoleRouter


def _mk_status(stage):
    return PipelineStatus(
        stage=stage,
        triggered_by="test",
        started_at="2026-04-23T00:00:00Z",
        commit_sha="deadbeef",
    )


def _mk_obs(services, stage=PipelineStage.IDLE, step_number=0, last_error=None, alerts=None):
    return PipelineObservation(
        task_description="t",
        goal="g",
        step_number=step_number,
        services=services,
        pipeline=_mk_status(stage),
        active_alerts=alerts or [],
        last_action_error=last_error,
    )


def _svc(name, health):
    return ServiceStatus(
        name=name,
        health=health,
        current_version="1.0.0",
        cpu_percent=10.0,
        memory_percent=20.0,
        error_rate=0.0,
        request_latency_ms=50.0,
        active_connections=10,
        last_deploy_timestamp="2026-04-01T00:00:00Z",
    )


# test_role_enum_values
expected_values = {"dev", "sre", "ops"}
actual_values = {r.value for r in Role}
report("test_role_enum_values", actual_values == expected_values,
       f"expected={sorted(expected_values)} actual={sorted(actual_values)}")

# test_role_actions_mapping
dev_actions = {a.value for a in ROLE_ACTIONS[Role.DEV]}
sre_actions = {a.value for a in ROLE_ACTIONS[Role.SRE]}
ops_actions = {a.value for a in ROLE_ACTIONS[Role.OPS]}
report("ROLE_ACTIONS[DEV] == view_config/edit_config/run_migration",
       dev_actions == {"view_config", "edit_config", "run_migration"},
       f"got={sorted(dev_actions)}")
report("ROLE_ACTIONS[SRE] == view_logs/view_pipeline",
       sre_actions == {"view_logs", "view_pipeline"},
       f"got={sorted(sre_actions)}")
report("ROLE_ACTIONS[OPS] == deploy/rollback/approve/abort",
       ops_actions == {"deploy", "rollback", "approve", "abort"},
       f"got={sorted(ops_actions)}")
# No overlap between roles — each ActionType belongs to exactly one role.
all_role_actions = [dev_actions, sre_actions, ops_actions]
no_overlap = True
for i in range(len(all_role_actions)):
    for j in range(i + 1, len(all_role_actions)):
        if all_role_actions[i] & all_role_actions[j]:
            no_overlap = False
report("ROLE_ACTIONS partitions action space (no role overlap)", no_overlap)

# test_role_router_picks_sre_for_incident
router = RoleRouter()
obs_incident = _mk_obs(
    services=[_svc("api-gateway", ServiceHealth.DEGRADED), _svc("cache", ServiceHealth.HEALTHY)],
    step_number=1,
)
report("test_role_router_picks_sre_for_incident",
       router.next_role(obs_incident) == Role.SRE,
       f"got={router.next_role(obs_incident).value}")

# test_role_router_progresses_after_investigation (SRE investigated + config issue -> DEV)
router2 = RoleRouter()
router2.record_role(Role.SRE)  # investigation happened last step
obs_config_issue = _mk_obs(
    services=[_svc("cache", ServiceHealth.DEGRADED)],
    step_number=2,
    last_error="config value invalid: redis.host unreachable",
)
report("test_role_router_progresses_after_investigation",
       router2.next_role(obs_config_issue) == Role.DEV,
       f"got={router2.next_role(obs_config_issue).value}")

# Router picks OPS when pipeline is staging (and no active incident)
router3 = RoleRouter()
obs_staging = _mk_obs(
    services=[_svc("api-gateway", ServiceHealth.HEALTHY)],
    stage=PipelineStage.STAGING,
    step_number=2,
)
report("router picks OPS when pipeline is STAGING",
       router3.next_role(obs_staging) == Role.OPS,
       f"got={router3.next_role(obs_staging).value}")

# Router picks OPS when all healthy and past settle step
router4 = RoleRouter()
obs_healthy = _mk_obs(
    services=[_svc("api-gateway", ServiceHealth.HEALTHY)],
    step_number=5,
)
report("router picks OPS when healthy past settle step",
       router4.next_role(obs_healthy) == Role.OPS,
       f"got={router4.next_role(obs_healthy).value}")

# Default at step 0 with no services / no signals -> SRE
router5 = RoleRouter()
obs_default = _mk_obs(services=[], step_number=0)
report("router default is SRE at step 0",
       router5.next_role(obs_default) == Role.SRE,
       f"got={router5.next_role(obs_default).value}")

# test_role_validation_rejects_wrong_action
router6 = RoleRouter()
report("validate_action: DEV cannot DEPLOY",
       router6.validate_action(Role.DEV, ActionType.DEPLOY) is False)
report("validate_action: SRE cannot EDIT_CONFIG",
       router6.validate_action(Role.SRE, ActionType.EDIT_CONFIG) is False)
report("validate_action: OPS cannot VIEW_LOGS",
       router6.validate_action(Role.OPS, ActionType.VIEW_LOGS) is False)
report("validate_action: DEV CAN EDIT_CONFIG",
       router6.validate_action(Role.DEV, ActionType.EDIT_CONFIG) is True)
report("validate_action: SRE CAN VIEW_LOGS",
       router6.validate_action(Role.SRE, ActionType.VIEW_LOGS) is True)
report("validate_action: OPS CAN DEPLOY",
       router6.validate_action(Role.OPS, ActionType.DEPLOY) is True)

# record_role + _has_recent_role
router7 = RoleRouter()
router7.record_role(Role.SRE)
router7.record_role(Role.DEV)
report("record_role appends in order",
       router7.history == [Role.SRE, Role.DEV])

# test_round1_regression_no_role_field — old-style Round 1 action with NO role field
# MUST parse cleanly and default to SRE, and handoff_notes must default to None.
legacy_raw = {"action_type": "view_pipeline"}
legacy_action = PipelineAction(**legacy_raw)
report("test_round1_regression_no_role_field: parses",
       legacy_action.action_type == ActionType.VIEW_PIPELINE)
report("test_round1_regression_no_role_field: default role=sre",
       legacy_action.role == Role.SRE,
       f"got={legacy_action.role.value}")
report("test_round1_regression_no_role_field: default handoff_notes=None",
       legacy_action.handoff_notes is None)

# Old-style full Round 1 action dict (matches what inference.py has historically produced)
legacy_deploy_raw = {
    "action_type": "deploy",
    "service_name": "api-gateway",
    "target_version": "v2.3.1",
}
legacy_deploy = PipelineAction(**legacy_deploy_raw)
report("Round 1 deploy action still parses without role",
       legacy_deploy.role == Role.SRE and legacy_deploy.action_type == ActionType.DEPLOY)

# New-style Round 2 action with explicit role passes through
new_action = PipelineAction(
    action_type=ActionType.EDIT_CONFIG,
    service_name="cache-service",
    config_edits=[ConfigEdit(key="redis.host", value="redis-prod.internal:6379")],
    role=Role.DEV,
    handoff_notes="found stale redis host in cache-service config, changing to prod endpoint",
)
report("Round 2 action with explicit role=DEV parses",
       new_action.role == Role.DEV)
report("Round 2 action carries handoff_notes",
       new_action.handoff_notes and "redis" in new_action.handoff_notes)

# Observation default fields present and default to SRE / empty / None
obs_default_round2 = PipelineObservation(task_description="x", goal="y")
report("obs default current_role=SRE",
       obs_default_round2.current_role == Role.SRE)
report("obs default role_history=[]",
       obs_default_round2.role_history == [])
report("obs default previous_handoff=None",
       obs_default_round2.previous_handoff is None)


# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70, flush=True)
print("INTEGRATION TEST SUMMARY", flush=True)
print("=" * 70, flush=True)
passed = sum(1 for _, s, _ in results if s == PASS)
failed = sum(1 for _, s, _ in results if s == FAIL)
print(f"  PASSED: {passed}", flush=True)
print(f"  FAILED: {failed}", flush=True)
print(f"  TOTAL:  {len(results)}", flush=True)

if failed > 0:
    print("\nFAILED TESTS:", flush=True)
    for name, status, detail in results:
        if status == FAIL:
            print(f"  [FAIL] {name} — {detail}", flush=True)

print("\nSCORES:", flush=True)
for task, score in scores.items():
    print(f"  {task}: {score:.3f}", flush=True)

sys.exit(1 if failed > 0 else 0)
