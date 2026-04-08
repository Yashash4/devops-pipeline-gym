---
title: DevOps Pipeline Environment
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# DevOps Pipeline Environment

## Overview

This environment enables training AI agents for automated DevOps incident management — an AI SRE agent that can diagnose failures, manage deployments, and make judgment calls under production pressure. It simulates a realistic microservice architecture where services have interdependent health metrics, cascading failures propagate through dependency chains, and every action has trade-off consequences.

CI/CD deployment management is the most common engineering workflow at companies like Meta, Google, and Amazon. This environment captures the real decision-making complexity of production deployments: flaky tests vs real bugs, config errors that only surface in staging, cascading failures that spiral through dependency chains, and production incidents where every minute of downtime costs revenue. The agent must investigate before acting, fix root causes before symptoms, and accept that every intervention has side effects.

This environment is useful for training RL agents to assist with deployment workflows, evaluating LLM reasoning under ambiguity and time pressure, and benchmarking investigation-before-action behavior. **Novel for OpenEnv**: No CI/CD pipeline environment exists on the Hub.

## Why This Environment Matters

CI/CD pipeline management is the most common engineering workflow at Meta, Google, and Amazon — every SRE team performs incident response daily. This environment fills a gap: no CI/CD pipeline environment exists on the OpenEnv Hub.

This environment trains agents to:
- **Investigate before acting** — partial observability forces information gathering
- **Diagnose root causes** — cascading failures require tracing through dependency chains
- **Make trade-off decisions** — every action has side effects (deploy spikes CPU, rollback risks regression)
- **Act under time pressure** — health degrades each step in incident tasks
- **Choose between valid strategies** — judgment_call has 3 resolution paths with different risk/reward profiles

The large gap between LLM baseline (0.18–0.70) and optimal (0.63–0.98) demonstrates significant room for RL training improvement across the full skill spectrum.

## Service Dependency Graph

```
database-primary (PostgreSQL — root, no dependencies)
├── auth-service (OAuth/JWT provider, depends on database-primary)
│   ├── api-gateway (router/load balancer, depends on database-primary + auth-service)
│   │   └── web-frontend (UI app, depends on api-gateway + auth-service)
│   └── web-frontend
└── cache-service (Redis cache, depends on database-primary)
```

Dependency chain: `database-primary → auth-service → api-gateway → web-frontend` and `database-primary → cache-service`. When an upstream service degrades, its dependents accumulate errors and latency each step.

## Tasks

### Task 1: Clean Deploy (Easy)
Deploy 2 services (api-gateway v2.3.1, web-frontend v1.9.0) with all tests passing. No complications — tests basic pipeline execution and deployment sequencing.
- **Max steps**: 15
- **Services**: database-primary, auth-service, api-gateway, web-frontend
- **Key challenge**: Execute staging → production deployment flow without breaking healthy services

### Task 2: Broken Pipeline (Medium)
Diagnose test failures, fix a config error, run a migration, and deploy 3 services. Not all test failures are blocking — the agent must distinguish flaky tests from real bugs.
- **Max steps**: 20
- **Services**: database-primary, auth-service, api-gateway, web-frontend, cache-service
- **Key challenge**: Wrong Redis host in cache-service config, pending migration blocks api-gateway deploy, 3 test failures (2 flaky, 1 deprecated)

### Task 3: The Judgment Call (Hard)
Production incident — api-gateway at 1500ms latency and 12 errors/sec. A partially-tested hotfix (v2.3.2) is available. Multiple valid resolution paths with different risk/reward tradeoffs. Health degrades every step (time pressure).
- **Max steps**: 12
- **Services**: database-primary (under load), auth-service, api-gateway (degraded), web-frontend
- **Key challenge**: Three valid paths — deploy hotfix + fix auth config (expert, highest score), rollback (safe but loses features), hotfix only (partial fix). Each has cascading consequences on web-frontend.

### Task 4: Cascading Failure (Medium-Hard)
Root cause analysis across a dependency chain. cache-service is down due to a config error (max_connections: 5), dragging api-gateway and web-frontend down via cascading failures. Fixing downstream services while root cause persists is futile.
- **Max steps**: 15
- **Services**: database-primary, auth-service, cache-service (root cause), api-gateway (degraded), web-frontend (degrading)
- **Key challenge**: Identify root cause in cache-service config, fix it, then recover downstream services in dependency order

### Task 5: Capacity Crisis (Medium-Hard)
database-primary is approaching capacity limits under a traffic surge. CPU climbing, connection pool near saturation. The agent must act proactively before cascading failures begin — once the database goes down, recovery is extremely difficult.
- **Max steps**: 15
- **Services**: database-primary (stressed), auth-service, api-gateway, cache-service, web-frontend
- **Key challenge**: Proactive intervention — increase max_connections and shared_buffers before tipping points trigger cascading collapse

### Task 6: Random Incident (Variable — Procedural Generation)
Procedurally generated incident from a seed. The failing service (api-gateway, cache-service, auth-service, or web-frontend), failure type (config_error, degraded_performance, capacity_limit, memory_leak, or certificate_expiry), and severity (moderate or severe) are all randomized. 30% chance of compound incident (two services failing simultaneously). Different seeds produce different scenarios — infinite variation for curriculum learning.
- **Max steps**: 15
- **Services**: All 5 (one randomly failing)
- **Key challenge**: Read the task description to identify the failing service and failure type, investigate, diagnose, and fix — with no prior knowledge of what's broken

## Procedural Generation

The `random_incident` task generates unique scenarios from a seed, enabling:
- **Curriculum learning**: Start with easy seeds, progressively increase difficulty
- **Generalization testing**: Verify agents handle novel failure combinations
- **Infinite training data**: Every seed produces a different incident

Failure space: 5 failure types × 4 services × 2 severities = 40 primary configurations, with 30% compound incidents and randomized initial conditions — hundreds of unique scenarios.

## Action Space

9 typed action types via `PipelineAction`:

| Action | Description | Required Fields |
|--------|-------------|-----------------|
| `view_pipeline` | View overall pipeline status and service summary | — |
| `view_logs` | View recent logs for a service (reveals CPU/memory) | `service_name` |
| `view_config` | View current config key-value pairs | `service_name` |
| `edit_config` | Modify config key-value pairs (causes restart latency spike) | `service_name`, `config_edits` |
| `run_migration` | Execute a pending database migration | `migration_name` |
| `deploy` | Deploy service version to staging, then promote to production | `service_name`, `target_version` |
| `rollback` | Rollback service to previous version (25% regression risk) | `service_name` |
| `approve` | Approve current state and end episode | `reason` |
| `abort` | Abort deployment and end episode | `reason` |

## Observation Space

`PipelineObservation` provides the agent's view of the system:

- **summary**: One-line status — highlights degraded/down services at a glance (e.g., `"WARNING: api-gateway degraded (lat=1500ms, err=12.0/s)"` or `"All services nominal."`)
- **services**: List of `ServiceStatus` — name, health, version, error_rate, latency, active_connections, last_deploy_timestamp. **Partial observability**: CPU and memory are hidden (show 0.0) until the agent runs `view_logs` for that service.
- **task_description** and **goal**: Natural language context for the current task
- **available_actions**: Context-sensitive list of valid action types
- **last_action_result / last_action_error**: Feedback from the previous step
- **pipeline**: Current stage, commit SHA, test pass/fail counts, build logs snippet
- **migrations**: Pending and applied migrations
- **active_alerts**: Critical/warning/info alerts with timestamps
- **config_snapshot**: Config key-value pairs (populated after `view_config` or `edit_config`)
- **step_number / max_steps**: Current progress

## Reward Design

Dense per-step reward that creates a learnable gradient for RL training. Investigation rewards use **diminishing-returns exploration** — first investigation of an unhealthy service gives +0.04, with decay as more services are investigated. Health improvements give proportional reward via system health delta (+0.005 per 1% improvement). **Sub-goal milestones** reward intermediate progress: config fixes (+0.08), migrations (+0.06), and alert resolution (+0.03). Breaking healthy services is heavily penalized (-0.30). All grading is outcome-based — no procedure-based criteria. Rewards are **task-adaptive** — harder tasks with time pressure get steeper gradients (1.0x–1.5x urgency scaling), creating a curriculum-aware reward landscape. Rewards are bounded [-0.35, +0.30] per step to prevent training instability.

| Signal | Reward | Condition |
|--------|--------|-----------|
| Service deployed to production | +0.15 | Service reaches prod successfully |
| Service verified in staging | +0.05 | Staging health check passes |
| Config error fixed | +0.08 | Service health improved after config change |
| Migration completed | +0.06 | Pending migration count decreased |
| Alert resolved | +0.03 | Alert count decreased |
| Investigation (degraded svc) | +0.04× | First-time view on unhealthy service (with decay) |
| Investigation (healthy svc) | +0.01× | First-time view on healthy service (with decay) |
| Health improvement | +0.005/1% | System health delta |
| Broke healthy service | -0.30 | Service went from healthy to degraded/down |
| Repeated investigation | -0.01/-0.03 | Same view on same target (-0.03 if consecutive) |
| Repeated exact action | -0.02 | Same action_type + service as last step |

### Reward Shaping Theory

The health delta component (+0.005 per 1% system health improvement) approximates potential-based reward shaping (Ng et al., 1999), where Φ(s) = system_health/100. This preserves optimal policy while providing continuous gradient signal.

Investigation bonuses use count-based exploration decay: reward = base × 1/(1 + 0.3n), consistent with Bellemare et al. (2016). This incentivizes initial exploration while preventing reward hacking through repeated view actions.

Task urgency scaling (1.0×–1.5×) implements curriculum-aware reward calibration — harder tasks receive steeper gradients to maintain learning signal despite longer optimal trajectories.

## Baseline Scores

Model: `Qwen/Qwen2.5-72B-Instruct` via HuggingFace Router

| Task | Difficulty | LLM Baseline | Optimal | Gap |
|------|-----------|-------------|---------|-----|
| clean_deploy | Easy | 0.700 | 0.947 | +0.247 |
| broken_pipeline | Medium | 0.482 | 0.890 | +0.408 |
| judgment_call | Hard | 0.184 | 0.935 | +0.751 |
| cascading_failure | Med-Hard | 0.280 | 0.883 | +0.603 |
| capacity_crisis | Med-Hard | 0.250 | 0.634 | +0.384 |
| random_incident (seed 6006) | Variable | 0.350 | 0.982 | +0.632 |

LLM baselines re-calibrated after environment tuning (v2). Optimal scores from scripted expert trajectories. The large gap between LLM baseline and optimal demonstrates significant room for RL training improvement — the environment produces meaningful reward signal across the full skill spectrum. The `random_incident` task generates unique scenarios from each seed, enabling curriculum learning.

### Seed Curriculum for RL Training

The `random_incident` task generates unique scenarios from each seed via `DEVOPS_SEED` env var. With 5 failure types, compound incidents, and randomized initial conditions, the configuration space produces hundreds of distinct scenarios.

**Recommended curriculum:**
- Seeds 1–20: Single-service failures, moderate severity (warm-up)
- Seeds 21–60: Mix of single and compound incidents (core training)
- Seeds 61–100: Severe failures with compound incidents (advanced)

Set `DEVOPS_SEED` at reset time (reads from env var each episode).

## Example Episode Trajectory

**Task: broken_pipeline** — diagnose and fix a broken deployment pipeline.

```
Step 1: view_logs("cache-service")       → reward +0.02  (investigation bonus, reveals Redis config error)
Step 2: edit_config("cache-service",
          redis.host → "redis-prod...")   → reward +0.10  (health improvement from fixing config)
Step 3: deploy("api-gateway", "v2.3.1")  → reward +0.05  (staging verified)
Step 4: deploy("api-gateway", "v2.3.1")  → reward +0.15  (promoted to production)
Step 5: approve("All services healthy")  → reward +0.03  (episode complete)
```

## Environment Features

- 6 tasks (5 hand-crafted + 1 procedurally generated) for curriculum learning
- 5 microservices with realistic dependency graph
- Stochastic simulation with seeded RNG for full reproducibility
- Realistic production logs (Java/Node stack traces, timestamps, red herrings)
- Partial observability (CPU/memory hidden until investigated via view_logs)
- Cascading failures propagate through dependency chain each step
- Cross-metric compounding (error → CPU → latency spirals, and reverse recovery)
- Non-linear tipping points (CPU cliff at 85%, latency cliff at 2000ms)
- Trade-off effects on every action (deploy → CPU spike, rollback → 25% regression risk, config edit → restart latency)
- Time pressure on incident tasks (health degrades each step in judgment_call)
- Multi-path task design (judgment_call has 3 valid resolution paths with different scores)
- Dense per-step reward with anti-reward-hacking safeguards (bounded, no procedure bonuses)
- Observation summary field for quick triage

## Formal MDP Description

| Property | Description |
|----------|-------------|
| **State space S** | 5 services × (health ∈ {healthy, degraded, down}, cpu ∈ [0,100], memory ∈ [0,100], error_rate ∈ [0,50], latency ∈ [0,5000], config: Dict) + pipeline status + migration status + alerts. Partially observable — CPU/memory/latency/error_rate hidden until investigated. |
| **Action space A** | 9 discrete action types × parameterized service targets. ~45 effective actions. |
| **Transition T** | Deterministic core with stochastic elements (8% transient staging failure, 25% rollback regression, deploy quality variance). Seeded RNG ensures reproducibility. |
| **Reward R** | Dense, bounded [-0.35, +0.30] per step. Potential-based health delta + milestone rewards + exploration bonuses with diminishing returns. Task-adaptive urgency scaling (1.0×–1.5×). |
| **Episode length** | 12–20 steps depending on task. Terminates on approve/abort/max_steps/catastrophic failure (health < 20%). |
| **Discount factor** | Recommended γ = 0.99 (short episodes, dense rewards). |

### Stochastic Elements (All Seeded)

All randomness uses `random.Random(seed)` — same seed + same actions = identical outcomes.

| Element | Probability | Location |
|---------|------------|----------|
| Transient staging failure | 8% per deploy | `deploy_to_staging()` |
| Rollback regression | 25% per rollback | `rollback()` |
| Deploy quality | 70% clean / 20% minor / 10% unstable | `deploy_to_production()` |
| Compound incident | 30% in random_incident | `RandomIncidentScenario.setup()` |
| Initial health variance | ±10 CPU, ±15 latency | `RandomIncidentScenario.setup()` |

Determinism guarantee: `reset()` re-seeds the RNG from fixed task seeds. Two resets with the same task produce identical initial states.

## Setup

```bash
# Install dependencies
uv sync

# Run locally (without Docker)
uv run python -m uvicorn server.app:app --host 0.0.0.0 --port 8000

# Build and run with Docker
docker build -t devops-pipeline-env .
docker run -p 8000:8000 devops-pipeline-env

# Test reset endpoint
curl -X POST -H "Content-Type: application/json" -d '{}' http://localhost:8000/reset

# Open web UI
# http://localhost:8000/web

# Run inference
export HF_TOKEN=your_token_here
uv run inference.py

# Validate and deploy
openenv validate
openenv push --repo-id your-username/devops-pipeline-env
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Reset environment (new episode) |
| `/step` | POST | Execute an action, returns observation |
| `/state` | GET | Get current environment state |
| `/tasks` | GET | List available tasks and action schema |
| `/health` | GET | Health check |
| `/baseline` | POST | Pre-recorded LLM baseline scores |
| `/grader` | POST | Score the current active episode |
| `/ws` | WS | WebSocket for persistent sessions |
| `/web` | GET | Gradio web interface |
