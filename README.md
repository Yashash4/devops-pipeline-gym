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

A **CI/CD deployment pipeline** environment for the OpenEnv platform where an AI agent manages deployments across interdependent microservices. The agent reads logs, interprets test results, edits configs, runs migrations, deploys services, and makes judgment calls under pressure.

**Novel for OpenEnv**: No CI/CD environment exists on the Hub.

## Motivation

CI/CD deployment management is the most common engineering workflow at companies like Meta, Google, and Amazon. This environment captures the real decision-making complexity of production deployments: flaky tests vs real bugs, config errors that only surface in staging, and production incidents where every minute costs revenue.

This environment is useful for:
- Training RL agents to assist with deployment workflows
- Evaluating LLM reasoning under ambiguity and time pressure
- Benchmarking agent investigation-before-action behavior

## Action Space

9 typed action types via `PipelineAction`:

| Action | Description | Required Fields |
|--------|-------------|-----------------|
| `view_pipeline` | View overall pipeline status | - |
| `view_logs` | View service logs | `service_name` |
| `view_config` | View service configuration | `service_name` |
| `edit_config` | Modify config key-value pairs | `service_name`, `config_edits` |
| `run_migration` | Execute a database migration | `migration_name` |
| `deploy` | Deploy service to staging/prod | `service_name`, `target_version` |
| `rollback` | Rollback service to previous version | `service_name` |
| `approve` | Approve and finalize deployment | `reason` |
| `abort` | Abort deployment | `reason` |

## Observation Space

`PipelineObservation` provides full system state:
- **task_description** and **goal**: Natural language context
- **services**: List of `ServiceStatus` (health, version, CPU, memory, error rate, latency)
- **pipeline**: Current stage, test results, build logs
- **migrations**: Pending/applied migrations
- **active_alerts**: Critical/warning alerts
- **available_actions**: Context-sensitive valid actions
- **last_action_result/error**: Feedback from previous step
- **config_snapshot**: Config key-value pairs (when viewing/editing)

## Tasks

### Task 1: "Clean Deploy" (Easy)
Deploy 2 services with all tests passing. No complications. Tests execution and basic pipeline management.
- **Max steps**: 15
- **Services**: database-primary (healthy), auth-service (healthy), api-gateway (v2.3.0 -> v2.3.1), web-frontend (v1.8.0 -> v1.9.0)
- **Dependency graph**: database-primary <- auth-service, database-primary + auth-service <- api-gateway, api-gateway + auth-service <- web-frontend

### Task 2: "Broken Pipeline" (Medium)
Diagnose test failures, fix a config error, run a migration, and deploy 3 services. Not all test failures are blocking -- the agent must distinguish flaky tests from real bugs.
- **Max steps**: 20
- **Services**: database-primary (healthy), auth-service (healthy), api-gateway, web-frontend, cache-service
- **Dependency graph**: database-primary <- auth-service, database-primary <- cache-service, database-primary + auth-service <- api-gateway, api-gateway + auth-service <- web-frontend
- **Challenges**: 3 test failures (2 flaky, 1 deprecated), wrong Redis host in cache-service config, pending migration blocks api-gateway

### Task 3: "The Judgment Call" (Hard)
Production incident with api-gateway at 1500ms latency and 12 errors/sec. A partially-tested hotfix is available. Multiple valid resolution paths with different risk/reward tradeoffs. Health degrades every step.
- **Max steps**: 12
- **Services**: database-primary (under load, CPU 72%), auth-service (healthy, warns about HS256->RS256 transition), api-gateway (degraded), web-frontend (healthy)
- **Dilemma**: Deploy untested hotfix (breaks web-frontend auth) vs rollback (loses web-frontend API endpoint) — every path has cascading consequences
- **Three valid resolution paths**: deploy hotfix + fix auth config (expert path), rollback (safe), or hotfix only (partial fix). Each scores differently.

### Task 4: "Cascading Failure" (Medium-Hard)
Root cause analysis across a dependency chain. cache-service is down (config error), dragging api-gateway and web-frontend down via cascading failures. Agent must identify and fix the root cause first — fixing downstream services while the root cause persists is futile.
- **Max steps**: 15
- **Services**: database-primary (healthy), auth-service (healthy), cache-service (root cause), api-gateway (degraded), web-frontend (degrading)
- **Challenge**: Fix cache-service config (max_connections: 5 -> 50), deploy cache-service, then recover downstream services in order

## Reward Design

Outcome-based rewards (never procedure-based):
- **+0.15** per service successfully deployed to production
- **+0.05** per service verified in staging
- **+0.02** for first-time investigation actions (view_pipeline, view_logs, view_config)
- **+0.005** per 1% system health improvement
- **-0.30** for breaking a healthy service (catastrophic penalty)
- **-0.01** for true no-ops (not investigation actions)

Actions have trade-off effects: deploys cause temporary CPU/latency spikes, rollbacks risk regression, config edits cause restart latency. Cross-metric compounding: degraded metrics worsen each other (error->CPU->latency spirals), healthy metrics help each other recover.

## Baseline Scores

Model: `Qwen/Qwen2.5-72B-Instruct` via HuggingFace Router

| Task | Steps | Score |
|------|-------|-------|
| clean_deploy | 6 | 0.585 |
| broken_pipeline | 6 | 0.482 |
| judgment_call | 6 | 0.184 |
| cascading_failure | 8 | 0.280 |

Scores from Qwen/Qwen2.5-72B-Instruct via HuggingFace Router.

## Setup

```bash
# Install dependencies
uv sync

# Build Docker image
docker build -t devops-pipeline-env .

# Run locally
docker run -p 8000:8000 devops-pipeline-env

# Test
curl -X POST -H "Content-Type: application/json" -d '{}' http://localhost:8000/reset

# Open web UI
# http://localhost:8000/web

# Run inference
export HF_TOKEN=your_token_here
uv run inference.py

# Deploy to HuggingFace Spaces
openenv push --repo-id your-username/devops-pipeline-env
```

## API Endpoints

- `POST /reset` - Reset environment (new episode)
- `POST /step` - Execute an action
- `GET /state` - Get current state
- `GET /tasks` - List tasks and action schema
- `GET /health` - Health check
- `POST /baseline` - Pre-recorded baseline scores
- `POST /grader` - Score current episode
- `WS /ws` - WebSocket for persistent sessions
- `GET /web` - Gradio web interface

## Project Structure

```
devops_pipeline_env/
├── __init__.py              # Exports: PipelineAction, PipelineObservation, DevopsPipelineEnv
├── models.py                # Pydantic models (extends OpenEnv Action/Observation)
├── client.py                # DevopsPipelineEnv(EnvClient)
├── openenv.yaml             # Environment metadata
├── pyproject.toml           # Package config + dependencies
├── inference.py             # LLM inference script (root)
├── Dockerfile               # Container image (root)
├── requirements.txt         # Server dependencies (root)
├── README.md                # This file
└── server/
    ├── __init__.py
    ├── app.py               # FastAPI app with create_app()
    ├── pipeline_environment.py  # Main Environment class
    ├── pipeline_engine.py   # Service state machine + simulation
    ├── scenarios.py         # Task 1-4 scenario definitions
    ├── graders.py           # 4 deterministic graders (0.0-1.0)
    └── rewards.py           # Outcome-based reward calculator
```
