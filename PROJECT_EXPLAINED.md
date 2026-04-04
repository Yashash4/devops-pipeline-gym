# DevOps Pipeline Environment — Complete Project Guide

---

## Table of Contents

1. [The Big Picture](#the-big-picture)
2. [Key Concepts](#key-concepts)
3. [How the Pieces Fit Together](#how-the-pieces-fit-together)
4. [File-by-File Deep Dive](#file-by-file-deep-dive)
   - [openenv.yaml](#1-openenvyaml--the-id-card)
   - [models.py](#2-modelspy--the-shared-language)
   - [server/pipeline_engine.py](#3-serverpipeline_enginepy--the-simulation)
   - [server/scenarios.py](#4-serverscenariospy--the-3-levels)
   - [server/rewards.py](#5-serverrewardspy--per-step-scoring)
   - [server/graders.py](#6-servergraderspy--final-scoring)
   - [server/pipeline_environment.py](#7-serverpipeline_environmentpy--the-glue)
   - [server/app.py](#8-serverapppy--the-web-server)
   - [client.py](#9-clientpy--the-remote-control)
   - [__init__.py](#10-__init__py--the-export-list)
   - [inference.py](#11-inferencepy--the-ai-player)
   - [Dockerfile](#12-dockerfile--the-shipping-container)
   - [pyproject.toml](#13-pyprojecttoml--the-package-config)
   - [requirements.txt](#14-requirementstxt--the-dependency-list)
5. [The Game Loop](#the-game-loop)

---

## The Big Picture

### What is this project?

Imagine a **board game** where:

- The **game board** is a simulated CI/CD pipeline with services, configs, and migrations
- The **rules** define what moves you can make (deploy, rollback, view logs, etc.)
- The **score** tells you how well you played
- The **player** is an AI (a large language model like ChatGPT)

This project builds that board game. The AI looks at the board, decides a move, the board updates, the AI gets a score for that move, and repeats until the game ends.

### Why does this exist?

This is for the **Meta PyTorch OpenEnv Hackathon**. The goal: build a "training environment" where AI agents can practice real-world tasks. Our environment simulates **managing software deployments** — the kind of work real engineers at Google, Meta, and Amazon do every day.

An AI plays through scenarios like "deploy these services" or "fix this production outage" and gets scored on how well it does.

### What is OpenEnv?

OpenEnv is a **framework** (a set of tools) built by Meta for creating AI training environments. Think of it like a game engine (Unity, Unreal) but for AI agents instead of video games. It provides:

- A standard way to create environments (the "game world")
- A standard way for AI agents to interact with them (reset, step, observe)
- A web interface for humans to test
- Docker packaging for deployment

### What is Reinforcement Learning?

RL is how you train an AI by giving it **rewards** and **penalties**:

- AI does something good (deploys successfully) = positive reward (+0.15)
- AI does something bad (breaks a server) = negative reward (-0.30)
- Over time, the AI learns which actions lead to more rewards

Our project gives rewards after **every action** (not just at the end), which helps the AI learn faster.

---

## Key Concepts

### CI/CD Pipeline

CI/CD stands for Continuous Integration / Continuous Deployment. It's the process of:

1. Writing new code
2. Running tests on it
3. Deploying it to a "staging" (test) server
4. If tests pass, deploying to "production" (real users)
5. If something breaks, rolling back to the old version

### Microservices

Instead of one big application, modern companies split their software into small independent services. Our environment has services like `api-gateway`, `web-frontend`, and `cache-service`. Each can be deployed, rolled back, or configured independently.

### Staging vs Production

- **Staging** = test server. Looks like production but no real users.
- **Production** = the real thing. Actual users are affected.

Always deploy to staging first. If staging looks good, deploy to production.

### Database Migrations

When you change the structure of your database (add a column, add an index), you write a "migration" — a script that makes the change. Some deployments require running migrations first.

---

## How the Pieces Fit Together

```
                        THE ARCHITECTURE
                        ================

  inference.py          client.py           server/app.py
  (the AI brain)   -->  (the messenger) --> (the receptionist)
       |                                         |
       |                                    pipeline_environment.py
       |                                    (the game master)
       |                                         |
       |                              +----------+----------+
       |                              |                     |
       |                        pipeline_engine.py    scenarios.py
       |                        (the simulation)      (the levels)
       |                              |
       |                    +---------+---------+
       |                    |                   |
       |               rewards.py          graders.py
       |               (per-step score)    (final score)
       |
  models.py
  (the shared language - data structures everyone uses)
```

### The Flow

1. `inference.py` (the AI) reads the game state
2. The AI decides an action (like "deploy api-gateway")
3. The action goes through `client.py` over the network to `server/app.py`
4. `app.py` passes it to `pipeline_environment.py` (the game master)
5. The game master validates the action, tells `pipeline_engine.py` to execute it
6. `rewards.py` calculates the per-step reward
7. The game master builds a new observation and sends it back
8. The AI reads the new state and decides the next action
9. This repeats until the game ends
10. `graders.py` gives the final grade

---

## File-by-File Deep Dive

---

### 1. openenv.yaml — The ID Card

**What is it?** The identity card for the project. When OpenEnv tools look at our project, this is the first file they read. It answers: "What is this? How do I run it?"

**Why does it exist?** Without it, the OpenEnv platform doesn't recognize our folder as a valid environment. Think of it like the label on a product — the store can't shelve it without a label.

**Every line explained:**

| Line | Code | Meaning |
|------|------|---------|
| 1 | `spec_version: 1` | "I follow version 1 of the OpenEnv format" |
| 2 | `name: devops_pipeline_env` | The unique name of our environment |
| 3 | `type: space` | Will be deployed as a HuggingFace Space (web app) |
| 4 | `runtime: fastapi` | Uses FastAPI (a Python web framework) |
| 5 | `app: server.app:app` | "Start the server by loading the `app` object from `server/app.py`" |
| 6 | `port: 8000` | The server listens on port 8000 (like a room number in a building) |
| 7 | `description: "CI/CD..."` | Human-readable description |
| 8 | `version: "0.1.0"` | Our version number |

**Connections:** `openenv validate` reads this. `openenv push` reads this. The Dockerfile must match the port and app path.

**If removed:** `openenv validate` fails. Can't deploy. The entire toolchain stops.

---

### 2. models.py — The Shared Language

**What is it?** Defines every data structure in the project. It's the dictionary that everyone agrees on. When the AI says "deploy api-gateway to v2.3.1", this file defines exactly what that sentence looks like in code.

**Why does it exist?** Without shared data structures, different parts of the project would disagree on what data looks like. It's like making sure everyone speaks the same language.

#### Enums — Named Choices

Enums are dropdown menus. You can only pick from the listed options.

**ActionType** — The 9 things an AI can do:

| Action | What it does |
|--------|-------------|
| `view_pipeline` | See overall system status |
| `view_logs` | Read a service's log messages |
| `view_config` | See a service's configuration |
| `edit_config` | Change a config value |
| `run_migration` | Execute a database migration |
| `deploy` | Deploy a service to staging/production |
| `rollback` | Revert to previous version |
| `approve` | "I'm done, everything looks good" |
| `abort` | "Something is wrong, stop everything" |

**ServiceHealth** — 4 possible states for a service:

| State | Meaning |
|-------|---------|
| `healthy` | Working normally |
| `degraded` | Working but with problems (slow, errors) |
| `down` | Completely broken |
| `unknown` | Status unclear |

**PipelineStage** — Where the deployment process is:

| Stage | Meaning |
|-------|---------|
| `idle` | Nothing happening |
| `staging` | Deploying to test server |
| `deploying` | Deploying to production |
| `deployed` | Successfully deployed |
| `rolled_back` | Reverted to old version |
| `failed` | Something went wrong |

#### Sub-models — Helper Structures

**ConfigEdit** — When the AI wants to change a config setting:
```
key = "redis.host"  (WHICH setting)
value = "redis-prod.internal:6379"  (WHAT to change it to)
```

**ServiceStatus** — Dashboard for one service:
```
name: "api-gateway"
health: "healthy"
current_version: "v2.3.0"
cpu_percent: 35.0
memory_percent: 42.0
error_rate: 0.1  (errors per second)
request_latency_ms: 45.0  (how slow, in milliseconds)
active_connections: 120
last_deploy_timestamp: "2026-04-01T00:00:00Z"
```

**PipelineStatus** — Overall pipeline info:
```
stage: "idle"
triggered_by: "deploy-bot"
test_pass_count: 145
test_fail_count: 0
build_logs_snippet: "Build succeeded..."
```

**MigrationStatus** — Database migration info:
```
pending_migrations: ["add_index_users_email"]
last_applied: null
```

**AlertInfo** — Warning/critical alerts:
```
severity: "critical"
message: "api-gateway p95 latency >1000ms for 15 minutes"
service_name: "api-gateway"
```

#### The Two Main Classes

**PipelineAction** — What the AI SENDS (its "move"):

| Field | Type | Required? | Example |
|-------|------|-----------|---------|
| `action_type` | ActionType | Always | `"deploy"` |
| `service_name` | text | For deploy, view_logs, etc. | `"api-gateway"` |
| `target_version` | text | For deploy | `"v2.3.1"` |
| `config_edits` | list | For edit_config | `[{"key": "redis.host", "value": "..."}]` |
| `migration_name` | text | For run_migration | `"add_index_users_email"` |
| `reason` | text | For approve/abort | `"All services deployed"` |

`Optional` means the field can be left blank. Not every action needs every field.

Extends OpenEnv's `Action` base class — required by the hackathon rules.

**PipelineObservation** — What the AI SEES (the "game board"):

| Field | What it shows |
|-------|--------------|
| `task_description` | "Deploy these services to production" |
| `goal` | "System health above 95%" |
| `step_number` | Which step we're on (0, 1, 2...) |
| `max_steps` | Maximum steps allowed |
| `services` | List of all service statuses |
| `pipeline` | Pipeline stage, test results, build logs |
| `migrations` | Pending migrations |
| `active_alerts` | Critical/warning alerts |
| `available_actions` | What actions are valid right now |
| `last_action_result` | "Deployed successfully" or error message |
| `last_action_error` | Error details if action was invalid |
| `config_snapshot` | Config key-values (when viewing config) |

Extends OpenEnv's `Observation` base class.

**Connections:** Every other file imports from models.py. It's the shared vocabulary.

**If removed:** Everything breaks. Every file depends on these definitions.

---

### 3. server/pipeline_engine.py — The Simulation

**What is it?** The game board and rules. Contains the state of every service and knows how to update it when the AI takes an action.

**Why does it exist?** Without it, there's nothing to simulate. The AI would have no game to play.

#### Class: ServiceState — One Service

Represents one microservice (like one chess piece on the board).

**Constructor** — Creates a service with initial stats:
```python
ServiceState(
    name="api-gateway",
    version="v2.3.0",
    health=HEALTHY,
    config={"database.pool_size": "20", ...},
    latency_ms=45.0,
    error_rate=0.1,
    cpu=35.0,
    memory=42.0
)
```

Also tracks deployment progress with three booleans:
- `staging_deployed` — Has this version been deployed to test? (starts False)
- `staging_verified` — Did staging tests pass? (starts False)
- `prod_deployed` — Is this version live for real users? (starts False)

**deploy_to_staging(version, scenario)** — Deploy to the test server:
- If the scenario says there's a config error: health drops to DEGRADED, error rate spikes
- If config is fine: health stays HEALTHY, staging is verified
- This is how Task 2's cache-service bug surfaces

**deploy_to_production(version)** — Deploy to real users:
- If staging wasn't verified first: DEGRADED health (risky!)
- If staging was verified: HEALTHY, production deploy succeeds
- Teaches the AI: always test in staging first

**rollback()** — Emergency revert:
- Health goes back to HEALTHY
- Staging flags reset (would need to re-deploy to try again)
- Like pressing the undo button

**set_config(key, value)** — Change a config setting:
- Records what changed (old value -> new value)
- Adds a log entry

**to_status()** — Convert internal state to a ServiceStatus object for the AI to see

#### Class: PipelineEngine — The Game Board

Manages all services together plus pipeline state, migrations, and alerts.

**Constructor:**
- Takes a `scenario` (which task/level to play) and a `seed` (for reproducible randomness)
- Creates empty services dict, empty migrations list
- Calls `scenario.setup(self)` — the scenario populates the board

**execute(action)** — The main action router:
- If time pressure is active (Task 3), degrades api-gateway each step
- Routes each action type to the right handler method
- Returns a human-readable result string

**_deploy(service_name, version):**
- First call -> staging (via `deploy_to_staging`)
- Second call -> production (via `deploy_to_production`)
- Checks migration dependencies (Task 2: must migrate before deploying api-gateway)
- After production deploy, notifies scenario for cascading effects (Task 3: hotfix breaks web-frontend)

**_edit_config(service_name, edits):**
- Changes config values
- Checks if the fix resolved a config error
- If yes: restores service health, resets staging flags for re-deploy

**_rollback(service_name):**
- Reverts the service
- Notifies scenario for cascading effects (Task 3: rollback also breaks web-frontend)

**snapshot():**
- Takes a "photograph" of the entire system state
- Used by rewards.py to compare before/after each action

**get_system_health():**
- Calculates a single number 0-100 for overall health
- Each service starts at 100 (HEALTHY), 50 (DEGRADED), or 0 (DOWN)
- Subtract points for high error rates and high latency
- Average across all services

**_apply_time_pressure():**
- Only active during Task 3
- Each step: api-gateway latency +200ms, error rate +2.0/s, CPU +2%
- Creates real urgency — the AI can't waste steps investigating

**Connections:**
- scenarios.py calls `engine.setup()` to configure initial state
- pipeline_environment.py creates a PipelineEngine and calls `execute()` each step
- rewards.py uses `snapshot()` to compare before/after states

**If removed:** No simulation exists. reset() and step() have nothing to simulate.

---

### 4. server/scenarios.py — The 3 Levels

**What is it?** Defines the 3 tasks (levels) of the game. Each scenario sets up the game board differently.

**Why does it exist?** Without it, there's only one way to play. Three scenarios create easy/medium/hard challenges.

#### Base Class: Scenario

A template every task follows:

| Method | Default | Purpose |
|--------|---------|---------|
| `setup(engine)` | Must override | Place pieces on the board |
| `run_migration(engine, name)` | Returns True | Handle database migration |
| `migration_blocks_deploy(service)` | Returns False | Does migration block deploy? |
| `check_config_error(service, config)` | Returns False | Is there a config bug? |

#### Task 1: CleanDeployScenario (Easy)

**Setup:**
- 2 services: api-gateway (v2.3.0 -> v2.3.1), web-frontend (v1.8.0 -> v1.9.0)
- 145/145 tests pass
- No migrations
- No config errors
- 15 steps allowed

**What the AI should do:** Deploy each service to staging then production, approve.

**Why it's easy:** Nothing goes wrong. No traps, no surprises.

#### Task 2: BrokenPipelineScenario (Medium)

**Setup:**
- 3 services (adds cache-service)
- 142/145 tests pass (2 flaky, 1 deprecated — none blocking)
- cache-service has WRONG config: `redis.host` points to `redis-staging.internal:6379` instead of `redis-prod.internal:6379`
- Migration `add_index_users_email` must run before deploying api-gateway
- 20 steps allowed

**Special methods:**
- `migration_blocks_deploy("api-gateway")` returns True
- `check_config_error("cache-service", config)` returns True if redis.host is wrong

**What the AI should do:** Read logs, run migration, deploy api-gateway, deploy cache-service (fails), view config, fix config, redeploy, deploy web-frontend, approve.

**Why it's medium:** Must diagnose problems. Distinguish fake failures from real ones.

#### Task 3: JudgmentCallScenario (Hard)

**Setup:**
- api-gateway is DEGRADED in production (2000ms latency, 15 errors/sec)
- Hotfix v2.3.2 available but only smoke-tested
- Revenue bleeding at $500/minute
- Health degrades EVERY step
- Only 12 steps allowed

**The traps:**

| Path | Fixes api-gateway? | Breaks web-frontend? | Why? |
|------|-------------------|---------------------|------|
| Deploy hotfix v2.3.2 | Yes | Yes | Auth middleware refactor breaks web-frontend's auth |
| Rollback to v2.3.0 | Yes | Yes | Loses v2.3.1 API endpoint web-frontend depends on |
| Do nothing | No (gets worse) | No | But api-gateway keeps degrading |

**Special methods:**
- `on_prod_deploy`: When v2.3.2 deploys, web-frontend gets 401 errors
- `on_rollback`: When api-gateway rolls back, web-frontend gets 404 errors
- `check_config_error`: After hotfix, web-frontend needs `api.auth_version=v2`

**Why it's hard:** Every path has cascading consequences. Only 12 steps. Health degrades each step. Must fix api-gateway AND handle the web-frontend fallout.

**Connections:**
- pipeline_environment.py calls `load_scenario(task_name)` to get the right scenario
- The scenario's `setup()` configures the engine
- The engine calls scenario methods during gameplay

**If removed:** The environment has no tasks. reset() would crash.

---

### 5. server/rewards.py — Per-Step Scoring

**What is it?** Calculates the reward after EVERY SINGLE ACTION. The reward tells the AI: "that was good" (+0.15) or "that was terrible" (-0.30).

**Why does it exist?** Without per-step feedback, the AI can't learn which actions are good. It would be like playing a game with no score display until the very end.

#### The 5 Reward Rules

**Rule 1: Health delta** — Did overall system health improve?
```
health_delta = after_health - before_health
reward += health_delta * 0.005
```
Example: Health went from 80% to 90% = +10 * 0.005 = **+0.05**

**Rule 2: Deployment progress** — Did a service advance?
```
Service reached production = +0.15
Service passed staging     = +0.05
```

**Rule 3: Breaking things** — Did a healthy service get sick?
```
Healthy service became degraded or down = -0.30
```
This is **twice as punishing** as a successful deploy is rewarding. Teaches caution.

**Rule 4: Investigation bonus** — First time looking at something?
```
First time using view_pipeline/view_logs/view_config = +0.02
```
Encourages the AI to investigate before acting. Each unique view action gets the bonus only once per episode.

**Rule 5: No-op penalty** — Did nothing useful?
```
No state change + not an investigation action = -0.01
```
Discourages wasting steps.

#### Reward Examples

| Action | Reward | Why |
|--------|--------|-----|
| First `view_pipeline` | +0.02 | Investigation bonus |
| Deploy to staging (passes) | +0.05 | Staging verified |
| Deploy to production | +0.15 | Reached production |
| Deploy breaks a service | -0.30 | Catastrophic penalty |
| Second `view_pipeline` | -0.01 | Already viewed, no state change |
| Fix config (health improves) | +0.14 | Health delta (degraded -> healthy) |

**Connections:**
- pipeline_environment.py calls `calculate_reward()` after every step
- Uses snapshots from the engine's `snapshot()` method
- `viewed_actions` set is owned by pipeline_environment.py (per-session)

**If removed:** Every step has 0 reward. The AI gets no feedback. Scores are always 0.

---

### 6. server/graders.py — Final Scoring

**What is it?** At the END of a game, this gives a final grade from 0.0 (total failure) to 1.0 (perfect). While rewards.py scores each step, this scores the ENTIRE game.

**Why does it exist?** The hackathon judges need a single score to rank submissions. The grader answers: "how well did the AI actually do?"

#### Key Rule: DETERMINISTIC

Same actions must always produce the same score. No randomness. This is critical for reproducibility.

#### Task 1 Grader: grade_clean_deploy

```
Score = 0.50 * (deployed_services / total_services)
      + 0.30 * (system_health / 100)
      + 0.20 * max(0, 1 - steps / 30)
```

| Component | Weight | What it checks |
|-----------|--------|---------------|
| Deploy ratio | 50% | Did you deploy the services? |
| System health | 30% | Is the system still healthy? |
| Efficiency | 20% | Did you do it in few steps? |

#### Task 2 Grader: grade_broken_pipeline

| Component | Weight | What it checks |
|-----------|--------|---------------|
| Config fix | 30% | Is cache-service redis.host correct? |
| Migration | 15% | Was add_index_users_email applied? |
| Deploy ratio | 30% | How many services reached production? |
| System health | 15% | Final system health |
| Efficiency | 10% | Step count |

#### Task 3 Grader: grade_judgment_call

| Component | Weight | What it checks |
|-----------|--------|---------------|
| Incident resolved | 30% | api-gateway latency < 100ms AND error_rate < 1.0 |
| Web-frontend healthy | 25% | web-frontend still healthy (THE TRAP) |
| Time to resolution | 15% | How quickly did you act? |
| No new issues | 15% | Did you avoid breaking things? |
| System health | 15% | Final aggregate health |

**Why Task 3 is hard:** Getting 30% for fixing api-gateway is straightforward (rollback or hotfix). But getting 25% for keeping web-frontend healthy is nearly impossible without ALSO handling the cascading damage.

**Connections:**
- server/app.py uses `grade_task()` for the `/grader` endpoint
- pipeline_environment.py imports it for access via the class reference

**If removed:** The /grader endpoint always returns 0.0. Judges can't evaluate quality.

---

### 7. server/pipeline_environment.py — The Glue

**What is it?** The main environment class that ties everything together. It implements the two functions the OpenEnv framework requires: `reset()` (start a new game) and `step()` (make a move).

**Why does it exist?** The OpenEnv framework needs a class that follows its `Environment` interface. This class is the adapter between "our simulation" and "the framework's expectations."

#### Class: PipelineEnvironment

**Constructor (`__init__`):**
- Creates initial state with a unique episode ID
- Sets up empty history and viewed_actions set
- `_last_instance` (class-level variable) lets the /grader endpoint access the environment

**reset()** — Start a new game:
1. Read which task from `os.environ["DEVOPS_TASK"]` (defaults to "clean_deploy")
2. Generate new episode ID
3. Clear history and viewed_actions
4. Store `self` as `_last_instance` (for /grader)
5. Load the scenario for this task
6. Create the engine (game board)
7. Return the initial observation (what the AI sees first)

**step(action)** — Process one move:
1. Increment step count
2. Take "before" snapshot
3. Validate the action (is it well-formed? does the service exist?)
   - If invalid: return error with -0.05 penalty
4. Execute the action on the engine
5. Take "after" snapshot
6. Calculate reward by comparing before/after
7. Check if the game is over (approve? abort? max steps? catastrophic failure?)
8. Track if a healthy service broke (for the grader)
9. Record in episode history
10. Build and return the new observation

**_validate_action(action)** — Catches mistakes:
- `deploy` without `service_name` -> error
- `deploy` with unknown service name -> error
- `deploy` without `target_version` -> error
- `edit_config` without `config_edits` -> error

**_check_done(action)** — Is the game over?
- AI said "approve" -> yes
- AI said "abort" -> yes
- Reached max steps -> yes
- System health below 20% -> yes (catastrophic failure)

**_get_available_actions()** — What can the AI do right now?
- Always: view_pipeline, view_logs, approve, abort
- If services exist: view_config, edit_config, deploy, rollback
- If migrations pending: run_migration

**Connections:**
- server/app.py creates instances of this class via `create_app`
- Uses pipeline_engine.py for simulation
- Uses scenarios.py for task setup
- Uses rewards.py for per-step scoring
- Uses graders.py (indirectly, via /grader endpoint)

**If removed:** The environment doesn't exist. No reset(), no step(). Nothing works.

---

### 8. server/app.py — The Web Server

**What is it?** Turns the environment into a web server with HTTP endpoints. Anyone on the internet (or locally) can connect and play.

**Why does it exist?** The AI player (inference.py) runs separately from the environment. They communicate over the network. This file creates the network interface.

#### create_app (line 14)

```python
app = create_app(
    PipelineEnvironment,    # The game controller class
    PipelineAction,         # What moves look like
    PipelineObservation,    # What the board looks like
    env_name="devops_pipeline_env",
    max_concurrent_envs=1,
)
```

This one call automatically creates all core endpoints:

| Endpoint | Method | What it does |
|----------|--------|-------------|
| `/reset` | POST | Start a new game |
| `/step` | POST | Make a move |
| `/state` | GET | Check current state |
| `/schema` | GET | Get action/observation schemas |
| `/ws` | WebSocket | Persistent low-latency connection |
| `/web` | GET | Gradio UI for browser testing |
| `/health` | GET | Health check (auto from framework) |

#### Custom Endpoints

**GET /tasks** — Returns the 3 tasks with difficulty, description, max steps, and the action schema.

**GET /health** — Simple health check returning `{"status": "ok"}`. The hackathon validators call this.

**POST /baseline** — Returns pre-recorded scores from a real inference run. Hackathon rules say: "Do NOT run inference.py inside the container." So we just return saved numbers.

**POST /grader** — Scores the current game. Uses `PipelineEnvironment._last_instance` to access the most recent environment, then calls the grader on it.

**main()** — Entry point for running the server directly. The `if __name__ == "__main__": main()` pattern is required by `openenv validate`.

**Connections:**
- Uses pipeline_environment.py (PipelineEnvironment class)
- Uses models.py (PipelineAction, PipelineObservation)
- Uses graders.py (for /grader endpoint)

**If removed:** No web server. No one can connect. The environment is trapped on your computer.

---

### 9. client.py — The Remote Control

**What is it?** A Python class that talks to the server over the network. Instead of calling `env.reset()` directly, you call it through the client, which sends an HTTP/WebSocket request to the server.

**Why does it exist?** The AI player (inference.py) and the environment (server) run in different processes, possibly on different machines. The client bridges them.

#### Class: DevopsPipelineEnv

Extends `EnvClient` from OpenEnv. Three methods:

**_step_payload(action)** — Converts a PipelineAction to JSON for network transmission. `exclude_none=True` means empty fields aren't sent (keeps messages small).

**_parse_result(payload)** — Converts the JSON response from the server back into a StepResult containing the observation, reward, and done flag.

**_parse_state(payload)** — Converts the state response into a State object with episode_id and step_count.

**Connections:**
- inference.py uses this class to connect to the server
- __init__.py exports it as the main client class

**If removed:** inference.py can't connect to the server. The AI has no way to play.

---

### 10. __init__.py — The Export List

**What is it?** Makes the folder a Python package and defines what's available when you `import devops_pipeline_env`.

**Why does it exist?** Without it, Python doesn't recognize the folder as a package. All imports would fail.

**What it exports:**

| Name | From | What it is |
|------|------|-----------|
| `PipelineAction` | models.py | What the AI sends |
| `PipelineObservation` | models.py | What the AI sees |
| `ConfigEdit` | models.py | Config change structure |
| `DevopsPipelineEnv` | client.py | Client for connecting |

`__all__` is like a restaurant menu — "these are the things we offer."

**If removed:** `from devops_pipeline_env import ...` fails everywhere. inference.py breaks.

---

### 11. inference.py — The AI Player

**What is it?** The script where the AI actually plays the game. It connects to the environment, reads observations, asks an LLM what to do, sends actions, and logs everything.

**Why does it exist?** Hackathon requirement. Must demonstrate that an AI can play through all 3 tasks.

#### Environment Variables (lines 14-20)

```python
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("HF_TOKEN or API_KEY environment variable is required")
```
HuggingFace token for authenticating with the LLM API. If missing, crashes immediately (hackathon requirement).

```python
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
```
These have defaults — you CAN override them but don't have to.

#### Log Functions (lines 32-52)

Produce EXACTLY the format the hackathon requires:
```
[START] task=clean_deploy env=devops_pipeline model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action={"action_type":"deploy",...} reward=0.05 done=false error=null
[END] success=true steps=7 score=0.585 rewards=0.05,0.05,...
```
`flush=True` ensures output appears immediately, not buffered.

#### System Prompt (line 56)

Instructions for the LLM: "You are a DevOps engineer. Read the JSON observation. Respond with a single JSON action. No explanations." Like giving a new employee their job description.

#### build_prompt (line 79)

Combines the current observation (as JSON) with the last 6 history entries so the LLM remembers what it already did.

#### parse_llm_action (line 91)

Tries to parse the LLM's response as JSON. LLMs sometimes wrap JSON in markdown code blocks, so it strips those. If parsing fails, falls back to the safe `view_pipeline` action (just looking, can't break anything).

#### run_task (line 105) — The Game Loop

For one task:
1. Set `DEVOPS_TASK` environment variable
2. Call `env.reset()` — start new game
3. Loop for up to max_steps:
   a. Build prompt from observation + history
   b. Ask LLM what to do (API call to HuggingFace)
   c. Parse LLM response into a PipelineAction
   d. Send action to environment via `env.step(action)`
   e. Log the step
   f. Add to conversation history
   g. If done, break
4. Calculate final score: `sum(rewards) / max_reward`, clamped to [0, 1]
5. **finally block:** ALWAYS logs `[END]`, even if the script crashes

#### main (line 175)

1. Creates an OpenAI-compatible client pointing at HuggingFace's router
2. Connects to the environment (via Docker image or direct URL)
3. Runs all 3 tasks in sequence
4. Closes the connection in a finally block

**Connections:**
- Uses client.py to connect to the server
- Uses models.py for PipelineAction and ActionType
- Requires the server (app.py) to be running

**If removed:** No AI player. Can't demonstrate that the environment works. Hackathon DQ.

---

### 12. Dockerfile — The Shipping Container

**What is it?** A recipe for building a Docker container — a self-contained virtual computer with everything needed to run the project.

**Why does it exist?** Ensures the project runs identically everywhere. No "works on my machine" problems.

**Line by line:**

| Line | Code | Purpose |
|------|------|---------|
| 1 | `FROM python:3.11-slim` | Start with minimal Linux + Python 3.11 |
| 3 | `WORKDIR /app` | All commands run in /app directory |
| 5 | `ENV ENABLE_WEB_INTERFACE=true` | Enable Gradio web UI at /web |
| 8 | `COPY requirements.txt .` | Copy dependency list |
| 9 | `RUN pip install ... -r requirements.txt` | Install all packages |
| 12 | `COPY . .` | Copy all project files |
| 15 | `RUN pip install ... -e .` | Install our package (makes imports work) |
| 17 | `EXPOSE 8000` | Document which port we use |
| 19 | `CMD ["uvicorn", ...]` | Start the web server when container runs |

**Connections:** Must match openenv.yaml (port 8000, server.app:app). Uses requirements.txt and pyproject.toml.

**If removed:** Can't build a Docker image. Can't deploy to HuggingFace Spaces.

---

### 13. pyproject.toml — The Package Config

**What is it?** The birth certificate of the Python package. Tells Python's packaging tools: what's the package called, what does it need, how to build it.

**Why does it exist?** The `pip install -e .` command in the Dockerfile reads this file to install our package. Without it, `from devops_pipeline_env.models import ...` wouldn't work inside the container.

**Key sections:**

```toml
[build-system]
requires = ["setuptools>=45", "wheel"]
```
"Use setuptools to build this package."

```toml
[project]
name = "devops-pipeline-env"
dependencies = [
    "openenv-core[core]>=0.2.2",
    "pydantic>=2.0",
    "fastapi>=0.104.0",
    "uvicorn>=0.24.0",
    "openai>=1.0.0",
]
```
Package name and what it needs to work.

```toml
[tool.setuptools]
package-dir = { "devops_pipeline_env" = ".", "devops_pipeline_env.server" = "server" }
```
Maps folder structure to Python package names. The current folder becomes `devops_pipeline_env`, and `server/` becomes `devops_pipeline_env.server`. This is why `from devops_pipeline_env.models import ...` works even though models.py is in the root folder.

**If removed:** `pip install -e .` fails. Imports break inside Docker. Nothing runs.

---

### 14. requirements.txt — The Dependency List

**What is it?** A simple list of Python packages the project needs.

**Why does it exist?** The Dockerfile runs `pip install -r requirements.txt` before installing our package. It's faster than resolving from pyproject.toml.

| Package | What it does |
|---------|-------------|
| `openenv-core>=0.2.1` | The hackathon framework (Environment, Action, Observation, etc.) |
| `pydantic>=2.0` | Data validation (ensures actions/observations have correct types) |
| `fastapi>=0.104.0` | Web framework (turns Python functions into HTTP endpoints) |
| `uvicorn>=0.24.0` | ASGI server (actually runs FastAPI and handles network connections) |
| `openai>=1.0.0` | Client library for calling LLMs (works with HuggingFace's router despite the name) |

**If removed:** Docker build fails at the `pip install` step. No packages = nothing works.

---

## The Game Loop

The entire system is one loop that repeats:

```
1. AI reads the observation (the game board)
           |
           v
2. AI decides an action (e.g., "deploy api-gateway v2.3.1")
           |
           v
3. Action goes through client -> server -> environment -> engine
           |
           v
4. Engine updates the simulation (service health changes, deploy progresses)
           |
           v
5. Rewards.py calculates per-step reward (+0.15 for production deploy)
           |
           v
6. Environment builds new observation (updated board state)
           |
           v
7. New observation sent back to the AI
           |
           v
8. Repeat from step 1... until:
   - AI says "approve" or "abort"
   - Max steps reached
   - System health drops below 20%
           |
           v
9. Graders.py gives final score (0.0 to 1.0)
```

Each complete run through steps 1-9 for one task is called an **episode**. The inference script runs 3 episodes (one per task), logs everything, and the hackathon judges evaluate the scores.
