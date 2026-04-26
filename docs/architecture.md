# DevOps Pipeline Gym, Architecture

This is the quick tour. An LLM agent sends actions. The **role gate** decides which actions count for the current role. A graph of 5 microservices reacts. Six deterministic reward terms add up to a bounded per-step reward in `[-0.40, +0.32]`.

---

## ASCII (terminal-safe fallback)

```text
+------------------------------------------------------------------------+
|                         Agent (LLM policy)                             |
|  Qwen3-1.7B (SFT + GRPO)  /  Qwen2.5-72B (baseline via HF Router)      |
+----------------------------------+-------------------------------------+
                                   | PipelineAction(role, action_type,
                                   |   service_name, target_version,
                                   |   config_edits, migration_name, ...)
                                   v
                       HTTP POST /step  (FastAPI, OpenEnv server)
                                   |
                                   v
                +--------------------------------------+
                |  Role Router  (state-driven gate)    |
                |  DEV: view/edit_config, run_migration|
                |  SRE: view_logs, view_pipeline       |
                |  OPS: deploy, rollback, approve, abort|
                |  mismatch -> -0.15 (no-op)           |
                |  bad-role-action -> -0.10 (no-op)    |
                +------------------+-------------------+
                                   v
                +--------------------------------------+
                |  PipelineEnvironment  (engine)       |
                |   * 6 tasks + CurriculumController   |
                |   * 5 microservices (dep graph):     |
                |       database-primary --> auth      |
                |       auth --> api-gateway           |
                |       api-gateway --> web-frontend   |
                |       database-primary --> cache     |
                |   * deterministic _rng (seeded)      |
                +------------------+-------------------+
                                   v
                +--------------------------------------+
                |  Reward fn (6 components, bounded)   |
                |  health-delta | deploy-progress |    |
                |  broke-healthy | sub-goals |         |
                |  investigation | role-alignment      |
                |  -> bound_step_reward [-0.40, +0.32] |
                +------------------+-------------------+
                                   v
              PipelineObservation(services, alerts, current_role,
                role_history, pipeline, summary, reward, done)
                                   |
                                   v
                          Agent (next step)
```

---

## Mermaid (rendered in README / GitHub)

```mermaid
flowchart TD
    A["Agent (LLM policy)<br/>Qwen3-1.7B SFT+GRPO<br/>Qwen2.5-72B baseline"]:::agent

    A -- "PipelineAction<br/>role, action_type,<br/>service_name, ..." --> H["HTTP POST /step<br/>FastAPI / OpenEnv server"]:::http

    H --> R{"Role Router<br/>state-driven gate"}:::router
    R -- DEV --> RDEV["DEV<br/>view_config<br/>edit_config<br/>run_migration"]:::dev
    R -- SRE --> RSRE["SRE<br/>view_logs<br/>view_pipeline"]:::sre
    R -- OPS --> ROPS["OPS<br/>deploy / rollback<br/>approve / abort"]:::ops
    R -. "role mismatch -0.15<br/>bad action -0.10" .-> PEN["No-op penalty path"]:::penalty

    RDEV --> ENV
    RSRE --> ENV
    ROPS --> ENV
    PEN --> OBS

    ENV["PipelineEnvironment<br/>6 tasks + CurriculumController<br/>seeded _rng (deterministic)"]:::env --> DEP

    DEP["5-microservice dependency graph<br/>database-primary &rarr; auth &rarr; api-gateway &rarr; web-frontend<br/>database-primary &rarr; cache"]:::deps --> REW

    REW["Reward fn (6 deterministic components)<br/>health-delta | deploy-progress | broke-healthy<br/>sub-goals | investigation | role-alignment<br/>bound_step_reward [-0.40, +0.32]"]:::reward --> OBS

    OBS["PipelineObservation<br/>services, alerts, current_role,<br/>role_history, pipeline, summary,<br/>reward, done"]:::obs --> A

    classDef agent  fill:#1f2937,stroke:#9ca3af,color:#f9fafb
    classDef http   fill:#374151,stroke:#9ca3af,color:#f9fafb
    classDef router fill:#4b5563,stroke:#d1d5db,color:#f9fafb
    classDef dev    fill:#1d4ed8,stroke:#1e3a8a,color:#ffffff
    classDef sre    fill:#15803d,stroke:#14532d,color:#ffffff
    classDef ops    fill:#c2410c,stroke:#7c2d12,color:#ffffff
    classDef penalty fill:#7f1d1d,stroke:#450a0a,color:#fecaca
    classDef env    fill:#0f172a,stroke:#475569,color:#f8fafc
    classDef deps   fill:#312e81,stroke:#1e1b4b,color:#e0e7ff
    classDef reward fill:#7c2d12,stroke:#431407,color:#fed7aa
    classDef obs    fill:#374151,stroke:#9ca3af,color:#f9fafb
```

---

## Key invariants the diagram encodes

- **Single policy, role-conditioned**: one model. The role lives in the observation
  (`current_role`) and on the action (`PipelineAction.role`). There are no
  separate policies per role.
- **Role gate is hard**: a mismatched `action.role` short-circuits with `-0.15`
  and the action does **not** execute. A wrong action for the current role costs
  `-0.10`.
- **Deterministic env**: scenarios load with `chosen_seed`. `random_incident`
  honours `DEVOPS_SEED`. There is no `random.random()` and no `hash()` anywhere.
- **No LLM in env runtime**: graders, curriculum, and scenarios are pure Python.
- **Reward = 5 outcome components + role_alignment**, then bounded to
  `[-0.40, +0.32]`. The terminal bonus fires once per episode (on approve, abort,
  or timeout) and sits on top of the per-step bound.
- **Curriculum picks the task** in `reset()` unless an explicit task is requested
  (via kwargs or the `DEVOPS_TASK` env var).

---

## How to embed in the README

Inline mermaid for GitHub:
~~~markdown
## Architecture

```mermaid
flowchart TD
    ... (paste from docs/architecture.md)
```
~~~

Rendered PNG for HF Space READMEs:
```markdown
## Architecture
![Architecture](docs/architecture.png)
```
Generate the PNG with:
```bash
python scripts/render_diagram.py
```

ASCII fallback:
````markdown
## Architecture
```text
... (paste ASCII block from docs/architecture.md)
```
````
