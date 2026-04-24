# SFT Trajectories — Format & Authoring Guide (Phase 6.5)

We need **30 expert trajectories** by Saturday morning to warm up GRPO training.
5 seeds are already in [`sft_trajectories.jsonl`](sft_trajectories.jsonl).
**25 more** are needed — please add yours directly to the same file.

## Format

One JSON object per line (JSONL). Each line is an independent episode.

```json
{"messages": [
  {"role": "system",    "content": "<SYSTEM_PROMPT verbatim>"},
  {"role": "user",      "content": "<observation text>"},
  {"role": "assistant", "content": "<action JSON as a string>"},
  {"role": "user",      "content": "<next observation text>"},
  {"role": "assistant", "content": "<next action JSON>"},
  …
]}
```

- Comment lines (start with `#`) and blank lines are ignored by the loader.
- Your JSON must be valid on a single line (no internal newlines; use `\n` in the `content` string if needed).
- The `content` of each assistant turn is a JSON **string** — not a JSON object. The string itself must parse to a valid `PipelineAction`.

## PipelineAction schema (mandatory — lowercase values!)

| Field | Values / Type | Notes |
|---|---|---|
| `action_type` | `view_pipeline`, `view_logs`, `view_config`, `edit_config`, `run_migration`, `deploy`, `rollback`, `approve`, `abort` | lowercase |
| `role` | `sre`, `dev`, `ops` | lowercase |
| `service_name` | `api-gateway`, `auth-service`, `cache-service`, `database-primary`, `web-frontend` | required for view_logs / view_config / edit_config / deploy / rollback |
| `target_version` | string (e.g. `v2.3.1`) | required for `deploy` |
| `config_edits` | **list** of `{"key": "...", "value": "..."}` objects | required for `edit_config`. Keys use dot-notation. NOT a dict-of-sections. |
| `migration_name` | string | required for `run_migration` |
| `reason` | string | optional; strongly recommended for approve/abort |
| `handoff_notes` | string | optional; include when transitioning between roles |

### Role → allowed actions

- **SRE**: `view_logs`, `view_pipeline`
- **DEV**: `view_config`, `edit_config`, `run_migration`
- **OPS**: `deploy`, `rollback`, `approve`, `abort`

Every action's `role` MUST match a role permitted for that `action_type`.

## System prompt

Copy the exact `SYSTEM` string from the first seed trajectory (first line of `sft_trajectories.jsonl` after the comments). Keep it byte-identical across all your trajectories so the dataset is consistent.

## Task coverage plan — 25 more trajectories needed

Seed file has 1 per task. Split the remaining 25:

| Task | Seeded | Gajanand adds | Likith adds | Total target |
|---|---|---|---|---|
| clean_deploy | 1 | — | 4 | 5 |
| broken_pipeline | 1 | — | 4 | 5 |
| judgment_call | 1 | 4 | — | 5 |
| cascading_failure | 1 | 4 | — | 5 |
| capacity_crisis | 1 | 4 | — | 5 |
| random_incident | — | — | 5 | 5 |
| **TOTAL** | **5** | **12** | **13** | **30** |

Aim for variety: different root causes, different role orderings, different branching decisions. A trajectory that always goes SRE → DEV → OPS in the same way isn't teaching the model any more than the seed already does.

## 2 complete mini-examples

### Example A — clean_deploy, SRE-only monitoring then OPS approve (3 actions)

```json
{"messages": [
  {"role": "system", "content": "<SYSTEM>"},
  {"role": "user", "content": "ROLE: SRE. TASK: Monitor pipeline. All services healthy. Available actions: view_pipeline, view_logs."},
  {"role": "assistant", "content": "{\"action_type\": \"view_pipeline\", \"role\": \"sre\", \"handoff_notes\": \"pipeline stable, ready for final approval\"}"},
  {"role": "user", "content": "ROLE: OPS. Pipeline green. Available actions: deploy, rollback, approve, abort."},
  {"role": "assistant", "content": "{\"action_type\": \"approve\", \"reason\": \"all services stable\", \"role\": \"ops\"}"}
]}
```

### Example B — edit_config with the right shape

```json
"{\"action_type\": \"edit_config\", \"service_name\": \"cache-service\", \"config_edits\": [{\"key\": \"redis.host\", \"value\": \"redis-prod.internal:6379\"}, {\"key\": \"redis.pool_size\", \"value\": \"20\"}], \"role\": \"dev\"}"
```

Note: `config_edits` is a **list** of `{"key", "value"}` objects. Multiple edits share the same list. Never use `{"environment": ..., "packages": ...}` style — the env rejects it.

## Common pitfalls

1. **Uppercase role or action_type** — `"SRE"`, `"DEPLOY"`. The parser lowercases these now, but write them lowercase from the start so your trajectories match what the trained model should emit.
2. **`config_edits` as a dict instead of a list** — always `[{"key": "...", "value": "..."}]`, never `{"redis": {"host": "..."}}`.
3. **Newlines inside a JSONL line** — every trajectory must fit on one line. Escape literal newlines as `\n` inside the `content` string.
4. **Assistant content as an object** — must be a JSON **string**, not a dict. The training loop parses it as a string and then parses the string as JSON.
5. **Role / action mismatch** — e.g. SRE trying to `deploy`. The env returns -0.10. Match role to action per the table above.
6. **Missing `service_name` on view_logs / edit_config / deploy** — required. The env returns an error.

## Workflow

```bash
git fetch origin
git checkout -b phase6.5-sft-trajectories origin/phase6.5-training-fix
# edit data/sft_trajectories.jsonl — ADD your new lines at the end
python -c "import json; [json.loads(l) for l in open('data/sft_trajectories.jsonl') if l.strip() and not l.startswith('#')]; print('OK')"
git add data/sft_trajectories.jsonl
git commit -m "SFT trajectories: add <N> for <tasks>"
git push origin phase6.5-sft-trajectories
# Open PR into phase6.5-training-fix
```

If the `python -c ...` validation prints anything other than `OK`, your trajectories have invalid JSON. Fix the line it chokes on before committing.

## Sanity check before PR

Run the loader manually:

```bash
python -c "from training.sft_warmup import load_trajectories; ds = load_trajectories('data/sft_trajectories.jsonl'); print(f'{len(ds)} trajectories loaded')"
```

Should print `30 trajectories loaded` (or whatever count is in the file) without warnings.
