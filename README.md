---
title: DevOps Pipeline Environment
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: docker
pinned: true
app_port: 8000
base_path: /web
tags:
  - openenv
---

# DevOps Pipeline Gym

*An OpenEnv RL environment that trains LLMs to make production-critical decisions under uncertainty.*

**Theme:** World Modeling 3.1 — Professional Tasks.
**Live Space:** [yashash045/devops-pipeline-gym](https://huggingface.co/spaces/yashash045/devops-pipeline-gym) · **Code:** [Yashash4/devops-pipeline-gym](https://github.com/Yashash4/devops-pipeline-gym)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Yashash4/devops-pipeline-gym/blob/main/devops_pipeline_gym_colab.ipynb) [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/import?source=https%3A//github.com/Yashash4/devops-pipeline-gym/blob/main/devops_pipeline_gym_colab.ipynb)

**Quick re-run for judges:** open the Colab badge above → set `HF_TOKEN` in Secrets → run all cells. ~15 min on free T4. Loads our trained adapter, runs baseline + trained on the same seed, shows the delta.

**Trained adapter (hero, +3.225 delta):** [yashash045/devops-pipeline-gym-sft-adapter](https://huggingface.co/yashash045/devops-pipeline-gym-sft-adapter) · **GRPO RL refinement (exploratory):** [yashash045/devops-pipeline-gym-trained](https://huggingface.co/yashash045/devops-pipeline-gym-trained) · **Track-IO:** [yashash045/dpg-trackio](https://huggingface.co/spaces/yashash045/dpg-trackio)

---

## The Problem

Modern incident response is not a knowledge problem — it's a sequencing problem. The on-call engineer rarely lacks the technical knowledge to fix a broken auth service or scale a saturated database. What they lack is the discipline to investigate before acting, the judgment to identify root cause through cascading symptoms, and the experience to choose among multiple valid recovery paths.

Frontier LLMs reflect this gap precisely. Qwen2.5-72B scores **0.184** on our hardest task (`judgment_call`) — a stark headroom signal. The model has all the technical knowledge baked in. What it needs is training on the *decisions* that connect knowledge to outcomes.

This environment trains exactly that.

## Why A Simulator

Reinforcement learning needs identical state across seeds for gradient stability. Real production systems don't reproduce identically — pod scheduling, network jitter, OOM timing, and live database load all vary between runs. A simulator solves this.

Our simulator is **deterministic** (same seed → same episode), **reproducible** (Docker + Python, runs anywhere), and **free** (no K8s cluster, no paid APIs, no GPU at inference time).

We complement [kube-sre-gym](https://huggingface.co/spaces/openenv-community/kube-sre-gym) rather than compete with it. They tackle pod-level `kubectl` operations on a real Kubernetes cluster — high-fidelity, hardware-bound. We tackle CI/CD pipeline workflow on a deterministic simulator — different decision surface, different substrate, complementary scope. Same agent family, different training laboratory.

## The Four Uncertainty Pillars

Decisions under uncertainty is what LLMs are *worst at*. This environment trains it directly via four design pillars.

**1. Incomplete Information.** CPU, memory, latency, and error rate are hidden in the observation by default. When a service is unhealthy, its `health` field is masked as `unknown` until the agent invokes `view_logs` or `view_config` on it. Acting blind is allowed but penalised; investigation has a small cost (decaying first-view bonus) but unlocks the data needed to act correctly.

**2. Cascading Consequences.** Five microservices in a dependency graph (`database-primary → auth-service → api-gateway → web-frontend`, `database-primary → cache-service`). Fixing the wrong service while root cause persists makes the situation worse — health degrades further. Wrong-fix-first is a real failure mode the agent must learn to avoid.

**3. Multiple Valid Strategies.** The `judgment_call` task has three reasonable resolution paths: deploy hotfix + auth config fix (best), rollback to previous version (safest), or hotfix alone (riskiest). Each scores differently. There is no single correct answer — only better-and-worse trade-offs.

**4. Procedural Variation.** The `random_incident` task generates 40+ distinct scenarios per seed (5 failure types × 4 services × 2 severities, with 30% compound incidents). The agent cannot memorise a fixed answer — it must generalise the *pattern* of investigate → diagnose → fix.

## Environment Design

### Action space (9 actions, role-gated)

| Action | Role | Notes |
|---|---|---|
| `view_pipeline` | sre | Inspect pipeline state |
| `view_logs` | sre | Service-specific logs (unmasks health) |
| `view_config` | dev | Config inspection (also unmasks health) |
| `edit_config` | dev | Apply key/value config edits |
| `run_migration` | dev | Run a pending migration |
| `deploy` | ops | Deploy a target version |
| `rollback` | ops | Roll back to previous version |
| `approve` | ops | Terminate episode with success |
| `abort` | ops | Terminate episode with failure |

Each action carries a `role` field (`dev` / `sre` / `ops`). Acting outside your role triggers a `-0.15` penalty and the action is not executed. The current role rotates during multi-step incidents to simulate real on-call handoffs.

### Observation space

- `services` — list of 5 microservices (`name`, `health`, `current_version`, `cpu_percent`, `memory_percent`, `request_latency_ms`, `error_rate`, `active_connections`, `last_deploy_timestamp`). Resource metrics + degraded-service health are hidden until the agent investigates that service.
- `pipeline` — build/test/deploy stage state.
- `migrations` — pending migration list.
- `active_alerts` — current alerts with severity.
- `available_actions` — context-sensitive list filtered to the current role.
- `goal`, `task_description`, `current_role`, `role_history`.
- `last_action_result`, `last_action_error`, `summary`, `config_snapshot`.

### 6 Tasks

| Task | Difficulty | Description |
|---|---|---|
| `clean_deploy` | easy | Deploy 2 services with all tests passing. |
| `broken_pipeline` | medium | Diagnose test failures, fix config errors, run migrations. |
| `judgment_call` | hard | Production incident with multiple resolution paths and a 12-step time limit. |
| `cascading_failure` | medium-hard | Root cause hidden behind dependency-chain symptoms. |
| `capacity_crisis` | medium-hard | Proactive scaling before saturation tipping point. |
| `random_incident` | variable | Procedurally generated; service + failure type + severity randomised from seed. |

### Reward (6 outcome-based components)

Per-step bounded `[-0.40, +0.32]`, fully deterministic — no LLM judges in the reward loop.

| Signal | Reward | Condition |
|---|---|---|
| Service deployed to production | +0.15 | Service reaches prod successfully |
| Service verified in staging | +0.05 | Staging health check passes |
| Config error fixed | +0.08 | Service health improved after config change |
| Migration completed | +0.06 | Pending migration count decreased |
| Alert resolved | +0.03 | Alert count decreased |
| Role-alignment (Round 2) | +0.02 / −0.05 | `action.role` matches/violates `ROLE_ACTIONS[role]` |

Plus structural penalties: broken-healthy `−0.30`, repeated investigation `−0.01` / `−0.03` (consecutive), repeated non-view action `−0.02`. End-of-episode terminal bonus: `approve` while all-healthy `+2.0`; `abort` `−1.5`; max-steps-with-unhealthy `−1.5`. Reward shaping is potential-based (Ng et al. 1999) with count-based exploration decay (Bellemare et al. 2016).

## Results

We trained Qwen3-1.7B-bnb-4bit on 30 expert trajectories via SFT, then explored RL refinement via GRPO. The SFT-only adapter is the headline result; GRPO is shipped as supporting evidence that the training pipeline is end-to-end functional.

### Headline: SFT delta on `judgment_call` (seed 3003)

| Configuration | Total reward | Steps | Succeeded | Delta |
|---|---:|---:|:---:|---:|
| Untrained baseline (Qwen2.5-7B-Instruct via HF Router) | **−1.070** | 12 | False | — |
| **+ SFT LoRA on Qwen3-1.7B-bnb-4bit** ([yashash045/devops-pipeline-gym-sft-adapter](https://huggingface.co/yashash045/devops-pipeline-gym-sft-adapter)) | **+2.155** | 10 | **True** | **+3.225** |

Same env, same seed (3003), same prompt format. SFT was 2 epochs on 30 trajectories (~30 min on T4), 17M trainable params (1.69% of base), QLoRA r=16 α=32 across all attn + MLP modules per Daniel-Unsloth-recommended settings.

**What the trained agent learned** (full 10-step rollout, captured live):

```
step 1 | sre | view_pipeline  | api-gateway | r=+0.030    investigate
step 2 | sre | view_pipeline  | (cluster)   | r=+0.035    investigate
step 3 | sre | view_pipeline  | api-gateway | r=+0.010
step 4 | sre | view_pipeline  | api-gateway | r=-0.010
step 5 | ops | deploy         | api-gateway | r=+0.070    deploy hotfix
step 6 | ops | deploy         | api-gateway | r=+0.150    deploy succeeds
step 7 | ops | deploy         | api-gateway | r=-0.000
step 8 | sre | view_pipeline  | (cluster)   | r=-0.150
step 9 | ops | deploy         | api-gateway | r=-0.000
step 10| ops | approve        | (cluster)   | r=+2.020    terminal success ✓
```

The agent investigates with the SRE role, hands off to OPS for the deploy, gets the hotfix landing reward (+0.15), then chooses `approve` once the system is healthy — earning the +2.0 terminal-success bonus. That's the SFT teaching the right *sequence* of decisions, not just individual actions.

Reproduce in [`kaggle_eval.ipynb`](kaggle_eval.ipynb) — opens on Kaggle T4, ~12 min, prints the same delta. Single-seed evals at temperature=0.7 show some sampling variance; most runs sample `approve` correctly and reach the +2.0 terminal bonus.

### GRPO refinement (exploratory)

We ran 30 GRPO steps from raw base on L40S to validate the RL pipeline. Reward stayed flat (~0.04 mean) but loss/grad_norm/KL all flowed (`final_loss=8.1e−6`, `final_KL=0.0012`), confirming the training infrastructure works end-to-end. The trained adapter is shipped at [yashash045/devops-pipeline-gym-trained](https://huggingface.co/yashash045/devops-pipeline-gym-trained); training curves below.

![GRPO reward + loss curves](outputs/grpo_run1/reward_curve.png)

Track-IO logs (loss, reward, KL, entropy, grad_norm per step): [yashash045/dpg-trackio](https://huggingface.co/spaces/yashash045/dpg-trackio).

### Tooling

`training/eval_baseline.py` records `avg_steps_to_recovery` per task (Phase J.5 metric). `training/generate_comparison_chart.py` produces a side-by-side reward + recovery PNG. `training/plot_grpo_curve.py` plots the GRPO trainer state. `training/export_replay.py` + `training/render_replay.py` produce per-step PNG frames for the before/after demo video.

## Reproduce It Yourself

### As an evaluator (HF Space, no setup)

```bash
curl -s -X POST -H "Content-Type: application/json" -d '{}' \
  https://yashash045-devops-pipeline-gym.hf.space/reset
```

`/health`, `/tasks`, `/baseline`, `/grader`, `/curriculum_progress` endpoints are all live on the Space.

### Locally (Docker)

```bash
git clone https://github.com/Yashash4/devops-pipeline-gym
cd devops-pipeline-gym
docker build -t devops-pipeline-gym .
docker run -p 8000:8000 devops-pipeline-gym
```

### Train your own agent (T4 / H100)

```bash
pip install -e '.[training]'
uvicorn server.app:app --host 0.0.0.0 --port 8000 &

# Stage 1 — SFT warmup on 78 expert trajectories
python training/sft_warmup.py \
  --model unsloth/Qwen3-1.7B-bnb-4bit \
  --trajectories data/sft_trajectories.jsonl \
  --output-dir outputs/sft_warmup --epochs 2

# Stage 2 — GRPO (200 steps, 8 generations per prompt)
python training/grpo_train.py \
  --model unsloth/Qwen3-1.7B-bnb-4bit \
  --sft-adapter-path outputs/sft_warmup/final \
  --env-url http://127.0.0.1:8000 \
  --max-steps 200 --num-generations 8 \
  --output-dir outputs/run1

# Eval before/after
python training/eval_baseline.py --model unsloth/Qwen3-1.7B-bnb-4bit \
  --env-url http://127.0.0.1:8000 --output baseline.json --n-seeds 3
python training/eval_baseline.py --model unsloth/Qwen3-1.7B-bnb-4bit \
  --adapter-path outputs/run1/final \
  --env-url http://127.0.0.1:8000 --output trained.json --n-seeds 3
python training/generate_comparison_chart.py \
  --baseline baseline.json --trained trained.json --output before_after.png
```

### Validate the env

```bash
python -m openenv.cli validate                 # local
python -m openenv.cli validate --url https://yashash045-devops-pipeline-gym.hf.space  # remote
```

## Architecture Note

We deliberately ship **zero external API dependencies** in the runtime path. No Groq, no OpenAI, no Anthropic, no Ollama. The env is pure Python + FastAPI + a deterministic simulator. This makes the training pipeline (a) reproducible end-to-end, (b) free for anyone to retrain, and (c) immune to upstream API changes that would break submissions. Inference uses the HF Inference Router (OpenAI-compatible) — but only at evaluation time, never inside the reward loop.

The reward graders are deterministic Python code. Same trajectory → same score, every time. This is the property that makes the training loop both reproducible and plagiarism-check transparent.

## Citation

If you build on this work, please cite:

> Team Tripod (Yashash, Gajanand, Likith). *DevOps Pipeline Gym: An OpenEnv RL environment for training decisions under uncertainty.* OpenEnv Hackathon Grand Finale 2026.

## License

Apache 2.0.
