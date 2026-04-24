# training/ — DevOps Pipeline Gym (Round 2)

GPU-only. Not installed on the HF Space. Run on Kaggle T4 or H100.

## Files

- `grpo_train.py` — Unsloth + TRL GRPO fine-tuning against the live env
- `eval_baseline.py` — Run any model (HF hub id or LoRA adapter) on all 6 tasks × N seeds → JSON
- `generate_comparison_chart.py` — Side-by-side bar chart from two eval JSONs

## Install (GPU host only)

```bash
pip install -e '.[training]'
# or equivalently
pip install -r requirements-training.txt
```

## Friday dry-run (Kaggle T4)

```bash
# Terminal 1 — env server
cd /kaggle/working/devops-pipeline-gym
pip install -r requirements.txt
pip install -e .
python -m uvicorn server.app:app --host 127.0.0.1 --port 8000 &
sleep 5
curl -X POST -H "Content-Type: application/json" -d '{}' http://127.0.0.1:8000/reset
# Expect: 200 OK

# Terminal 2 — training
pip install -r requirements-training.txt
python training/grpo_train.py \
    --model unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit \
    --env-url http://127.0.0.1:8000 \
    --max-steps 5 \
    --num-generations 2 \
    --prompts-per-task 2 \
    --output-dir ./dryrun
```

Success = `dryrun/final/` contains an adapter, `dryrun/reward_curve.png` exists, rewards are non-zero.

## Saturday H100 real training

See **[SATURDAY_PLAYBOOK.md](SATURDAY_PLAYBOOK.md)** for the full runbook with
tier-escalation plan, timings, and recovery procedures.

Quick reference (primary Tier 1 command):

```bash
python training/grpo_train.py \
    --model unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit \
    --env-url http://127.0.0.1:8000 \
    --max-steps 150 \
    --num-generations 8 \
    --max-completion-length 512 \
    --prompts-per-task 6 \
    --learning-rate 5e-6 \
    --output-dir ./runs/saturday_v1
```

Notes:
- `--num-generations 8` minimum (T4 dry-run with 2 got `frac_reward_zero_std=0.5`).
- `--max-completion-length 512` on H100 (256 was the T4 ceiling — every completion clipped).
- `--use-vllm` is NOT in the primary run. Add only after Tier 1 converges and you want faster rollouts; it's brittle with GRPO+Unsloth.

Escalation tiers if reward is flat — see `round2/REF/PHASE_EXECUTION.md` Phase 10 decision tree.

## Before/after eval

```bash
# Baseline (untrained)
python training/eval_baseline.py \
    --model unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit \
    --env-url http://127.0.0.1:8000 \
    --output baseline_v2.json \
    --n-seeds 3

# Trained adapter (from Saturday)
python training/eval_baseline.py \
    --model ./outputs/run1/final \
    --env-url http://127.0.0.1:8000 \
    --output trained_v2.json \
    --n-seeds 3

# Chart
python training/generate_comparison_chart.py \
    --baseline baseline_v2.json \
    --trained trained_v2.json \
    --output before_after.png
```

## Constraints

- `max-completion-length` ≤ 512 on T4 (OOM otherwise); 768-1024 safe on H100
- `num-generations` × `max-completion-length` × `batch-size` is the memory-dominant product
- Adapter-only save (`model.save_pretrained`) — do NOT merge+upcast LoRA per Help Guide
- Env server must be reachable at `--env-url`; training hits it via HTTP for every reward evaluation
