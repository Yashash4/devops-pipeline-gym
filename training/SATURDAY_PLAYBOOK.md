# Saturday H100 Training Playbook — Round 2

**Event:** Meta × HF × PyTorch OpenEnv Hackathon Grand Finale, Bangalore
**When:** Saturday April 25, 2026
**Goal:** Train Qwen 1.5B with GRPO to beat the Qwen 72B baseline on ≥1 task

Prerequisite: Friday Kaggle T4 dry-run **PASSED** (2026-04-24). The pipeline is
proven. Saturday is execution + tier escalation if reward is flat.

---

## 0. Pre-flight (10 min)

```bash
# Activate HF compute credits. Clone + install.
git clone https://github.com/Yashash4/devops-pipeline-gym ~/gym
cd ~/gym
pip install -r requirements.txt && pip install -e .

# Env server (run in a tmux pane — stays up the whole session)
python -m uvicorn server.app:app --host 127.0.0.1 --port 8000 &
sleep 6
curl -X POST -H "Content-Type: application/json" -d '{}' http://127.0.0.1:8000/reset
# MUST return 200 with clean_deploy observation. If not, STOP — fix env first.

# Training deps
pip install -r requirements-training.txt

# Sanity: grpo_train --help prints clean CLI (no heavy imports trigger)
python training/grpo_train.py --help
```

**Validated on Kaggle T4:** unsloth 2026.4.8, trl 0.29.x, torch 2.10.0+cu128, transformers 4.56.2, bitsandbytes, peft. H100 should have compatible wheels.

---

## 1. Baseline eval (30 min, BEFORE training)

Capture untrained Qwen 1.5B's score on all 6 tasks so we have a before-picture.

```bash
python training/eval_baseline.py \
    --model unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit \
    --env-url http://127.0.0.1:8000 \
    --output baseline_v2.json \
    --n-seeds 3
```

Expected wall-time: 6 tasks × 3 seeds × 20 steps × ~2s/step ≈ 12 min. Save `baseline_v2.json` to the artifact bucket.

Expected scores (reference — Round 1 Qwen 72B, we're running Qwen 1.5B so expect lower):

| Task | Round 1 Qwen 72B baseline |
|---|---|
| clean_deploy | 0.700 |
| broken_pipeline | 0.482 |
| judgment_call | 0.184 |
| cascading_failure | 0.280 |
| capacity_crisis | 0.250 |
| random_incident (seed 6006) | 0.350 |

Qwen 1.5B will likely score 0.1-0.3 across the board. That's the delta we're trying to close with GRPO.

---

## 2. Primary training run (Tier 1) — 2-3 hours on H100

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

Flag rationale:
- `--num-generations 8` — 4 was too low on T4 dry-run (`frac_reward_zero_std: 0.5`, half the groups got identical rewards so no learning signal). 8 gives enough group variance.
- `--max-completion-length 512` — dry-run hit the 256-ceiling every completion (`clipped_ratio: 1.0`). 512 on H100 fits the new schema-rich prompt (~1.2k tokens system + user → ~2k total with 512 completion, within Qwen 1.5B's 32k context).
- `--max-steps 150` — enough steps for a reward trend to emerge; 100 was the IMPLEMENTATION_PLAN minimum.
- `--prompts-per-task 6` — 36 total prompts per task across 6 tasks × 6 seeds = 216 dataset entries. Enough diversity without overfitting.
- `--use-vllm` is NOT in the primary run — enable on a second pass if the first converges and we want faster rollouts. vLLM + GRPO + Unsloth is brittle; only add when baseline works.
- `--learning-rate 5e-6` — Unsloth's recommended default for Qwen 1.5B LoRA. Don't tune blind.

**Monitor every 10 steps:**

```bash
# Reward trend (should climb)
tail -f runs/saturday_v1/trainer_state.json 2>/dev/null \
  | grep -oE '"loss": *-?[0-9.]+' | head -20

# Tokenizer clipping (should fall as model learns brevity)
grep 'clipped_ratio' runs/saturday_v1/*.log | tail -5

# Reward group variance (frac_reward_zero_std should drop below 0.3)
grep 'frac_reward_zero_std' runs/saturday_v1/*.log | tail -5
```

---

## 3. Tier 1 assessment (after step 30)

**Look at:** mean reward across the last 10 logging steps vs first 10.

| Outcome | Probability | Action |
|---|---|---|
| Mean climbing, span > 0.05 | ~40% | Let it run to 150 steps. Go to §5 (eval + export). |
| Noisy but trending up | ~35% | Let it run. If still flat after step 80, go to §4. |
| Dead flat (std < 0.02 across 30 steps) | ~20% | **Kill run, go to §4 Tier 2.** |
| Diverging (reward dropping) | ~5% | Kill, reduce LR to 3e-6, restart. |

---

## 4. Escalation tiers (if Tier 1 fails)

Each tier runs for at most 1 hour before moving up. We commit to one of Tiers 1-3; Tier 4 is fallback for submission.

### Tier 2 — tighten generation, lower variance (1 hour)

```bash
python training/grpo_train.py \
    --model unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit \
    --env-url http://127.0.0.1:8000 \
    --max-steps 100 \
    --num-generations 4 \
    --max-completion-length 384 \
    --prompts-per-task 6 \
    --learning-rate 3e-6 \
    --output-dir ./runs/saturday_v2
```

Changes: smaller groups (less variance per group but cheaper), shorter completions (more training steps per wall-hour), lower LR.

### Tier 3 — smaller model, single task (1 hour)

Switch to Qwen 0.5B to get a learning signal faster. Lock training to **judgment_call** (hardest task, biggest headroom for improvement — Round 1 baseline 0.184, optimal 0.935).

```bash
# Temporarily lock curriculum off by forcing the env to always pick judgment_call
DEVOPS_TASK=judgment_call python -m uvicorn server.app:app --host 127.0.0.1 --port 8000 &

python training/grpo_train.py \
    --model unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit \
    --env-url http://127.0.0.1:8000 \
    --max-steps 100 \
    --num-generations 4 \
    --max-completion-length 384 \
    --prompts-per-task 20 \
    --learning-rate 1e-5 \
    --output-dir ./runs/saturday_v3
```

Goal: "trained 0.184 → ≥0.30 on judgment_call." One task, one story.

### Tier 4 — submit what we have

If nothing above converged by Saturday 6 PM, submit Tier 1's log/curve AS-IS. Reframe the pitch: *"Here's what we learned about the reward landscape and what would need to change for convergence."* Honesty beats overclaiming. Reward curve going sideways still beats no curve — per BATTLEPLAN §10.

---

## 5. Trained eval (after any successful training run)

```bash
python training/eval_baseline.py \
    --model ./runs/saturday_v1/final \
    --env-url http://127.0.0.1:8000 \
    --output trained_v2.json \
    --n-seeds 3
```

Compare:

```bash
python training/generate_comparison_chart.py \
    --baseline baseline_v2.json \
    --trained trained_v2.json \
    --output before_after.png \
    --title "Qwen 1.5B — untrained vs GRPO-trained (150 steps)"
```

---

## 6. Export adapter to HF Hub (before Sunday pitch)

```bash
# Adapter only — do NOT merge + upcast per Participant Help Guide.
huggingface-cli login   # already logged in from prior phases
cd runs/saturday_v1/final
hf_hub_upload <username>/devops-agent-trained .
# OR via Python:
python -c "
from huggingface_hub import upload_folder
upload_folder(
    repo_id='yashash045/devops-agent-trained',
    folder_path='runs/saturday_v1/final',
    commit_message='GRPO-trained Qwen 1.5B, 150 steps, Saturday 2026-04-25',
)
"
```

---

## 7. Artifacts to save for pitch

Copy these to `round2/REF/artifacts/` (gitignored, keep locally):

- [ ] `baseline_v2.json`
- [ ] `trained_v2.json`
- [ ] `runs/saturday_v1/reward_curve.png`
- [ ] `before_after.png`
- [ ] One raw episode log per task for the demo (untrained failure, trained success)
- [ ] Screenshots of the HF Hub adapter page

---

## 8. Known pitfalls from Friday's dry-run

1. **Role uppercase** — FIXED in `parse_completion`. Should not recur.
2. **Config_edits wrong shape** — Prompt now has explicit `{"key": "...", "value": "..."}` example. If training output still shows dict-of-sections shape, the model is ignoring the schema — try lowering temperature in the GRPO rollout config.
3. **All completions hit 256 ceiling** — bumped to 512 for Saturday.
4. **`frac_reward_zero_std: 0.5`** — half the generation groups had zero reward variance, killing the learning signal. Fix: `--num-generations 8` (was 2 on T4).

---

## 9. Wall-time budget (Saturday)

| Block | Time |
|---|---|
| Pre-flight + env up | 0:10 |
| Baseline eval | 0:30 |
| Tier 1 training | 2:30 |
| Assessment + tier branching buffer | 0:30 |
| Trained eval | 0:30 |
| Before/after chart + logs | 0:20 |
| Export adapter to HF | 0:20 |
| **Subtotal** | **4:50** |
| Tier 2/3 escalation budget | +2:00 |
| Pitch rehearsal | 0:30 |
| **Total** | **~7:20** |

Start by 9:00 AM to finish by 4:20 PM. H100 credits last — don't idle the instance.

---

## 10. Emergency rollback

If anything breaks the env server, revert to the Round 1 snapshot:

```bash
git checkout round2-start   # pre-Round-2 commit
# re-start uvicorn, old behaviour
```

Round 2 code is additive, so rolling back to `round2-start` gives a working Round 1 env. Worst-case submission is Round 1 + pitch framed around "we built the Round 2 infra but training didn't converge in time." Still a finalist submission.
