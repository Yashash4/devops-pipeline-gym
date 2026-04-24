# Saturday H100 Training Playbook — Round 2 (Phase 6.5 revision)

**Event:** Meta × HF × PyTorch OpenEnv Hackathon Grand Finale, Bangalore
**When:** Saturday April 25, 2026
**Goal:** GRPO-train a small model (Qwen3-0.6B or Qwen2.5-1.5B) to beat its
own untrained baseline on ≥1 task.

Phase 6.5 fundamentally rewired the training loop:
- **Multi-step reward via closure** (Option B) — the GRPO completion is now
  step 1 of a 12-step episode, not a one-shot bandit action.
- **SFT warmup** — 30 expert trajectories teach the action JSON schema
  BEFORE GRPO sees the model, killing the zero-variance generation groups.
- **Groq LLM judge** — episode-end bonus (±1.0) on top of rule rewards.
- **Terminal reward bonus/penalty** — approve+healthy = +2.0, abort/timeout
  = −1.5, approve+unresolved = −0.5. Widens episode reward from [−0.5, +1.0]
  to roughly [−3, +5].

---

## 0. Pre-flight (10 min)

```bash
# Clone + install env deps
git clone https://github.com/Yashash4/devops-pipeline-gym ~/gym
cd ~/gym
pip install -r requirements.txt && pip install -e .

# Env server (keep in a tmux pane)
python -m uvicorn server.app:app --host 127.0.0.1 --port 8000 &
sleep 6
curl -X POST -H "Content-Type: application/json" -d '{}' http://127.0.0.1:8000/reset
# MUST return 200 with clean_deploy observation.

# Training deps
pip install -r requirements-training.txt

# Secrets
export HF_TOKEN=hf_...
export GROQ_API_KEY=gsk_...          # required for --use-judge
export OLLAMA_API_KEY=...            # optional; fallback judge

# Sanity
python training/grpo_train.py --help   # should list --use-judge, --sft-adapter-path, ...
python training/sft_warmup.py --help
python -c "from server.judge_client import GroqJudgeClient; print('OK')"
```

Pinned on Kaggle T4 (validated 2026-04-24): unsloth 2026.4.8, torch 2.10.0+cu128,
transformers 4.56.2, trl 0.24-0.29, bitsandbytes, peft.

---

## 1. SFT warmup (15 min on H100 / T4)

Teaches the model the action JSON schema so GRPO generation groups have
non-zero reward variance.

```bash
python training/sft_warmup.py \
    --model unsloth/Qwen3-0.6B-bnb-4bit \
    --trajectories data/sft_trajectories.jsonl \
    --output-dir ./outputs/sft_warmup \
    --epochs 2 \
    --learning-rate 2e-4 \
    --batch-size 2
```

Output: `./outputs/sft_warmup/final/adapter_config.json` + `adapter_model.safetensors`.
Requires ≥5 non-comment JSONL lines in `data/sft_trajectories.jsonl`.

**If you see zero variance groups even after SFT**, add more trajectories. The
loader accepts anywhere from 5 (minimum) to ~100+ without issue.

---

## 2. Baseline eval (≈20 min)

Capture the UNTRAINED baseline BEFORE GRPO overwrites the weights. Same model
that training will fine-tune.

```bash
python training/eval_baseline.py \
    --model unsloth/Qwen3-0.6B-bnb-4bit \
    --env-url http://127.0.0.1:8000 \
    --output ./outputs/baseline_v2.json \
    --n-seeds 3
```

For reference (Round 1 Qwen 2.5 72B, different model family):

| Task | Round 1 Qwen 72B reference |
|---|---|
| clean_deploy | 0.700 |
| broken_pipeline | 0.482 |
| judgment_call | 0.184 |
| cascading_failure | 0.280 |
| capacity_crisis | 0.250 |
| random_incident | 0.350 |

A 0.6B model will score much lower — that's the headroom GRPO should close.

---

## 3. GRPO training (2-3 hours on H100, 4-6 hours on T4)

**Primary (Tier 1) command — full stack with SFT + judge:**

```bash
python training/grpo_train.py \
    --model unsloth/Qwen3-0.6B-bnb-4bit \
    --sft-adapter-path ./outputs/sft_warmup/final \
    --env-url http://127.0.0.1:8000 \
    --max-steps 50 \
    --num-generations 8 \
    --max-completion-length 512 \
    --prompts-per-task 6 \
    --max-episode-steps 12 \
    --learning-rate 5e-6 \
    --use-judge \
    --judge-model llama-3.3-70b-versatile \
    --judge-weight 1.0 \
    --output-dir ./outputs/grpo_training
```

Flag rationale:
- `--sft-adapter-path` loads the Phase 6.5 SFT warmup, merges into base, then
  GRPO attaches a fresh LoRA on top.
- `--max-episode-steps 12` caps multi-step rollout per reward call.
- `--num-generations 8` keeps group variance high enough (T4 dry-run with 2
  had `frac_reward_zero_std=0.5`).
- `--max-completion-length 512` avoids the 256-ceiling clipping seen in Phase 6.
- `--use-judge` adds the Groq episode-end bonus.
- `--judge-weight 1.0` — Tier 1 starts here.
- vLLM intentionally omitted — enable only after the primary run converges.

**Monitor every 10 steps:**

```bash
# Reward magnitude — Phase 6.5 target is roughly [-3, +5] episode-summed.
# If you still see [-0.1, +0.1], the multi-step path isn't firing.
grep -E 'reward_mean|reward_std|frac_reward_zero_std' outputs/grpo_training/*.log | tail -20

# Generation health
grep -E 'clipped_ratio' outputs/grpo_training/*.log | tail -5
```

---

## 4. Tier 1 assessment (after step 20)

| Outcome | Probability | Action |
|---|---|---|
| reward_mean climbing, span > 0.5 | ~45% | Let it run to 50 steps. Go to §6. |
| Noisy but trending up | ~30% | Keep running. Re-check at step 30. |
| Dead flat (std < 0.1 across 20 steps) | ~20% | **Kill, go to Tier 2.** |
| Diverging (reward dropping) | ~5% | Kill, halve LR to 2.5e-6, restart. |

---

## 5. Escalation tiers

### Tier 1 → Tier 2 — bump judge weight

If reward is flat after step 20, the rule reward isn't discriminating and the
judge isn't weighted enough. Restart with:

```bash
... --judge-weight 1.5
```

### Tier 2 → Tier 3 — reduce group size + prompts

If STILL flat after step 30 of Tier 2:

```bash
... --num-generations 4 --prompts-per-task 4 --judge-weight 1.5
```

Smaller groups = faster steps, more opportunities to see gradient.

### Tier 3 → Tier 4 — disable judge, fall back to rules

If reward collapses to a single value (judge saturation), restart WITHOUT judge:

```bash
... (remove --use-judge)
```

Rule-based reward on its own after SFT warmup should still give gradient.

### Tier 5 — fallback model

If Qwen 0.6B exhausts all tiers, swap to Qwen 1.5B (more capacity):

```bash
... --model unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit \
    --sft-adapter-path ./outputs/sft_warmup_1_5b/final
```

Note: you'd have to re-run SFT warmup for 1.5B first.

### Tier 6 — submit what we have

If nothing above converges by Saturday 6 PM, submit Tier 1's log/curve as-is.
Reframe the pitch as *"here's what we learned about the reward landscape"*.
Honesty beats overclaiming.

---

## 6. Trained eval (≈20 min)

```bash
python training/eval_baseline.py \
    --model ./outputs/grpo_training/final \
    --env-url http://127.0.0.1:8000 \
    --output ./outputs/trained_v2.json \
    --n-seeds 3
```

---

## 7. Before/after chart

```bash
python training/generate_comparison_chart.py \
    --baseline ./outputs/baseline_v2.json \
    --trained ./outputs/trained_v2.json \
    --output ./outputs/before_after.png \
    --title "Qwen3 0.6B — untrained vs GRPO+SFT+judge (50 steps)"
```

---

## 8. Export to HF Hub (before Sunday pitch)

```bash
# Adapter only — do NOT merge + upcast per Participant Help Guide.
huggingface-cli login
python -c "
from huggingface_hub import upload_folder
upload_folder(
    repo_id='yashash045/devops-agent-trained',
    folder_path='outputs/grpo_training/final',
    commit_message='GRPO+SFT+judge Qwen3-0.6B, Saturday 2026-04-25',
)
"
```

---

## 9. Artifacts for pitch

Copy to `round2/REF/artifacts/` (gitignored, local only):

- [ ] `outputs/baseline_v2.json`
- [ ] `outputs/trained_v2.json`
- [ ] `outputs/grpo_training/reward_curve.png`
- [ ] `outputs/before_after.png`
- [ ] One raw episode log per task (untrained failure + trained success if possible)
- [ ] Screenshot of the HF Hub adapter page

---

## 10. Phase 6.5 diagnostic checklist

Per Phase 6.5 BUILD_LOG — confirm these before claiming success:

- [ ] SFT warmup completed and adapter_config.json exists at `outputs/sft_warmup/final/`
- [ ] GRPO training loaded the SFT adapter (log line "SFT adapter merged into base model")
- [ ] First reward_mean in log is between −3 and +5 (not between −0.1 and +0.1)
- [ ] Terminal bonus fires: for episodes ending in approve with all_healthy, reward spikes by ~+2
- [ ] Judge bonus: look for Groq calls in debug output (when --use-judge is on)
- [ ] `frac_reward_zero_std` drops below 0.3 after SFT warmup (was 0.5-1.0 before)

If any of these fail, diagnose before continuing — they're the signals that
Phase 6.5 is doing what it was supposed to.

---

## 11. Emergency rollback

```bash
git checkout round2-start    # pre-Round-2 commit (tagged)
# OR
git checkout phase6.5-start  # pre-6.5 commit (tagged)
```

Round 2 code is additive. `round2-start` gives a working Round 1 env.
`phase6.5-start` gives Phase 7 state (single-step reward, no SFT, no judge)
— still works, just with the known training-loop bug.
