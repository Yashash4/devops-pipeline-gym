#!/bin/bash
# Phase M launch script. RUN THIS SATURDAY 8 AM IST.
# GPU: 1x Nvidia L4 (l4x1, 24GB VRAM, BF16 native, FA2)
#
# Prerequisites:
#   - .env file at project root with HF_TOKEN=hf_...
#   - hf CLI installed (huggingface_hub >= 0.30)
#   - Track-IO Space exists at yashash045/dpg-trackio
#
# Usage:
#   cd devops-pipeline-gym
#   ./scripts/run_phase_m.sh

set -e

if [ -f .env ]; then
    # shellcheck disable=SC1091
    source .env
else
    echo "ERROR: .env not found at $(pwd)/.env. Cannot proceed without HF_TOKEN."
    exit 1
fi

if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN not set in .env"
    exit 1
fi

export HF_TOKEN

echo "=========================================="
echo "Phase M: Train Qwen3-1.7B on DevOps Pipeline Gym"
echo "=========================================="
echo "Stage 1: SFT warmup  (~12 min, \$0.16)"
echo "Stage 2: GRPO 200    (~35 min, \$0.47)"
echo "Total estimated:     ~\$0.63 of \$30 budget"
echo "=========================================="
echo ""

echo "[Stage 1] Launching SFT warmup..."
hf jobs uv run \
    --flavor l4x1 \
    --timeout 1h \
    --secrets HF_TOKEN \
    scripts/phase_m_sft_job.py

echo ""
echo "[Stage 1 complete. Sanity check the SFT adapter:]"
echo "  https://huggingface.co/yashash045/devops-pipeline-gym-sft-adapter"
echo ""
read -r -p "Press ENTER to launch Stage 2 (GRPO), or Ctrl+C to abort: "

echo "[Stage 2] Launching GRPO training..."
hf jobs uv run \
    --flavor l4x1 \
    --timeout 4h \
    --secrets HF_TOKEN \
    scripts/phase_m_grpo_job.py

echo ""
echo "=========================================="
echo "Phase M COMPLETE"
echo "=========================================="
echo "Final adapter: https://huggingface.co/yashash045/devops-pipeline-gym-trained"
echo "Track-IO:      https://huggingface.co/spaces/yashash045/dpg-trackio"
echo "Next:          Phase N (eval)"
