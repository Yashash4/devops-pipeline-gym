#!/bin/bash
# Launch GRPO retry on L40S.
#
# IMPORTANT: This is the budget-burner. ~$8-10 on L40S × 1 for 3-4 hours.
# Run only AFTER eval_base + eval_sft come back with the SFT delta confirmed.
# If SFT delta is strong (>=+1.0 mean reward), GRPO retry is optional.
# If SFT delta is weak, GRPO retry is the last shot.
#
# Prerequisites:
#   - .env at project root with HF_TOKEN=hf_...
#   - SFT adapter exists at yashash045/devops-pipeline-gym-sft-adapter
#
# Usage:
#   cd devops-pipeline-gym
#   ./scripts/hf_jobs/run_grpo_retry.sh

set -e

if [ -f .env ]; then
    source .env
fi

if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN not set"
    exit 1
fi

export HF_TOKEN

echo "=========================================="
echo "GRPO RETRY (300 steps from SFT adapter)"
echo "GPU:    L40S × 1"
echo "Cost:   ~\$8-10"
echo "Time:   ~3-4 hours"
echo "Adapter: yashash045/devops-pipeline-gym-trained (overwrites)"
echo "=========================================="
read -r -p "Confirm? (y/N): " confirm
if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
    echo "Aborted."
    exit 0
fi

hf jobs uv run \
    --flavor l40sx1 \
    --timeout 5h \
    --secrets HF_TOKEN \
    scripts/hf_jobs/grpo_retry_job.py

echo ""
echo "=== GRPO retry submitted. Watch with: hf jobs ls ==="
