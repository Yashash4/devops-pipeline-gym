#!/bin/bash
# Launch the 3 multi-seed evals on HF Jobs.
#
# Order: base + sft launched in parallel (different jobs, both T4),
# eval_grpo launched after we have GRPO adapter.
#
# Estimated cost: ~$3 total ($1 each on T4-small for ~45 min)
# Estimated wall time: ~45 min for parallel evals (base + sft together)
#
# Prerequisites:
#   - .env at project root with HF_TOKEN=hf_...
#   - hf CLI installed (huggingface_hub >= 0.30)
#   - hf auth login (or HF_TOKEN exported)
#
# Usage:
#   cd devops-pipeline-gym
#   ./scripts/hf_jobs/run_evals.sh base
#   ./scripts/hf_jobs/run_evals.sh sft
#   ./scripts/hf_jobs/run_evals.sh grpo   # only after GRPO retry completes

set -e

if [ -f .env ]; then
    # shellcheck disable=SC1091
    source .env
fi

if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN not set in .env or environment"
    exit 1
fi

export HF_TOKEN

MODE="${1:-base}"

case "$MODE" in
    base|sft|grpo) ;;
    *) echo "Usage: $0 {base|sft|grpo}"; exit 1 ;;
esac

echo "=========================================="
echo "Eval mode: $MODE"
echo "GPU:       T4-small (~\$1/hr)"
echo "Timeout:   1h"
echo "=========================================="

hf jobs uv run \
    --flavor t4-small \
    --timeout 1h \
    --secrets HF_TOKEN \
    scripts/hf_jobs/eval_job.py \
    -- --mode "$MODE" --n-seeds 5 --temperature 0.3

echo ""
echo "=== Eval $MODE submitted. ==="
echo "Watch logs: hf jobs ls"
echo "Results will land at: https://huggingface.co/yashash045/devops-pipeline-gym-sft-adapter/blob/main/eval_${MODE}.json"
