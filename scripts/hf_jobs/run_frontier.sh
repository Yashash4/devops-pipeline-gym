#!/bin/bash
# Frontier-model baselines via HF Router. CPU-only, ~$0.50 total.
#
# Tests our env against 4 frontier models (Qwen-72B, Llama-70B, DeepSeek-V3, Mistral-Large)
# to set up the headline: "1.7B trained beats giant untrained."

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
echo "FRONTIER BASELINES (CPU-only)"
echo "Models: Qwen2.5-72B, Llama-3.3-70B, DeepSeek-V3.1, Mistral-Large"
echo "Cost:   ~\$0.50 total (HF Router inference)"
echo "Time:   ~2 hours wall (4 models × ~30 min each)"
echo "=========================================="

# CPU-basic flavor — no GPU needed since we're using HF Router
hf jobs uv run \
    --flavor cpu-basic \
    --timeout 3h \
    --secrets HF_TOKEN \
    scripts/hf_jobs/eval_frontier_job.py

echo ""
echo "=== Frontier baselines submitted. Watch with: hf jobs ls ==="
