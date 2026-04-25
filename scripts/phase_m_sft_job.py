# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch>=2.4",
#     "unsloth",
#     "trl>=0.12",
#     "peft>=0.13",
#     "datasets>=3.0",
#     "bitsandbytes>=0.43",
#     "huggingface_hub>=0.30",
#     "fastapi>=0.104",
#     "uvicorn>=0.24",
#     "openenv-core[core]>=0.2.2",
#     "pydantic>=2.0",
#     "trackio",
# ]
# ///
"""Phase M Stage 1: SFT warmup on HF Jobs L4.

Clones the repo, installs dependencies, runs SFT, pushes adapter to Hub.

Adapter destination: yashash045/devops-pipeline-gym-sft-adapter
"""

import os
import subprocess
import sys
from pathlib import Path

# Step 1: Clone repo (HF Jobs working dir is /workspace by default)
WORKDIR = Path("/workspace/devops-pipeline-gym")
if not WORKDIR.exists():
    print("[1/5] Cloning repo...", flush=True)
    subprocess.run(
        ["git", "clone", "https://github.com/Yashash4/devops-pipeline-gym", str(WORKDIR)],
        check=True,
    )
os.chdir(WORKDIR)

# Step 2: Install local package (UV envs don't ship pip — use UV's resolver)
print("[2/5] Installing devops-pipeline-gym package...", flush=True)
subprocess.run(
    ["uv", "pip", "install", "-e", ".", "--quiet"],
    check=True,
)

# Step 3: Verify SFT script + dataset
print("[3/5] Verifying training script + dataset...", flush=True)
assert Path("training/sft_warmup.py").exists(), "sft_warmup.py missing"
assert Path("data/sft_trajectories.jsonl").exists(), "trajectories missing"

with open("data/sft_trajectories.jsonl") as f:
    n_traj = sum(1 for line in f if line.strip() and not line.lstrip().startswith("#"))
print(f"    Loaded {n_traj} SFT trajectories (target: 78)", flush=True)

# Step 4: Run SFT warmup
print("[4/5] Running SFT warmup (Qwen3-1.7B QLoRA + LoRA r=16/alpha=32)...", flush=True)
subprocess.run(
    [
        sys.executable, "training/sft_warmup.py",
        "--model", "unsloth/Qwen3-1.7B-bnb-4bit",
        "--trajectories", "data/sft_trajectories.jsonl",
        "--output-dir", "/workspace/sft_output",
        "--epochs", "2",
        "--batch-size", "4",
        "--lora-r", "16",
        "--lora-alpha", "32",
    ],
    check=True,
)

# Step 5: Push SFT adapter to Hub
print("[5/5] Pushing SFT adapter to Hub...", flush=True)
from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(
    "yashash045/devops-pipeline-gym-sft-adapter",
    repo_type="model",
    exist_ok=True,
    private=False,
)
api.upload_folder(
    folder_path="/workspace/sft_output",
    repo_id="yashash045/devops-pipeline-gym-sft-adapter",
    repo_type="model",
    commit_message="Phase M Stage 1: SFT warmup adapter",
)

print("\n=== Phase M Stage 1 COMPLETE ===", flush=True)
print("Adapter at: https://huggingface.co/yashash045/devops-pipeline-gym-sft-adapter", flush=True)
