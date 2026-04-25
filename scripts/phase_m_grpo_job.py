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
#     "httpx>=0.25",
#     "trackio",
# ]
# ///
"""Phase M Stage 2: GRPO 200 steps on HF Jobs L4.

Clones repo, pulls SFT adapter from Hub, boots env-server in background,
runs GRPO training, pushes final adapter to Hub.

Final adapter: yashash045/devops-pipeline-gym-trained
"""

import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

WORKDIR = Path("/workspace/devops-pipeline-gym")

# Step 1: Clone repo
if not WORKDIR.exists():
    print("[1/7] Cloning repo...", flush=True)
    subprocess.run(
        ["git", "clone", "https://github.com/Yashash4/devops-pipeline-gym", str(WORKDIR)],
        check=True,
    )
os.chdir(WORKDIR)

# Step 2: Install package (UV envs don't ship pip — use UV's resolver)
print("[2/7] Installing devops-pipeline-gym package...", flush=True)
subprocess.run(
    ["uv", "pip", "install", "--system", "-e", ".", "--quiet"],
    check=True,
)

# Step 3: Pull SFT adapter from Hub
print("[3/7] Pulling SFT adapter from Hub...", flush=True)
from huggingface_hub import snapshot_download

sft_path = snapshot_download(
    "yashash045/devops-pipeline-gym-sft-adapter",
    token=os.environ["HF_TOKEN"],
    local_dir="/workspace/sft_adapter",
)
print(f"    SFT adapter at: {sft_path}", flush=True)

# Step 4: Boot env-server in background
print("[4/7] Booting env-server on localhost:8000...", flush=True)
env_proc = subprocess.Popen(
    [
        sys.executable, "-m", "uvicorn", "server.app:app",
        "--host", "127.0.0.1", "--port", "8000",
    ],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)

time.sleep(8)

# Verify env is responding
for i in range(10):
    try:
        req = urllib.request.Request(
            "http://localhost:8000/reset",
            method="POST",
            data=b"{}",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5) as r:
            if r.status == 200:
                print(f"    Env-server is healthy (after {8 + i*2}s)", flush=True)
                break
    except Exception:
        pass
    time.sleep(2)
else:
    env_proc.terminate()
    raise RuntimeError("env-server failed to come up in 28s")

try:
    # Step 5: Run GRPO training
    print("[5/7] Running GRPO 200 steps (Qwen3-1.7B + SFT adapter)...", flush=True)
    subprocess.run(
        [
            sys.executable, "training/grpo_train.py",
            "--model", "unsloth/Qwen3-1.7B-bnb-4bit",
            "--sft-adapter-path", sft_path,
            "--env-url", "http://localhost:8000",
            "--max-steps", "200",
            "--batch-size", "4",
            "--num-generations", "8",
            "--learning-rate", "5e-6",
            "--output-dir", "/workspace/grpo_output",
        ],
        check=True,
        env={
            **os.environ,
            "TRACKIO_SPACE_NAME": "yashash045/dpg-trackio",
        },
    )
finally:
    # Step 6: Stop env-server
    print("[6/7] Stopping env-server...", flush=True)
    env_proc.send_signal(signal.SIGTERM)
    try:
        env_proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        env_proc.kill()

# Step 7: Push final adapter to Hub
print("[7/7] Pushing GRPO-trained adapter to Hub...", flush=True)
from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(
    "yashash045/devops-pipeline-gym-trained",
    repo_type="model",
    exist_ok=True,
    private=False,
)
api.upload_folder(
    folder_path="/workspace/grpo_output",
    repo_id="yashash045/devops-pipeline-gym-trained",
    repo_type="model",
    commit_message="Phase M Stage 2: GRPO 200 steps adapter",
)

print("\n=== Phase M Stage 2 COMPLETE ===", flush=True)
print("Final adapter at: https://huggingface.co/yashash045/devops-pipeline-gym-trained", flush=True)
print("Track-IO logs at: https://huggingface.co/spaces/yashash045/dpg-trackio", flush=True)
