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
    ["uv", "pip", "install", "-e", ".", "--quiet"],
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

# Verify expected adapter file exists at <sft_path>/final
adapter_config = os.path.join(sft_path, "final", "adapter_config.json")
if not os.path.exists(adapter_config):
    print(f"ERROR: adapter_config.json not found at {adapter_config}", flush=True)
    print(f"Contents of {sft_path}:", flush=True)
    for root, dirs, files in os.walk(sft_path):
        for f in files[:20]:
            print(f"  {os.path.join(root, f)}", flush=True)
    raise FileNotFoundError(f"adapter_config.json missing at {adapter_config}")
print(f"    [OK] Verified adapter_config.json exists", flush=True)
print(f"    SFT adapter root: {sft_path}", flush=True)
print(f"    SFT adapter resolved: {sft_path}/final (adapter_config.json + adapter_model.safetensors)", flush=True)

# Step 4: Boot env-server in background
print("[4/7] Booting env-server on localhost:8000...", flush=True)
env_log_path = "/workspace/env_server.log"
env_log_fh = open(env_log_path, "w")
env_proc = subprocess.Popen(
    [
        sys.executable, "-m", "uvicorn", "server.app:app",
        "--host", "127.0.0.1", "--port", "8000",
        "--log-level", "info",
    ],
    stdout=env_log_fh,
    stderr=subprocess.STDOUT,
)
print(f"    env-server PID: {env_proc.pid}, logs: {env_log_path}", flush=True)

# Give uvicorn time to boot. L4 cold start + import-time work can be slow.
time.sleep(15)

health_ok = False
last_error = None

# Try up to 90 more seconds (60 attempts at 1.5s each)
for i in range(60):
    # Check if process died
    if env_proc.poll() is not None:
        env_log_fh.flush()
        with open(env_log_path) as f:
            log_content = f.read()
        print(f"    env-server PROCESS DIED with exit code {env_proc.returncode}", flush=True)
        print(f"    --- env-server log ---", flush=True)
        print(log_content[-3000:], flush=True)
        print(f"    --- end log ---", flush=True)
        raise RuntimeError(f"env-server died during boot. Exit: {env_proc.returncode}")

    # Try /reset health check
    try:
        req = urllib.request.Request(
            "http://localhost:8000/reset",
            method="POST",
            data=b"{}",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5) as r:
            if r.status == 200:
                # Also verify /tasks responds (stricter check)
                req2 = urllib.request.Request("http://localhost:8000/tasks")
                with urllib.request.urlopen(req2, timeout=5) as r2:
                    if r2.status == 200:
                        print(f"    Env-server is healthy (after {15 + i*1.5:.0f}s)", flush=True)
                        health_ok = True
                        break
    except Exception as e:
        last_error = str(e)[:120]

    time.sleep(1.5)

if not health_ok:
    env_log_fh.flush()
    with open(env_log_path) as f:
        log_content = f.read()
    print(f"    Health check FAILED. Last error: {last_error}", flush=True)
    print(f"    --- env-server log (last 3KB) ---", flush=True)
    print(log_content[-3000:], flush=True)
    print(f"    --- end log ---", flush=True)
    env_proc.terminate()
    try:
        env_proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        env_proc.kill()
    raise RuntimeError(f"env-server failed to come up in 105s")

try:
    # Step 5: Run GRPO training
    print("[5/7] Running GRPO 200 steps (Stage B production run; Qwen3-1.7B + SFT adapter)...", flush=True)
    subprocess.run(
        [
            sys.executable, "training/grpo_train.py",
            "--model", "unsloth/Qwen3-1.7B-bnb-4bit",
            "--sft-adapter-path", f"{sft_path}/final",
            "--env-url", "http://localhost:8000",
            "--max-steps", "200",
            "--batch-size", "4",
            "--num-generations", "8",
            "--learning-rate", "5e-6",
            "--max-completion-length", "512",
            "--output-dir", "/workspace/grpo_output",
        ],
        check=True,
        env={
            **os.environ,
            "TRACKIO_SPACE_ID": "yashash045/dpg-trackio",
            "TRACKIO_PROJECT": "devops-pipeline-gym-grpo",
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
