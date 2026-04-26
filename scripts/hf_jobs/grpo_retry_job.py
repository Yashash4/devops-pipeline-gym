# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch>=2.4",
#     "trl>=0.12",
#     "peft>=0.18.0,<0.19",
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
"""GRPO retry from SFT adapter — denser config to try to break the flat-reward problem.

Differences from phase_m_grpo_job.py:
  - START FROM SFT ADAPTER (not raw base) — the SFT prior gives a better starting
    distribution; previous "from raw base" run learned but reward stayed flat.
  - 300 steps (was 30) — give RL real time to climb.
  - num_generations=16 (was 4) — more group diversity per prompt.
  - max_completion_length=512 (was 256) — completions were hitting cap; longer
    horizon reduces clipped_ratio.
  - temperature=1.0, beta=0.01 — looser KL, more exploration.
  - mask_truncated_completions=True — drop clipped completions from loss now
    that we've widened the window (clipped_ratio should drop below 0.5).

Cost: ~$8-10 on L40S × 1 (3-4 hours wall clock).

Final adapter: yashash045/devops-pipeline-gym-trained (overwrites the prior one)
"""

import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

WORKDIR = Path("/workspace/devops-pipeline-gym")
REPO_URL = "https://github.com/Yashash4/devops-pipeline-gym"
SFT_ADAPTER = "yashash045/devops-pipeline-gym-sft-adapter"

# Step 1: Clone
if not WORKDIR.exists():
    print(f"[1/7] Cloning {REPO_URL}...", flush=True)
    subprocess.run(["git", "clone", REPO_URL, str(WORKDIR)], check=True)
os.chdir(WORKDIR)

# Step 2: Install
print("[2/7] Installing devops-pipeline-gym...", flush=True)
subprocess.run(["uv", "pip", "install", "-e", ".", "--quiet"], check=True)

# Step 3: Download SFT adapter to local dir (so PEFT can load it as the trainable LoRA)
print(f"[3/7] Downloading SFT adapter {SFT_ADAPTER}...", flush=True)
from huggingface_hub import snapshot_download
import glob
sft_adapter_root = snapshot_download(
    repo_id=SFT_ADAPTER,
    local_dir="/workspace/sft_adapter",
)
print(f"    SFT adapter downloaded to: {sft_adapter_root}", flush=True)

# adapter_config.json may be at root, in final/, or in checkpoint-N/.
# Find it and use that directory as the actual adapter path.
sft_adapter_local = None
candidates = [sft_adapter_root, os.path.join(sft_adapter_root, "final")]
# Also any checkpoint-N dirs (try highest N first)
checkpoints = sorted(
    glob.glob(os.path.join(sft_adapter_root, "checkpoint-*")),
    key=lambda p: int(p.rsplit("-", 1)[-1]) if p.rsplit("-", 1)[-1].isdigit() else 0,
    reverse=True,
)
candidates.extend(checkpoints)

for candidate in candidates:
    if os.path.exists(os.path.join(candidate, "adapter_config.json")):
        sft_adapter_local = candidate
        print(f"    Using adapter at: {sft_adapter_local}", flush=True)
        break

if sft_adapter_local is None:
    # Last resort: recursive glob
    matches = glob.glob(
        os.path.join(sft_adapter_root, "**", "adapter_config.json"),
        recursive=True,
    )
    if matches:
        sft_adapter_local = os.path.dirname(matches[0])
        print(f"    Found adapter via recursive glob: {sft_adapter_local}", flush=True)
    else:
        # Show what was downloaded for debugging
        print("    Files in sft_adapter_root:", flush=True)
        for root, dirs, files in os.walk(sft_adapter_root):
            for f in files[:50]:
                print(f"      {os.path.join(root, f)}", flush=True)
        raise RuntimeError(
            f"No adapter_config.json found in {sft_adapter_root} or any subdir"
        )

# Step 4: Boot env-server
print("[4/7] Booting env-server on localhost:8000...", flush=True)
env_log_path = "/workspace/env_server.log"
env_log_fh = open(env_log_path, "w")
env_proc = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "server.app:app",
     "--host", "127.0.0.1", "--port", "8000", "--log-level", "info"],
    stdout=env_log_fh,
    stderr=subprocess.STDOUT,
)

time.sleep(15)
health_ok = False
for i in range(60):
    if env_proc.poll() is not None:
        with open(env_log_path) as f:
            print(f.read()[-3000:], flush=True)
        raise RuntimeError(f"env-server died, exit={env_proc.returncode}")
    try:
        req = urllib.request.Request(
            "http://localhost:8000/reset", method="POST", data=b"{}",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5) as r:
            if r.status == 200:
                health_ok = True
                print(f"    env-server healthy after {15 + i*1.5:.0f}s", flush=True)
                break
    except Exception:
        pass
    time.sleep(1.5)

if not health_ok:
    raise RuntimeError("env-server failed health check in 105s")

try:
    # Step 5: GRPO with denser config
    print("[5/7] GRPO RETRY: 200 steps from SFT adapter, num_gen=16, max_comp_len=512", flush=True)
    print("    Hypothesis: more samples + longer completions + SFT prior -> reward signal escapes the flat-band trap", flush=True)
    subprocess.run(
        [
            sys.executable, "training/grpo_train.py",
            "--model", "unsloth/Qwen3-1.7B-bnb-4bit",
            "--env-url", "http://localhost:8000",
            "--sft-adapter-path", sft_adapter_local,
            "--max-steps", "200",
            "--batch-size", "1",
            "--grad-accum", "4",
            "--num-generations", "16",
            "--prompts-per-task", "2",
            "--learning-rate", "5e-6",
            "--max-completion-length", "512",
            "--output-dir", "/workspace/grpo_retry_output",
        ],
        check=True,
        env={
            **os.environ,
            "TRACKIO_SPACE_ID": "yashash045/dpg-trackio",
            "TRACKIO_PROJECT": "devops-pipeline-gym-grpo-retry",
            "VLLM_ENFORCE_EAGER": "1",
            "NO_UNSLOTH": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "TRL_EXPERIMENTAL_SILENCE": "1",
        },
    )
finally:
    print("[6/7] Stopping env-server...", flush=True)
    env_proc.send_signal(signal.SIGTERM)
    try:
        env_proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        env_proc.kill()

# Step 7: Push adapter
print("[7/7] Pushing GRPO retry adapter to Hub...", flush=True)
from huggingface_hub import HfApi
import shutil

logs_dir = "/workspace/grpo_retry_output/logs"
os.makedirs(logs_dir, exist_ok=True)
for src_name, src_path in (("env_server.log", "/workspace/env_server.log"),):
    if os.path.exists(src_path):
        shutil.copy2(src_path, os.path.join(logs_dir, src_name))

api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(
    "yashash045/devops-pipeline-gym-trained",
    repo_type="model", exist_ok=True, private=False,
)
api.upload_folder(
    folder_path="/workspace/grpo_retry_output",
    repo_id="yashash045/devops-pipeline-gym-trained",
    repo_type="model",
    commit_message="GRPO retry: 300 steps from SFT, num_gen=16, max_comp_len=512",
)

print("\n=== GRPO RETRY COMPLETE ===", flush=True)
print("Adapter: https://huggingface.co/yashash045/devops-pipeline-gym-trained", flush=True)
print("Track-IO: https://huggingface.co/spaces/yashash045/dpg-trackio", flush=True)
