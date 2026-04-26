# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "huggingface_hub>=0.30",
#     "fastapi>=0.104",
#     "uvicorn>=0.24",
#     "openenv-core[core]>=0.2.2",
#     "pydantic>=2.0",
#     "httpx>=0.25",
#     "openai>=1.40",
# ]
# ///
"""Frontier-model baselines via HF Inference Router. CPU-only, ~$0.60 total.

Runs eval_baseline.py against 5 frontier models so we can publish a
'1.7B trained beats 70B+ untrained' headline. No GPU needed — all
inference happens through the router.

Models tested:
  - Qwen/Qwen2.5-72B-Instruct       (72B dense, our existing baseline reference)
  - meta-llama/Llama-3.3-70B-Instruct (70B dense, different family)
  - deepseek-ai/DeepSeek-V3.1        (671B MoE, sparse giant)
  - mistralai/Mistral-Large-Instruct-2411 (123B dense, diverse family)
  - openai/gpt-oss-120b              (120B MoE, OpenAI's open weights)

Each model: 6 tasks × 3 seeds = 18 episodes (kept lower than 5 to control cost).
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

FRONTIER_MODELS = [
    ("Qwen/Qwen2.5-72B-Instruct", "qwen25_72b"),
    ("meta-llama/Llama-3.3-70B-Instruct", "llama33_70b"),
    ("deepseek-ai/DeepSeek-V3.1", "deepseek_v31"),
    ("mistralai/Mistral-Large-Instruct-2411", "mistral_large"),
    ("openai/gpt-oss-120b", "gpt_oss_120b"),
]


def main():
    # 1. Clone
    if not WORKDIR.exists():
        print(f"[1/5] Cloning {REPO_URL}...", flush=True)
        subprocess.run(["git", "clone", REPO_URL, str(WORKDIR)], check=True)
    os.chdir(WORKDIR)

    # 2. Install
    print("[2/5] Installing devops-pipeline-gym...", flush=True)
    subprocess.run(["uv", "pip", "install", "-e", ".", "--quiet"], check=True)

    # 3. Boot env-server
    print("[3/5] Booting env-server on localhost:8000...", flush=True)
    env_log_fh = open("/workspace/env_server.log", "w")
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
            raise RuntimeError(f"env-server died, exit={env_proc.returncode}")
        try:
            req = urllib.request.Request(
                "http://localhost:8000/reset", method="POST", data=b"{}",
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=5) as r:
                if r.status == 200:
                    health_ok = True
                    print(f"    healthy after {15 + i*1.5:.0f}s", flush=True)
                    break
        except Exception:
            pass
        time.sleep(1.5)
    if not health_ok:
        raise RuntimeError("env-server did not come up")

    try:
        # 4. Run eval against each frontier model
        for model_id, output_tag in FRONTIER_MODELS:
            output_json = f"/workspace/eval_frontier_{output_tag}.json"
            print(f"\n[4/5] Eval against {model_id}...", flush=True)
            try:
                subprocess.run(
                    [
                        sys.executable, "training/eval_baseline.py",
                        "--model", model_id,
                        "--use-hf-router",
                        "--env-url", "http://localhost:8000",
                        "--output", output_json,
                        "--n-seeds", "3",
                        "--temperature", "0.3",
                        "--max-tokens", "300",
                    ],
                    check=True,
                    timeout=1800,  # 30 min per model
                )
            except subprocess.TimeoutExpired:
                print(f"    TIMEOUT on {model_id} after 30 min — skipping", flush=True)
            except subprocess.CalledProcessError as e:
                print(f"    FAILED on {model_id}: {e} — skipping", flush=True)

    finally:
        # 5. Stop env
        env_proc.send_signal(signal.SIGTERM)
        try:
            env_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            env_proc.kill()

    # 6. Push all results to Hub
    print("[5/5] Uploading frontier baselines to Hub...", flush=True)
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ["HF_TOKEN"])
    for _, tag in FRONTIER_MODELS:
        json_path = f"/workspace/eval_frontier_{tag}.json"
        if os.path.exists(json_path):
            api.upload_file(
                path_or_fileobj=json_path,
                path_in_repo=f"eval_frontier_{tag}.json",
                repo_id="yashash045/devops-pipeline-gym-sft-adapter",
                repo_type="model",
                commit_message=f"Frontier baseline: {tag}",
            )
            print(f"    uploaded eval_frontier_{tag}.json", flush=True)

    print("\n=== FRONTIER BASELINES COMPLETE ===", flush=True)
    print("Results: https://huggingface.co/yashash045/devops-pipeline-gym-sft-adapter/tree/main", flush=True)


if __name__ == "__main__":
    main()
