# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch>=2.4",
#     "transformers>=4.45",
#     "peft>=0.10",
#     "bitsandbytes>=0.43",
#     "accelerate>=0.30",
#     "huggingface_hub>=0.30",
#     "fastapi>=0.104",
#     "uvicorn>=0.24",
#     "openenv-core[core]>=0.2.2",
#     "pydantic>=2.0",
#     "httpx>=0.25",
#     "openai>=1.40",
# ]
# ///
"""Multi-seed multi-task eval as an HF Job.

Runs Qwen3-1.7B (with optional adapter) across 6 tasks × 5 seeds, producing
a JSON results file. Pushes results to the model repo as `eval_<mode>.json`.

Modes:
  base : Qwen3-1.7B-bnb-4bit, no adapter
  sft  : Qwen3-1.7B-bnb-4bit + SFT adapter from yashash045/devops-pipeline-gym-sft-adapter
  grpo : Qwen3-1.7B-bnb-4bit + SFT + GRPO adapter from yashash045/devops-pipeline-gym-trained

For random_incident, eval seeds are 5000-5004 (training used 6000+, no overlap).

Usage on HF Jobs:
  hf jobs uv run --flavor t4-small --timeout 1h --secrets HF_TOKEN \\
    scripts/hf_jobs/eval_job.py --mode base
"""

import argparse
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

WORKDIR = Path("/workspace/devops-pipeline-gym")
REPO_URL = "https://github.com/Yashash4/devops-pipeline-gym"

ADAPTERS = {
    "base": None,
    "sft": "yashash045/devops-pipeline-gym-sft-adapter",
    "grpo": "yashash045/devops-pipeline-gym-trained",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["base", "sft", "grpo"], required=True)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--push-to", default="yashash045/devops-pipeline-gym-sft-adapter",
                        help="Hub model repo to push the eval JSON to")
    args = parser.parse_args()

    # 1. Clone repo
    if not WORKDIR.exists():
        print(f"[1/5] Cloning {REPO_URL}...", flush=True)
        subprocess.run(["git", "clone", REPO_URL, str(WORKDIR)], check=True)
    os.chdir(WORKDIR)

    # 2. Install package
    print("[2/5] Installing devops-pipeline-gym...", flush=True)
    subprocess.run(["uv", "pip", "install", "-e", ".", "--quiet"], check=True)

    # 3. Boot env-server
    print("[3/5] Booting env-server on localhost:8000...", flush=True)
    env_log_path = "/workspace/env_server.log"
    env_log_fh = open(env_log_path, "w")
    env_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "server.app:app",
         "--host", "127.0.0.1", "--port", "8000", "--log-level", "info"],
        stdout=env_log_fh,
        stderr=subprocess.STDOUT,
    )

    # Wait for env-server health
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
        # 4. Run eval_baseline.py
        adapter = ADAPTERS[args.mode]
        output_json = f"/workspace/eval_{args.mode}.json"

        cmd = [
            sys.executable, "training/eval_baseline.py",
            "--model", "unsloth/Qwen3-1.7B-bnb-4bit",
            "--env-url", "http://localhost:8000",
            "--output", output_json,
            "--n-seeds", str(args.n_seeds),
        ]
        # Note: eval_baseline.py treats --model as either HF id or local dir.
        # For sft/grpo we pass the HF adapter id - the script's _ModelAdapter
        # will check `is_dir()` and skip the PEFT path; we need to download
        # the adapter to a local dir first then pass that path.
        if adapter:
            print(f"[4a/5] Downloading adapter {adapter}...", flush=True)
            from huggingface_hub import snapshot_download
            local_adapter_dir = snapshot_download(
                repo_id=adapter,
                local_dir=f"/workspace/{args.mode}_adapter",
            )
            # Replace --model with the local adapter dir so eval_baseline.py's
            # PEFT loader picks it up
            cmd[cmd.index("--model") + 1] = local_adapter_dir

        # Override eval seeds to avoid overlap with training seeds (6000+)
        env_with_eval_seed_offset = {**os.environ, "DEVOPS_EVAL_SEED_BASE": "5000"}

        print(f"[4b/5] Running eval (mode={args.mode}, n_seeds={args.n_seeds}, temp={args.temperature})...", flush=True)
        subprocess.run(cmd, check=True, env=env_with_eval_seed_offset)

    finally:
        # Stop env-server
        env_proc.send_signal(signal.SIGTERM)
        try:
            env_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            env_proc.kill()

    # 5. Push results to Hub
    print(f"[5/5] Uploading results to {args.push_to}...", flush=True)
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.upload_file(
        path_or_fileobj=output_json,
        path_in_repo=f"eval_{args.mode}.json",
        repo_id=args.push_to,
        repo_type="model",
        commit_message=f"Add multi-seed eval results: mode={args.mode}",
    )
    print(f"\n=== Eval {args.mode} COMPLETE ===", flush=True)
    print(f"Results: https://huggingface.co/{args.push_to}/blob/main/eval_{args.mode}.json", flush=True)


if __name__ == "__main__":
    main()
