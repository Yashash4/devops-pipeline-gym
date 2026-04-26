"""Kaggle T4 single-cell eval helper for devops-pipeline-gym.

Designed to run in ONE Kaggle notebook cell. Free T4 GPU. No HF Jobs cost.

Usage in Kaggle notebook (paste this entire script as one cell, or use
the 4-line shim below that imports and calls run_eval()):

    # Single-cell shim
    !git clone https://github.com/Yashash4/devops-pipeline-gym /kaggle/working/dpg 2>/dev/null
    %cd /kaggle/working/dpg
    !pip install -q -e .
    import scripts.kaggle_eval as ke; ke.run_eval(mode="base")  # or "sft" or "grpo"

Modes:
    base : Qwen3-1.7B-bnb-4bit, no adapter
    sft  : + SFT adapter (yashash045/devops-pipeline-gym-sft-adapter)
    grpo : + SFT + GRPO adapter (yashash045/devops-pipeline-gym-trained)

Set HF_TOKEN in Kaggle Add-ons → Secrets before running. Results upload
to the Hub model repo as eval_<mode>.json (or saved locally if HF_TOKEN
absent).
"""

import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def boot_env_server(port: int = 8000, timeout_s: int = 105):
    """Boot uvicorn on localhost in background, wait for /reset 200."""
    log_path = "/tmp/env_server.log"
    log_fh = open(log_path, "w")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "server.app:app",
         "--host", "127.0.0.1", "--port", str(port),
         "--log-level", "info"],
        stdout=log_fh, stderr=subprocess.STDOUT,
    )
    time.sleep(15)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            with open(log_path) as f:
                tail = f.read()[-2000:]
            raise RuntimeError(f"env-server died:\n{tail}")
        try:
            req = urllib.request.Request(
                f"http://localhost:{port}/reset", method="POST",
                data=b"{}", headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=5) as r:
                if r.status == 200:
                    print(f"env-server healthy (PID {proc.pid})", flush=True)
                    return proc
        except Exception:
            pass
        time.sleep(1.5)
    raise RuntimeError(f"env-server failed health check in {timeout_s}s")


def _ensure_kaggle_deps():
    """Upgrade bitsandbytes on Kaggle (default image ships <0.46 which can't 4-bit
    quantize Qwen3 models with the API our eval_baseline.py uses).
    Idempotent — safe to call multiple times. Costs ~10s on first call."""
    print("[deps] Upgrading bitsandbytes>=0.46.1 (Kaggle ships an older version)...",
          flush=True)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-U",
         "bitsandbytes>=0.46.1"],
        check=False,  # don't crash the whole eval if pip flakes; bnb may already be new enough
    )


def run_eval(mode: str = "base", n_seeds: int = 5, temperature: float = 0.3,
             upload_to_hub: bool = True):
    """Run multi-seed eval on T4. Saves to /kaggle/working/eval_<mode>.json."""

    assert mode in ("base", "sft", "grpo"), f"mode must be base/sft/grpo, got {mode}"
    print(f"=== Eval mode={mode} n_seeds={n_seeds} temp={temperature} ===", flush=True)

    # 0. Ensure bnb >= 0.46 (Kaggle image fix)
    _ensure_kaggle_deps()

    adapters = {
        "base": None,
        "sft": "yashash045/devops-pipeline-gym-sft-adapter",
        "grpo": "yashash045/devops-pipeline-gym-trained",
    }

    # 1. Boot env-server
    print("[1/4] Booting env-server...", flush=True)
    env_proc = boot_env_server()

    try:
        # 2. Download adapter if needed
        model_arg = "unsloth/Qwen3-1.7B-bnb-4bit"
        if adapters[mode]:
            print(f"[2/4] Downloading adapter {adapters[mode]}...", flush=True)
            from huggingface_hub import snapshot_download
            model_arg = snapshot_download(
                repo_id=adapters[mode],
                local_dir=f"/kaggle/working/{mode}_adapter",
            )
            print(f"    adapter local: {model_arg}", flush=True)

        # 3. Run eval_baseline.py
        output_json = f"/kaggle/working/eval_{mode}.json"
        print(f"[3/4] Running eval (output: {output_json})...", flush=True)
        cmd = [
            sys.executable, "training/eval_baseline.py",
            "--model", model_arg,
            "--env-url", "http://localhost:8000",
            "--output", output_json,
            "--n-seeds", str(n_seeds),
        ]
        subprocess.run(cmd, check=True, env={
            **os.environ,
            "DEVOPS_EVAL_SEED_BASE": "5000",  # avoid training seeds (6000+)
        })

        # 4. Optional Hub upload
        if upload_to_hub and os.environ.get("HF_TOKEN"):
            print("[4/4] Uploading to Hub...", flush=True)
            try:
                from huggingface_hub import HfApi
                api = HfApi(token=os.environ["HF_TOKEN"])
                api.upload_file(
                    path_or_fileobj=output_json,
                    path_in_repo=f"eval_{mode}.json",
                    repo_id="yashash045/devops-pipeline-gym-sft-adapter",
                    repo_type="model",
                    commit_message=f"Kaggle eval: mode={mode}, n_seeds={n_seeds}",
                )
                print(f"    uploaded: https://huggingface.co/yashash045/"
                      f"devops-pipeline-gym-sft-adapter/blob/main/eval_{mode}.json",
                      flush=True)
            except Exception as e:
                print(f"    upload failed (saved locally): {e}", flush=True)
        else:
            print(f"[4/4] Saved locally: {output_json} (set HF_TOKEN to auto-upload)",
                  flush=True)
    finally:
        env_proc.send_signal(signal.SIGTERM)
        try:
            env_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            env_proc.kill()

    print(f"\n=== EVAL {mode} DONE ===\n", flush=True)


def run_frontier(models=None, n_seeds: int = 3):
    """Frontier-model baselines via HF Router. CPU-only (no GPU needed)."""
    if models is None:
        models = [
            ("Qwen/Qwen2.5-72B-Instruct", "qwen25_72b"),
            ("meta-llama/Llama-3.3-70B-Instruct", "llama33_70b"),
            ("deepseek-ai/DeepSeek-V3.1", "deepseek_v31"),
            ("mistralai/Mistral-Large-Instruct-2411", "mistral_large"),
            ("openai/gpt-oss-120b", "gpt_oss_120b"),
        ]

    env_proc = boot_env_server()
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=os.environ.get("HF_TOKEN"))

        for model_id, tag in models:
            output_json = f"/kaggle/working/eval_frontier_{tag}.json"
            print(f"\n=== Frontier: {model_id} ===", flush=True)
            try:
                subprocess.run(
                    [sys.executable, "training/eval_baseline.py",
                     "--model", model_id,
                     "--use-hf-router",
                     "--env-url", "http://localhost:8000",
                     "--output", output_json,
                     "--n-seeds", str(n_seeds),
                     "--temperature", "0.3",
                     "--max-tokens", "300"],
                    check=True, timeout=1800,
                )
                if api.token:
                    api.upload_file(
                        path_or_fileobj=output_json,
                        path_in_repo=f"eval_frontier_{tag}.json",
                        repo_id="yashash045/devops-pipeline-gym-sft-adapter",
                        repo_type="model",
                        commit_message=f"Kaggle frontier eval: {tag}",
                    )
            except Exception as e:
                print(f"    {model_id} FAILED: {e}", flush=True)
    finally:
        env_proc.send_signal(signal.SIGTERM)
        try: env_proc.wait(timeout=10)
        except subprocess.TimeoutExpired: env_proc.kill()

    print("\n=== FRONTIER BASELINES DONE ===\n", flush=True)
