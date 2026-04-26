"""Kaggle Account 1 cell: eval_base + eval_sft on T4.

Usage in Kaggle (paste this 2-line cell):
    !wget -q https://raw.githubusercontent.com/Yashash4/devops-pipeline-gym/main/scripts/kaggle_cell_acc1.py -O /tmp/cell.py
    exec(open("/tmp/cell.py").read())
"""
import os
import subprocess
import sys
from pathlib import Path

# 1. Auth via Kaggle secret
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    print("HF_TOKEN loaded from Kaggle secrets", flush=True)
except Exception as e:
    print(f"WARNING: could not load HF_TOKEN from Kaggle secrets: {e}", flush=True)

# 2. Clone repo
WORKDIR = Path("/kaggle/working/dpg")
if not WORKDIR.exists():
    print("Cloning devops-pipeline-gym...", flush=True)
    subprocess.run(["git", "clone",
                    "https://github.com/Yashash4/devops-pipeline-gym",
                    str(WORKDIR)], check=True)
os.chdir(WORKDIR)

# 3. Install package
print("Installing devops-pipeline-gym...", flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "."], check=True)

# 4. Make scripts importable
sys.path.insert(0, str(WORKDIR))

# 5. Run sequential evals
import scripts.kaggle_eval as ke
ke.run_eval(mode="base", n_seeds=5)
ke.run_eval(mode="sft", n_seeds=5)

print("\n=== Account 1 DONE — eval_base.json + eval_sft.json uploaded to Hub ===", flush=True)
