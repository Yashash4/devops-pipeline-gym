"""Kaggle Account 2 cell: eval_frontier (5 frontier models via HF Router).

Usage in Kaggle (paste this 2-line cell):
    !wget -q https://raw.githubusercontent.com/Yashash4/devops-pipeline-gym/main/scripts/kaggle_cell_acc2.py -O /tmp/cell.py
    exec(open("/tmp/cell.py").read())

GPU optional (frontier eval calls HF Router, doesn't need local GPU).
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

# 5. Run frontier evals
import scripts.kaggle_eval as ke
ke.run_frontier(n_seeds=3)

print("\n=== Account 2 DONE — 5 frontier eval JSONs uploaded to Hub ===", flush=True)
