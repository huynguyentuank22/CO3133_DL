"""
Kaggle Runner Script
====================
Dung de chay toan bo pipeline training tren Kaggle thong qua Kaggle CLI.

Workflow:
  1. Clone GitHub repo ve /kaggle/working/
  2. Cai them dependencies neu can
  3. Link dataset tu /kaggle/input/ vao repo
  4. Chay prepare_data.py
  5. Chay cac script training tuy chon

Cach dung voi Kaggle CLI (chay tu thu muc project local):
  kaggle kernels push -p .
"""

import os
import sys
import subprocess

# --- Config -------------------------------------------------------------------
GITHUB_REPO    = "https://github.com/huynguyentuank22/CO3133_DL.git"
REPO_DIR       = "/kaggle/working/CO3133_DL"
# Dataset huy281204/clothing-reviews chua san train/val/test.csv
SPLITS_INPUT   = "/kaggle/input/clothing-reviews"

# --- Helper -------------------------------------------------------------------
def run(cmd, cwd=None):
    """Chay shell command, raise neu loi."""
    print(f"\n>>> {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed (exit {result.returncode}): {cmd}")

# --- Step 1: Clone repo -------------------------------------------------------
if os.path.exists(REPO_DIR):
    print(f"[INFO] Repo da ton tai tai {REPO_DIR}, pull latest...")
    run("git pull", cwd=REPO_DIR)
else:
    print(f"[INFO] Cloning repo tu {GITHUB_REPO}...")
    run(f"git clone {GITHUB_REPO} {REPO_DIR}")

# Them repo vao sys.path de import src.*
sys.path.insert(0, REPO_DIR)
os.chdir(REPO_DIR)
print(f"[INFO] Working directory: {os.getcwd()}")

# --- Step 2: Cai dependencies con thieu --------------------------------------
# Kaggle da co san torch, transformers, sklearn, pandas, numpy, matplotlib, seaborn, tqdm
# Chi cai them nhung gi Kaggle chua co
EXTRA_DEPS = ["captum", "lime", "shap", "nltk", "imbalanced-learn"]
run(f"pip install -q {' '.join(EXTRA_DEPS)}")

# --- Step 3: Symlink data/splits -> /kaggle/input/clothing-reviews -----------
# Khong can copy - doc thang tu Kaggle input
splits_link = "data/splits"
if not os.path.exists(splits_link):
    os.symlink(SPLITS_INPUT, splits_link)
    print(f"[INFO] Symlinked: {splits_link} -> {SPLITS_INPUT}")
else:
    print(f"[INFO] data/splits da ton tai")

# --- Step 5: Training ---------------------------------------------------------
# Bo comment script nao muon chay:

print("\n[INFO] Bat dau training ...")

# BiLSTM
run("python scripts/train_bilstm.py --epochs 1")

# BiLSTM + Attention
# run("python scripts/train_bilstm_attention.py")

# DistilBERT
# run("python scripts/train_distilbert.py --finetune full --epochs 4")

# BERT / Transformer Large
# run("python scripts/train_transformer_large.py --finetune full --epochs 3")

# --- (Tuy chon) Post-training analysis ----------------------------------------
# run("python scripts/evaluate_all.py")
# run("python scripts/run_xai.py")
# run("python scripts/run_error_analysis.py")
# run("python scripts/run_robustness.py")
# run("python scripts/run_ensemble.py")
# run("python scripts/run_efficiency.py")

print("\n[DONE] Pipeline hoan thanh!")
