"""
Kaggle Runner Script
====================
Dùng để chạy toàn bộ pipeline training trên Kaggle thông qua Kaggle CLI.

Workflow:
  1. Clone GitHub repo về /kaggle/working/
  2. Cài thêm dependencies nếu cần
  3. Link dataset từ /kaggle/input/ vào repo
  4. Chạy prepare_data.py
  5. Chạy các script training tuỳ chọn

Cách dùng với Kaggle CLI (chạy từ thư mục project local):
  kaggle kernels push -p .
"""

import os
import sys
import subprocess

# ─── Config ───────────────────────────────────────────────────────────────────
GITHUB_REPO    = "https://github.com/huynguyentuank22/CO3133_DL.git"   # <-- đổi thành repo của bạn
REPO_DIR       = "/kaggle/working/CO3133_DL"
KAGGLE_DATASET = "/kaggle/input/clothing-reviews"                # dataset source trên Kaggle
RAW_CSV_NAME   = "Womens Clothing E-Commerce Reviews.csv"

# ─── Helper ───────────────────────────────────────────────────────────────────
def run(cmd, cwd=None):
    """Chạy shell command, raise nếu lỗi."""
    print(f"\n>>> {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed (exit {result.returncode}): {cmd}")

# ─── Step 1: Clone repo ────────────────────────────────────────────────────────
if os.path.exists(REPO_DIR):
    print(f"[INFO] Repo đã tồn tại tại {REPO_DIR}, pull latest...")
    run(f"git pull", cwd=REPO_DIR)
else:
    print(f"[INFO] Cloning repo từ {GITHUB_REPO}...")
    run(f"git clone {GITHUB_REPO} {REPO_DIR}")

# Thêm repo vào sys.path để import src.*
sys.path.insert(0, REPO_DIR)
os.chdir(REPO_DIR)
print(f"[INFO] Working directory: {os.getcwd()}")

# ─── Step 2: Cài dependencies còn thiếu ───────────────────────────────────────
# Kaggle đã có sẵn torch, transformers, sklearn, pandas, numpy, matplotlib, seaborn, tqdm
# Chỉ cài thêm những gì Kaggle chưa có
EXTRA_DEPS = ["captum", "lime", "shap", "nltk", "imbalanced-learn"]
run(f"pip install -q {' '.join(EXTRA_DEPS)}")

# ─── Step 3: Link dataset từ Kaggle input ─────────────────────────────────────
os.makedirs("data/raw", exist_ok=True)
src_csv = os.path.join(KAGGLE_DATASET, RAW_CSV_NAME)
dst_csv = os.path.join("data/raw", RAW_CSV_NAME)

if os.path.exists(src_csv):
    if not os.path.exists(dst_csv):
        import shutil
        shutil.copy(src_csv, dst_csv)
        print(f"[INFO] Copied dataset: {src_csv} -> {dst_csv}")
    else:
        print(f"[INFO] Dataset đã có tại {dst_csv}")
else:
    raise FileNotFoundError(
        f"Không tìm thấy dataset tại {src_csv}.\n"
        f"Hãy thêm dataset 'clothing-reviews' vào kernel (kernel-metadata.json -> dataset_sources)."
    )

# ─── Step 4: Prepare data (chỉ chạy nếu splits chưa có) ──────────────────────
splits_done = all(
    os.path.exists(f"data/splits/{f}")
    for f in ["train.csv", "val.csv", "test.csv"]
)
if not splits_done:
    print("\n[INFO] Chạy prepare_data.py ...")
    run("python scripts/prepare_data.py")
else:
    print("[INFO] Data splits đã tồn tại, bỏ qua prepare_data.")

# ─── Step 5: Training ─────────────────────────────────────────────────────────
# Bỏ comment script nào muốn chạy:

print("\n[INFO] Bắt đầu training ...")

# --- BiLSTM ---
run("python scripts/train_bilstm.py")

# --- BiLSTM + Attention ---
# run("python scripts/train_bilstm_attention.py")

# --- DistilBERT ---
# run("python scripts/train_distilbert.py --finetune full --epochs 4")

# --- BERT / Transformer Large ---
# run("python scripts/train_transformer_large.py --finetune full --epochs 3")

# ─── (Tuỳ chọn) Post-training analysis ───────────────────────────────────────
# run("python scripts/evaluate_all.py")
# run("python scripts/run_xai.py")
# run("python scripts/run_error_analysis.py")
# run("python scripts/run_robustness.py")
# run("python scripts/run_ensemble.py")
# run("python scripts/run_efficiency.py")

print("\n[DONE] Pipeline hoàn thành!")
