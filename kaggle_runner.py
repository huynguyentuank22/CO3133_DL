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
import importlib
import time
import shutil
from itertools import product

# --- Config -------------------------------------------------------------------
GITHUB_REPO    = "https://github.com/huynguyentuank22/CO3133_DL.git"
REPO_DIR       = "/kaggle/working/CO3133_DL"
# Dataset huy281204/clothing-reviews chua san train/val/test.csv
SPLITS_INPUT   = "/kaggle/input/datasets/huy281204/clothing-reviews"
SECRET_RETRY_ATTEMPTS = 1
SECRET_RETRY_SLEEP_SEC = 2.0
W_ANDB_FALLBACK_ENV_FILES = [
    "/kaggle/input/datasets/huy281204/wandb-scerets/wandb.env",
    "/kaggle/input/wandb-secrets/.env",
    "/kaggle/input/wandb-secrets/wandb.env",
    "/kaggle/input/wandb-scerets/.env",
    "/kaggle/input/wandb-scerets/wandb.env",
    "/kaggle/input/huy281204-wandb-scerets/.env",
    "/kaggle/input/huy281204-wandb-scerets/wandb.env",
    "/kaggle/input/wandb/.env",
    "/kaggle/input/wandb/wandb.env",
]

CHECKPOINTS_INPUT_CANDIDATES = [
    "/kaggle/input/datasets/huy281204/co3133-checkpoints",
    "/kaggle/input/huy281204-co3133-checkpoints",
]

VOCABS_INPUT_CANDIDATES = [
    "/kaggle/input/datasets/huy281204/co3133-vocabs",
    "/kaggle/input/huy281204-co3133-vocabs",
]

# Che do evaluate nhanh bang checkpoints da upload len Kaggle.
# Khong xoa logic train hien co, chi them fast-path de chay evaluate_all.
RUN_EVALUATE_ALL = False
EVALUATE_ONLY = True

# Che do chay XAI voi checkpoints da train san.
RUN_XAI = False
XAI_ONLY = True

# Che do chay Error Analysis voi checkpoints da train san.
RUN_ERROR_ANALYSIS = False
ERROR_ANALYSIS_ONLY = True

# Che do chay Robustness voi checkpoints da train san.
RUN_ROBUSTNESS = True
ROBUSTNESS_ONLY = True

# Bat/tat cac nhom sweep
ENABLE_RNN_SWEEP = False
ENABLE_DISTILBERT_SWEEP = False
ENABLE_BERT_SWEEP = False

# --- Helper -------------------------------------------------------------------
def run(cmd, cwd=None):
    """Chay shell command, raise neu loi."""
    print(f"\n>>> {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed (exit {result.returncode}): {cmd}")


def _get_secret_value(name, default=None):
    """Lay secret tu env truoc, sau do den Kaggle Secrets."""
    value = os.getenv(name)
    if value:
        print(f"[INFO] Secret '{name}' duoc lay tu environment variable.")
        return value

    last_error = None
    for attempt in range(1, SECRET_RETRY_ATTEMPTS + 1):
        try:
            kaggle_secrets = importlib.import_module("kaggle_secrets")
            client = kaggle_secrets.UserSecretsClient()
            value = client.get_secret(name)
            if value:
                print(f"[INFO] Secret '{name}' duoc lay tu Kaggle Secrets (attempt {attempt}).")
                return value
        except Exception as e:
            last_error = e
            print(
                f"[WARN] Khong doc duoc secret '{name}' (attempt {attempt}/{SECRET_RETRY_ATTEMPTS}): "
                f"{type(e).__name__}: {e}"
            )
            if attempt < SECRET_RETRY_ATTEMPTS:
                time.sleep(SECRET_RETRY_SLEEP_SEC * attempt)

    if last_error is not None:
        print(
            f"[WARN] Bo qua secret '{name}' sau {SECRET_RETRY_ATTEMPTS} lan thu. "
            "Se dung default/fallback neu co."
        )
    return default


def _mask_secret(value):
    if not value:
        return "<empty>"
    if len(value) <= 10:
        return "*" * len(value)
    return f"{value[:6]}...{value[-4:]}"


def _find_splits_input_path():
    """Tu dong tim duong dan dataset splits tren Kaggle."""
    candidates = [
        SPLITS_INPUT,
        "/kaggle/input/clothing-reviews",
        "/kaggle/input/huy281204-clothing-reviews",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return SPLITS_INPUT


def _find_checkpoints_input_path():
    """Tu dong tim dataset input chua checkpoints (.pt)."""
    for path in CHECKPOINTS_INPUT_CANDIDATES:
        if os.path.exists(path):
            return path

    input_root = "/kaggle/input"
    if not os.path.exists(input_root):
        return None

    for entry in os.listdir(input_root):
        candidate_dir = os.path.join(input_root, entry)
        if not os.path.isdir(candidate_dir):
            continue
        try:
            if any(name.endswith("_best.pt") for name in os.listdir(candidate_dir)):
                return candidate_dir
        except OSError:
            continue

    return None


def _prepare_checkpoints_for_evaluation(repo_dir):
    """Dam bao outputs/checkpoints co day du file _best.pt de evaluate_all doc duoc."""
    target_dir = os.path.join(repo_dir, "outputs", "checkpoints")
    os.makedirs(target_dir, exist_ok=True)

    existing = [f for f in os.listdir(target_dir) if f.endswith("_best.pt")]
    if existing:
        print(f"[INFO] Tim thay {len(existing)} checkpoints local trong {target_dir}.")
        return

    source_dir = _find_checkpoints_input_path()
    if not source_dir:
        raise FileNotFoundError(
            "Khong tim thay dataset input chua checkpoints. "
            "Hay them dataset source 'huy281204/co3133-checkpoints' vao kernel metadata."
        )

    copied = 0
    for name in os.listdir(source_dir):
        if not name.endswith("_best.pt"):
            continue
        src = os.path.join(source_dir, name)
        dst = os.path.join(target_dir, name)
        try:
            os.symlink(src, dst)
        except Exception:
            shutil.copy2(src, dst)
        copied += 1

    if copied == 0:
        raise FileNotFoundError(
            f"Khong tim thay file *_best.pt trong {source_dir}."
        )

    print(f"[INFO] Da nap {copied} checkpoints vao {target_dir} tu {source_dir}.")


def _find_vocabs_input_path():
    """Tu dong tim dataset input chua cac file *_vocab.json."""
    for path in VOCABS_INPUT_CANDIDATES:
        if os.path.exists(path):
            return path

    input_root = "/kaggle/input"
    if not os.path.exists(input_root):
        return None

    for entry in os.listdir(input_root):
        candidate_dir = os.path.join(input_root, entry)
        if not os.path.isdir(candidate_dir):
            continue
        try:
            if any(name.endswith("_vocab.json") for name in os.listdir(candidate_dir)):
                return candidate_dir
        except OSError:
            continue

    return None


def _prepare_vocabs_for_evaluation(repo_dir):
    """Dam bao data/processed/vocabs co day du vocab cho RNN models."""
    target_dir = os.path.join(repo_dir, "data", "processed", "vocabs")
    os.makedirs(target_dir, exist_ok=True)

    existing = [f for f in os.listdir(target_dir) if f.endswith("_vocab.json")]
    if existing:
        print(f"[INFO] Tim thay {len(existing)} vocab files local trong {target_dir}.")
        return

    source_dir = _find_vocabs_input_path()
    if not source_dir:
        raise FileNotFoundError(
            "Khong tim thay dataset input chua vocabs. "
            "Hay them dataset source 'huy281204/co3133-vocabs' vao kernel metadata."
        )

    copied = 0
    for name in os.listdir(source_dir):
        if not name.endswith("_vocab.json"):
            continue
        src = os.path.join(source_dir, name)
        dst = os.path.join(target_dir, name)
        try:
            os.symlink(src, dst)
        except Exception:
            shutil.copy2(src, dst)
        copied += 1

    if copied == 0:
        raise FileNotFoundError(
            f"Khong tim thay file *_vocab.json trong {source_dir}."
        )

    print(f"[INFO] Da nap {copied} vocab files vao {target_dir} tu {source_dir}.")


def _read_env_file(path):
    """Doc file .env don gian va tra ve dict key/value."""
    data = {}
    if not os.path.exists(path):
        return data

    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key:
                continue
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"\"", "'"}:
                value = value[1:-1]
            data[key] = value
    return data


def _load_wandb_from_fallback_files():
    """Thu lay W&B key tu file .env trong /kaggle/input neu Secrets service loi."""
    for env_path in W_ANDB_FALLBACK_ENV_FILES:
        env_data = _read_env_file(env_path)
        if not env_data:
            continue

        api_key = env_data.get("WANDB_API_KEY") or env_data.get("WANDB") or env_data.get("wandb_api_key")
        if api_key:
            print(f"[INFO] Lay W&B key tu fallback file: {env_path}")
            return {
                "api_key": api_key,
                "project": env_data.get("WANDB_PROJECT") or "co3133_dl",
                "entity": env_data.get("WANDB_ENTITY"),
            }
    return None


def setup_wandb_env(repo_dir):
    """Tao file .env runtime de cac train script tu bat W&B."""
    # Ho tro ca WANDB_API_KEY (chuan) va WANDB (legacy)
    wandb_api_key = (
        _get_secret_value("WANDB_API_KEY")
        or _get_secret_value("WANDB")
        or _get_secret_value("wandb_api_key")
    )
    wandb_project = _get_secret_value("WANDB_PROJECT", "co3133_dl")
    wandb_entity = _get_secret_value("WANDB_ENTITY")

    if not wandb_api_key:
        fallback = _load_wandb_from_fallback_files()
        if fallback:
            wandb_api_key = fallback["api_key"]
            wandb_project = fallback["project"]
            wandb_entity = fallback["entity"]

    if not wandb_api_key:
        print("[WARN] Khong tim thay WANDB_API_KEY/WANDB. Se train khong log len W&B.")
        print("[HINT] Tren Kaggle, can tick checkbox ben trai secret de cap quyen cho kernel hien tai.")
        print("[HINT] Sau khi tick, bam 'Save Version' roi chay lai.")
        print("[HINT] Neu Secrets service van loi, tao private dataset chua file .env va mount vao /kaggle/input/wandb-secrets/.env")
        return {
            "enabled": False,
            "project": wandb_project,
            "entity": wandb_entity,
        }

    # Set vao env hien tai de dung ngay trong process
    os.environ["WANDB_API_KEY"] = wandb_api_key
    os.environ["WANDB_PROJECT"] = wandb_project
    if wandb_entity:
        os.environ["WANDB_ENTITY"] = wandb_entity

    # Tao .env trong repo de cac script co the auto-doc
    dotenv_path = os.path.join(repo_dir, ".env")
    lines = [
        f"WANDB_API_KEY={wandb_api_key}",
        f"WANDB_PROJECT={wandb_project}",
    ]
    if wandb_entity:
        lines.append(f"WANDB_ENTITY={wandb_entity}")

    with open(dotenv_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[INFO] Da tao .env runtime tai: {dotenv_path}")
    print(f"[INFO] W&B key: {_mask_secret(wandb_api_key)}")
    print(f"[INFO] W&B project/entity: {wandb_project}/{wandb_entity}")

    return {
        "enabled": True,
        "project": wandb_project,
        "entity": wandb_entity,
    }

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
run(
    "python -m pip uninstall -y torch torchvision torchaudio && "
    "python -m pip install --no-cache-dir "
    "torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 "
    "--index-url https://download.pytorch.org/whl/cu126"
)

# --- Step 3: Symlink data/splits -> /kaggle/input/clothing-reviews -----------
# Khong can copy - doc thang tu Kaggle input
resolved_splits_input = _find_splits_input_path()
splits_link = "data/splits"
if not os.path.exists(splits_link):
    os.symlink(resolved_splits_input, splits_link)
    print(f"[INFO] Symlinked: {splits_link} -> {resolved_splits_input}")
else:
    print(f"[INFO] data/splits da ton tai")

# --- Step 3b: Nap checkpoints/vocabs tu Kaggle input (phuc vu evaluate/XAI/error analysis/robustness) --
if RUN_EVALUATE_ALL or RUN_XAI or RUN_ERROR_ANALYSIS or RUN_ROBUSTNESS:
    _prepare_checkpoints_for_evaluation(REPO_DIR)
    _prepare_vocabs_for_evaluation(REPO_DIR)

# --- Step 4: Setup W&B .env tu Kaggle Secrets/env ----------------------------
# Tren Kaggle, them secrets: WANDB_API_KEY, WANDB_PROJECT, WANDB_ENTITY
wandb_cfg = setup_wandb_env(REPO_DIR)
wandb_args = ""
if wandb_cfg.get("enabled"):
    wandb_args = f" --wandb --wandb_project {wandb_cfg['project']}"
    if wandb_cfg.get("entity"):
        wandb_args += f" --wandb_entity {wandb_cfg['entity']}"
    print(f"[INFO] W&B da duoc BAT. wandb_args = {wandb_args}")
else:
    print("[INFO] W&B dang TAT cho cac run train.")

if RUN_EVALUATE_ALL and EVALUATE_ONLY:
    print("\n[INFO] Chay evaluate_all voi checkpoints co san tren Kaggle input ...")
    run("python scripts/evaluate_all.py")
    print("\n[DONE] Evaluate_all hoan thanh!")
    sys.exit(0)

if RUN_XAI and XAI_ONLY:
    print("\n[INFO] Chay run_xai voi checkpoints co san tren Kaggle input ...")
    run("python scripts/run_xai.py")
    print("\n[DONE] run_xai hoan thanh!")
    sys.exit(0)

if RUN_ERROR_ANALYSIS and ERROR_ANALYSIS_ONLY:
    print("\n[INFO] Chay run_error_analysis voi checkpoints co san tren Kaggle input ...")
    run("python scripts/run_error_analysis.py")
    print("\n[DONE] run_error_analysis hoan thanh!")
    sys.exit(0)

if RUN_ROBUSTNESS and ROBUSTNESS_ONLY:
    print("\n[INFO] Chay run_robustness voi checkpoints co san tren Kaggle input ...")
    run("python scripts/run_robustness.py")
    print("\n[DONE] run_robustness hoan thanh!")
    sys.exit(0)

# --- Step 5: Training ---------------------------------------------------------
# Chay sweep hyperparameter de tim best config

print("\n[INFO] Bat dau training ...")


def _fmt_lr(lr):
    return str(lr).replace(".", "p")


def _run_train(script_cmd, run_name):
    cmd = script_cmd
    if wandb_cfg.get("enabled"):
        cmd += f"{wandb_args} --wandb_run_name {run_name}"
    run(cmd)


def _build_transformer_sweep(finetunes, imbalances, lrs, batch_sizes, epochs_list):
    """Tao full-grid sweep (Cartesian product) cho transformer models."""
    return [
        {
            "finetune": finetune,
            "imbalance": imbalance,
            "lr": lr,
            "batch_size": batch_size,
            "epochs": epochs,
        }
        for finetune, imbalance, lr, batch_size, epochs in product(
            finetunes, imbalances, lrs, batch_sizes, epochs_list
        )
    ]


if ENABLE_RNN_SWEEP:
    print("[INFO] Chay RNN hyperparameter sweep ...")

    bilstm_sweep = [
        {"imbalance": "weighted_ce", "lr": 1e-3, "batch_size": 64, "epochs": 15},
        {"imbalance": "weighted_ce", "lr": 5e-4, "batch_size": 64, "epochs": 20},
        {"imbalance": "weighted_ce", "lr": 1e-3, "batch_size": 32, "epochs": 20},
        {"imbalance": "undersample_ce", "lr": 1e-3, "batch_size": 64, "epochs": 15},
        {"imbalance": "undersample_ce", "lr": 5e-4, "batch_size": 64, "epochs": 20},
        {"imbalance": "undersample_ce", "lr": 1e-3, "batch_size": 32, "epochs": 20},
    ]

    for cfg in bilstm_sweep:
        run_name = (
            f"bilstm_{cfg['imbalance']}_lr{_fmt_lr(cfg['lr'])}"
            f"_bs{cfg['batch_size']}_ep{cfg['epochs']}"
        )
        base_cmd = (
            "python scripts/train_bilstm.py "
            f"--imbalance {cfg['imbalance']} --lr {cfg['lr']} "
            f"--batch_size {cfg['batch_size']} --epochs {cfg['epochs']} "
        )
        _run_train(base_cmd, run_name)

    bilstm_attn_sweep = [
        {"imbalance": "weighted_ce", "lr": 1e-3, "batch_size": 64, "epochs": 15},
        {"imbalance": "weighted_ce", "lr": 5e-4, "batch_size": 64, "epochs": 20},
        {"imbalance": "weighted_ce", "lr": 1e-3, "batch_size": 32, "epochs": 20},
        {"imbalance": "undersample_ce", "lr": 1e-3, "batch_size": 64, "epochs": 15},
        {"imbalance": "undersample_ce", "lr": 5e-4, "batch_size": 64, "epochs": 20},
        {"imbalance": "undersample_ce", "lr": 1e-3, "batch_size": 32, "epochs": 20},
    ]

    for cfg in bilstm_attn_sweep:
        run_name = (
            f"bilstm_attn_{cfg['imbalance']}_lr{_fmt_lr(cfg['lr'])}"
            f"_bs{cfg['batch_size']}_ep{cfg['epochs']}"
        )
        base_cmd = (
            "python scripts/train_bilstm_attention.py "
            f"--imbalance {cfg['imbalance']} --lr {cfg['lr']} "
            f"--batch_size {cfg['batch_size']} --epochs {cfg['epochs']} "
        )
        _run_train(base_cmd, run_name)


if ENABLE_DISTILBERT_SWEEP:
    print("[INFO] Chay DistilBERT hyperparameter sweep ...")

    distilbert_sweep = _build_transformer_sweep(
        finetunes=["full", "llrd"],
        imbalances=["weighted_ce", "undersample_ce"],
        lrs=[3e-5],
        batch_sizes=[16, 32],
        epochs_list=[4],
    )
    print(f"[INFO] DistilBERT sweep runs: {len(distilbert_sweep)}")

    for cfg in distilbert_sweep:
        run_name = (
            f"distilbert_{cfg['finetune']}_{cfg['imbalance']}_lr{_fmt_lr(cfg['lr'])}"
            f"_bs{cfg['batch_size']}_ep{cfg['epochs']}"
        )
        base_cmd = (
            "python scripts/train_distilbert.py "
            f"--finetune {cfg['finetune']} --imbalance {cfg['imbalance']} --lr {cfg['lr']} "
            f"--batch_size {cfg['batch_size']} --epochs {cfg['epochs']} "
        )
        _run_train(base_cmd, run_name)


if ENABLE_BERT_SWEEP:
    print("[INFO] Chay BERT-base hyperparameter sweep ...")

    bert_sweep = _build_transformer_sweep(
        finetunes=["full", "llrd"],
        imbalances=["weighted_ce", "undersample_ce"],
        lrs=[3e-5],
        batch_sizes=[8, 16],
        epochs_list=[4],
    )
    print(f"[INFO] BERT-base sweep runs: {len(bert_sweep)}")

    for cfg in bert_sweep:
        run_name = (
            f"bert_{cfg['finetune']}_{cfg['imbalance']}_lr{_fmt_lr(cfg['lr'])}"
            f"_bs{cfg['batch_size']}_ep{cfg['epochs']}"
        )
        base_cmd = (
            "python scripts/train_transformer_large.py "
            f"--finetune {cfg['finetune']} --imbalance {cfg['imbalance']} --lr {cfg['lr']} "
            f"--batch_size {cfg['batch_size']} --epochs {cfg['epochs']} "
        )
        _run_train(base_cmd, run_name)

if RUN_EVALUATE_ALL and not EVALUATE_ONLY:
    run("python scripts/evaluate_all.py")

if RUN_XAI and not XAI_ONLY:
    run("python scripts/run_xai.py")

if RUN_ERROR_ANALYSIS and not ERROR_ANALYSIS_ONLY:
    run("python scripts/run_error_analysis.py")

if RUN_ROBUSTNESS and not ROBUSTNESS_ONLY:
    run("python scripts/run_robustness.py")

# --- (Tuy chon) Post-training analysis ----------------------------------------
# run("python scripts/train_bilstm.py --imbalance undersample_ce --lr 1e-3 --batch_size 64 --epochs 15")
# run("python scripts/train_bilstm.py --imbalance weighted_ce --lr 1e-3 --batch_size 32 --epochs 20")
# run("python scripts/train_bilstm_attention.py --imbalance undersample_ce --lr 1e-3 --batch_size 32 --epochs 20")
# run("python scripts/train_bilstm_attention.py --imbalance weighted_ce --lr 1e-3 --batch_size 32 --epochs 20")

# run("python scripts/train_distilbert.py --finetune full --imbalance weighted_ce --lr 1e-5 --batch_size 16 --epochs 5")
# run("python scripts/train_distilbert.py --finetune full --imbalance undersample_ce --lr 2e-5 --batch_size 16 --epochs 4")
# run("python scripts/train_distilbert.py --finetune llrd --imbalance weighted_ce --lr 3e-5 --batch_size 16 --epochs 4")
# run("python scripts/train_distilbert.py --finetune llrd --imbalance undersample_ce --lr 3e-5 --batch_size 16 --epochs 4")
# run("python scripts/train_distilbert.py --finetune freeze --imbalance weighted_ce --lr 2e-5 --batch_size 16 --epochs 4")
# run("python scripts/train_distilbert.py --finetune freeze --imbalance undersample_ce --lr 2e-5 --batch_size 16 --epochs 4")

# run("python scripts/train_transformer_large.py --finetune full --imbalance weighted_ce --lr 3e-5 --batch_size 16 --epochs 4")
# run("python scripts/train_transformer_large.py --finetune full --imbalance undersample_ce --lr 3e-5 --batch_size 16 --epochs 4")
# run("python scripts/train_transformer_large.py --finetune llrd --imbalance weighted_ce --lr 2e-5 --batch_size 16 --epochs 4")
# run("python scripts/train_transformer_large.py --finetune llrd --imbalance undersample_ce --lr 2e-5 --batch_size 16 --epochs 4")
# run("python scripts/train_transformer_large.py --finetune freeze --imbalance weighted_ce --lr 2e-5 --batch_size 16 --epochs 4")
# run("python scripts/train_transformer_large.py --finetune freeze --imbalance undersample_ce --lr 2e-5 --batch_size 16 --epochs 4")

# run("python scripts/evaluate_all.py")
# run("python scripts/run_xai.py")
# run("python scripts/run_error_analysis.py")
# run("python scripts/run_robustness.py")
# run("python scripts/run_ensemble.py")
# run("python scripts/run_efficiency.py")

print("\n[DONE] Pipeline hoan thanh!")
