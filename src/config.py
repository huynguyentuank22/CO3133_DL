"""
Centralized configuration for the NLP Rating Classification project.
"""
import os
import torch

# ─── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
DATA_SPLITS_DIR = os.path.join(PROJECT_ROOT, "data", "splits")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
CHECKPOINT_DIR = os.path.join(OUTPUTS_DIR, "checkpoints")
FIGURES_DIR = os.path.join(OUTPUTS_DIR, "figures")
REPORTS_DIR = os.path.join(OUTPUTS_DIR, "reports")
TABLES_DIR = os.path.join(OUTPUTS_DIR, "tables")
LOGS_DIR = os.path.join(OUTPUTS_DIR, "logs")

RAW_CSV = os.path.join(DATA_RAW_DIR, "Womens Clothing E-Commerce Reviews.csv")

# ─── General ──────────────────────────────────────────────────────────────────
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 5
LABEL_MAP = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}   # Rating → index
LABEL_MAP_INV = {v: k for k, v in LABEL_MAP.items()}  # index → Rating

# ─── Preprocessing ────────────────────────────────────────────────────────────
MIN_TOKEN_LENGTH = 3            # minimum number of simple whitespace tokens
LOWERCASE_RNN = True            # whether to lowercase for RNN
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ─── RNN Hyperparameters ─────────────────────────────────────────────────────
RNN_EMBEDDING_DIM = 128
RNN_HIDDEN_DIM = 256
RNN_NUM_LAYERS = 2
RNN_DROPOUT = 0.3
RNN_BATCH_SIZE = 64
RNN_LR = 1e-3
RNN_EPOCHS = 15
RNN_MAX_LENGTH = 256
RNN_GRAD_CLIP = 5.0
RNN_MAX_VOCAB_SIZE = 30_000

# ─── Transformer Hyperparameters ─────────────────────────────────────────────
TRANSFORMER_MAX_LENGTH = 256
TRANSFORMER_BATCH_SIZE = 16
TRANSFORMER_LR = 2e-5
TRANSFORMER_EPOCHS = 4
TRANSFORMER_WEIGHT_DECAY = 0.01
TRANSFORMER_WARMUP_RATIO = 0.1

DISTILBERT_MODEL_NAME = "distilbert-base-uncased"
BERT_MODEL_NAME = "bert-base-uncased"

# ─── Fine-tune strategies ────────────────────────────────────────────────────
# "freeze": freeze backbone, train head only
# "full":   full fine-tune
# "llrd":   layer-wise learning-rate decay
FINETUNE_STRATEGY = "full"
LLRD_DECAY_FACTOR = 0.85

# ─── Imbalance strategies ────────────────────────────────────────────────────
# "weighted_ce":   use class-weighted CrossEntropyLoss on original train set
# "undersample_ce": undersample train set, use standard CE
IMBALANCE_STRATEGY = "weighted_ce"

# ─── Early stopping ──────────────────────────────────────────────────────────
EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_METRIC = "macro_f1"  # monitor val macro-F1

# ─── Augmentation ────────────────────────────────────────────────────────────
AUGMENTATION_ENABLED = False
AUGMENTATION_SYNONYM_PROB = 0.1     # probability of replacing a word
AUGMENTATION_DELETE_PROB = 0.05     # probability of deleting a word

# ─── Ensemble ────────────────────────────────────────────────────────────────
ENSEMBLE_ALPHAS = [0.3, 0.4, 0.5, 0.6, 0.7]


def ensure_dirs():
    """Create all output directories if they don't exist."""
    for d in [DATA_PROCESSED_DIR, DATA_SPLITS_DIR, CHECKPOINT_DIR,
              FIGURES_DIR, REPORTS_DIR, TABLES_DIR, LOGS_DIR]:
        os.makedirs(d, exist_ok=True)
