"""
Run augmentation experiment: compare with/without augmentation.
Usage: python scripts/run_augmentation.py
"""
import sys, os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src import config
from src.utils import set_seed, get_logger
from src.data_utils import Vocabulary
from src.datasets import RNNDataset
from src.rnn_models import BiLSTMAttention
from src.imbalance import get_loss_function, log_class_distribution
from src.trainer import Trainer
from src.augmentation import augment_dataframe
from src.visualization import plot_training_curves

logger = get_logger("augmentation", os.path.join(config.LOGS_DIR, "augmentation.log"))


def train_with_setting(train_df, val_df, model_name, epochs=10):
    """Train BiLSTM+Attention with given training data."""
    vocab = Vocabulary()
    vocab.build_from_texts(train_df["full_text"].tolist())

    train_ds = RNNDataset(train_df["full_text"].tolist(), train_df["label"].tolist(), vocab)
    val_ds = RNNDataset(val_df["full_text"].tolist(), val_df["label"].tolist(), vocab)
    train_loader = DataLoader(train_ds, batch_size=config.RNN_BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config.RNN_BATCH_SIZE, shuffle=False, num_workers=0)

    model = BiLSTMAttention(vocab_size=len(vocab))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.RNN_LR)
    criterion = get_loss_function("weighted_ce", train_df["label"].tolist(), config.DEVICE)

    trainer = Trainer(model, optimizer, criterion, model_type="rnn",
                      model_name=model_name, grad_clip=config.RNN_GRAD_CLIP)
    history = trainer.train(train_loader, val_loader, epochs=epochs)
    plot_training_curves(history, model_name)
    return history


def main():
    set_seed()
    config.ensure_dirs()

    train_df = pd.read_csv(os.path.join(config.DATA_SPLITS_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(config.DATA_SPLITS_DIR, "val.csv"))

    results = []

    # Without augmentation
    logger.info("Training WITHOUT augmentation")
    h1 = train_with_setting(train_df, val_df, "bilstm_attn_no_aug", epochs=10)
    results.append({
        "setting": "no_augmentation",
        "train_samples": len(train_df),
        "best_val_macro_f1": max(h1["val_macro_f1"]),
        "best_val_accuracy": max(h1["val_accuracy"]),
    })

    # With augmentation
    logger.info("Training WITH augmentation (synonym + deletion)")
    aug_train_df = augment_dataframe(train_df, methods=["synonym", "deletion"])
    h2 = train_with_setting(aug_train_df, val_df, "bilstm_attn_with_aug", epochs=10)
    results.append({
        "setting": "with_augmentation",
        "train_samples": len(aug_train_df),
        "best_val_macro_f1": max(h2["val_macro_f1"]),
        "best_val_accuracy": max(h2["val_accuracy"]),
    })

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(config.TABLES_DIR, "augmentation_results.csv"), index=False)
    print("\n" + results_df.to_string(index=False))
    logger.info("[OK] Augmentation experiment complete")


if __name__ == "__main__":
    main()


