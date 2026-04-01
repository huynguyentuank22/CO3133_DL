"""
Train BiLSTM + Attention model.
Usage: python scripts/train_bilstm_attention.py [--imbalance weighted_ce|undersample_ce] [--epochs N]
"""
import sys, os, argparse, importlib
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src import config
from src.utils import set_seed, get_logger, resolve_wandb_config
from src.data_utils import Vocabulary
from src.datasets import RNNDataset
from src.rnn_models import BiLSTMAttention
from src.imbalance import random_undersample, get_loss_function, log_class_distribution
from src.trainer import Trainer
from src.visualization import plot_training_curves

logger = get_logger("train_bilstm_attention",
                    os.path.join(config.LOGS_DIR, "train_bilstm_attention.log"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imbalance", default="weighted_ce", choices=["weighted_ce", "undersample_ce"])
    parser.add_argument("--epochs", type=int, default=config.RNN_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=config.RNN_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.RNN_LR)
    parser.add_argument("--wandb", action="store_true", help="Force enable W&B logging")
    parser.add_argument("--wandb_project", default="co3133_dl", help="W&B project name")
    parser.add_argument("--wandb_entity", default=None, help="W&B entity/team (optional)")
    parser.add_argument("--wandb_run_name", default=None, help="Custom W&B run name (optional)")
    args = parser.parse_args()

    wandb_cfg = resolve_wandb_config(
        enable_flag=args.wandb,
        project=args.wandb_project,
        entity=args.wandb_entity,
    )
    wandb_enabled = bool(wandb_cfg["enabled"])

    set_seed()
    config.ensure_dirs()

    train_df = pd.read_csv(os.path.join(config.DATA_SPLITS_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(config.DATA_SPLITS_DIR, "val.csv"))

    log_class_distribution(train_df["label"].tolist(), "train_original")

    if args.imbalance == "undersample_ce":
        train_df = random_undersample(train_df)
        log_class_distribution(train_df["label"].tolist(), "train_undersampled")

    model_name = f"bilstm_attention_{args.imbalance}"
    vocab_dir = os.path.join(config.DATA_PROCESSED_DIR, "vocabs")
    os.makedirs(vocab_dir, exist_ok=True)
    vocab_path = os.path.join(vocab_dir, f"{model_name}_vocab.json")

    # Build/load vocabulary
    vocab = Vocabulary()
    if os.path.exists(vocab_path):
        vocab.load(vocab_path)
        logger.info(f"Vocabulary loaded: {vocab_path} ({len(vocab)} tokens)")
    else:
        vocab.build_from_texts(train_df["full_text"].tolist())
        vocab.save(vocab_path)
        logger.info(f"Vocabulary saved: {vocab_path} ({len(vocab)} tokens)")

    train_ds = RNNDataset(train_df["full_text"].tolist(), train_df["label"].tolist(), vocab)
    val_ds = RNNDataset(val_df["full_text"].tolist(), val_df["label"].tolist(), vocab)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = BiLSTMAttention(vocab_size=len(vocab))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = get_loss_function(args.imbalance, train_df["label"].tolist(), config.DEVICE)

    wandb_run = None
    if wandb_enabled:
        try:
            wandb = importlib.import_module("wandb")
        except ImportError as e:
            raise ImportError("wandb is not installed. Run: pip install wandb") from e

        api_key = wandb_cfg.get("api_key")
        if isinstance(api_key, str) and api_key:
            wandb.login(key=api_key, relogin=True)

        run_name = args.wandb_run_name or model_name
        wandb_run = wandb.init(
            project=wandb_cfg["project"],
            entity=wandb_cfg["entity"],
            name=run_name,
            tags=["rnn", "bilstm_attention", args.imbalance],
            config={
                "model_name": model_name,
                "model_type": "rnn",
                "architecture": "BiLSTM+Attention",
                "imbalance_strategy": args.imbalance,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "seed": config.SEED,
                "train_samples": len(train_df),
                "val_samples": len(val_df),
                "vocab_size": len(vocab),
            },
        )

    trainer = Trainer(
        model, optimizer, criterion,
        model_type="rnn", model_name=model_name,
        grad_clip=config.RNN_GRAD_CLIP,
        use_wandb=wandb_enabled, wandb_run=wandb_run,
    )

    try:
        history = trainer.train(train_loader, val_loader, epochs=args.epochs)
        plot_training_curves(history, model_name)
        logger.info(f"[OK] Training complete: {model_name}")
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()


