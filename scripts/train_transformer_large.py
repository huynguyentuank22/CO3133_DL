"""
Train BERT-base model (the "larger" Transformer).
Usage: python scripts/train_transformer_large.py [--imbalance weighted_ce|undersample_ce]
       [--finetune freeze|full|llrd] [--epochs N]
"""
import sys, os, argparse
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src import config
from src.utils import set_seed, get_logger
from src.data_utils import get_transformer_tokenizer
from src.datasets import TransformerDataset
from src.transformer_models import BertClassifier
from src.imbalance import random_undersample, get_loss_function, log_class_distribution
from src.trainer import Trainer, get_linear_warmup_scheduler
from src.visualization import plot_training_curves

logger = get_logger("train_bert",
                    os.path.join(config.LOGS_DIR, "train_bert.log"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imbalance", default="weighted_ce", choices=["weighted_ce", "undersample_ce"])
    parser.add_argument("--finetune", default="full", choices=["freeze", "full", "llrd"])
    parser.add_argument("--epochs", type=int, default=config.TRANSFORMER_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=config.TRANSFORMER_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.TRANSFORMER_LR)
    args = parser.parse_args()

    set_seed()
    config.ensure_dirs()

    train_df = pd.read_csv(os.path.join(config.DATA_SPLITS_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(config.DATA_SPLITS_DIR, "val.csv"))

    log_class_distribution(train_df["label"].tolist(), "train_original")

    if args.imbalance == "undersample_ce":
        train_df = random_undersample(train_df)
        log_class_distribution(train_df["label"].tolist(), "train_undersampled")

    tokenizer = get_transformer_tokenizer(config.BERT_MODEL_NAME)
    train_ds = TransformerDataset(train_df["full_text"].tolist(), train_df["label"].tolist(), tokenizer)
    val_ds = TransformerDataset(val_df["full_text"].tolist(), val_df["label"].tolist(), tokenizer)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = BertClassifier(finetune_strategy=args.finetune)

    if args.finetune == "llrd":
        param_groups = model.get_llrd_param_groups(base_lr=args.lr)
        optimizer = torch.optim.AdamW(param_groups, weight_decay=config.TRANSFORMER_WEIGHT_DECAY)
    else:
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr, weight_decay=config.TRANSFORMER_WEIGHT_DECAY,
        )

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * config.TRANSFORMER_WARMUP_RATIO)
    scheduler = get_linear_warmup_scheduler(optimizer, warmup_steps, total_steps)

    criterion = get_loss_function(args.imbalance, train_df["label"].tolist(), config.DEVICE)

    model_name = f"bert_{args.finetune}_{args.imbalance}"
    trainer = Trainer(
        model, optimizer, criterion,
        model_type="transformer", model_name=model_name,
        scheduler=scheduler, grad_clip=1.0,
    )

    history = trainer.train(train_loader, val_loader, epochs=args.epochs)
    plot_training_curves(history, model_name)
    logger.info(f"[OK] Training complete: {model_name}")


if __name__ == "__main__":
    main()


