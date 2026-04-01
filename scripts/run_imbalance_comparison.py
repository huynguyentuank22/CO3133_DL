"""
Compare imbalance strategies: weighted_ce vs undersample_ce.
Usage: python scripts/run_imbalance_comparison.py
"""
import sys, os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src import config
from src.utils import set_seed, get_logger, load_checkpoint
from src.data_utils import Vocabulary, get_transformer_tokenizer
from src.datasets import RNNDataset, TransformerDataset
from src.rnn_models import BiLSTMAttention
from src.transformer_models import DistilBertClassifier
from src.imbalance import (
    random_undersample, get_loss_function,
    log_class_distribution, plot_class_distributions,
)
from src.trainer import Trainer
from src.metrics import compute_metrics, plot_confusion_matrix
from src.visualization import plot_training_curves

logger = get_logger("imbalance_comparison",
                    os.path.join(config.LOGS_DIR, "imbalance_comparison.log"))


def train_and_eval_rnn(train_df, val_df, test_df, strategy, vocab, model_name):
    """Train and evaluate BiLSTM+Attention with given strategy."""
    train_ds = RNNDataset(train_df["full_text"].tolist(), train_df["label"].tolist(), vocab)
    val_ds = RNNDataset(val_df["full_text"].tolist(), val_df["label"].tolist(), vocab)
    test_ds = RNNDataset(test_df["full_text"].tolist(), test_df["label"].tolist(), vocab)

    train_loader = DataLoader(train_ds, batch_size=config.RNN_BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config.RNN_BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=config.RNN_BATCH_SIZE, shuffle=False, num_workers=0)

    model = BiLSTMAttention(vocab_size=len(vocab))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.RNN_LR)
    criterion = get_loss_function(strategy, train_df["label"].tolist(), config.DEVICE)

    trainer = Trainer(model, optimizer, criterion, model_type="rnn",
                      model_name=model_name, grad_clip=config.RNN_GRAD_CLIP)
    history = trainer.train(train_loader, val_loader, epochs=config.RNN_EPOCHS)
    plot_training_curves(history, model_name)

    # Load best checkpoint and evaluate on test
    from src.utils import load_checkpoint as load_ckpt
    best_path = os.path.join(config.CHECKPOINT_DIR, f"{model_name}_best.pt")
    if os.path.exists(best_path):
        load_ckpt(best_path, model)
    model.to(config.DEVICE)

    _, metrics, preds, labels = trainer.evaluate(test_loader)
    cm_path = os.path.join(config.FIGURES_DIR, f"{model_name}_cm.png")
    plot_confusion_matrix(labels, preds, title=model_name, save_path=cm_path)

    return {
        "model_name": "BiLSTM+Attention",
        "imbalance_strategy": strategy,
        "train_samples": len(train_df),
        **metrics,
        "train_time_sec": round(history.get("train_time_sec", 0), 2),
        "confusion_matrix_path": cm_path,
    }


def main():
    set_seed()
    config.ensure_dirs()

    train_df_orig = pd.read_csv(os.path.join(config.DATA_SPLITS_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(config.DATA_SPLITS_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(config.DATA_SPLITS_DIR, "test.csv"))

    train_df_under = random_undersample(train_df_orig)

    # Save undersampled train set
    train_df_under.to_csv(os.path.join(config.DATA_SPLITS_DIR, "train_undersampled.csv"), index=False)

    # Log distributions
    orig_dist = log_class_distribution(train_df_orig["label"].tolist(), "original_train")
    under_dist = log_class_distribution(train_df_under["label"].tolist(), "undersampled_train")
    val_dist = log_class_distribution(val_df["label"].tolist(), "validation")
    test_dist = log_class_distribution(test_df["label"].tolist(), "test")

    plot_class_distributions(
        {"Original Train": orig_dist, "Undersampled Train": under_dist,
         "Validation": val_dist, "Test": test_dist},
        title="Class Distribution Comparison",
        save_path=os.path.join(config.FIGURES_DIR, "class_distribution_comparison.png"),
    )

    # Build vocab from original train
    vocab = Vocabulary()
    vocab.build_from_texts(train_df_orig["full_text"].tolist())
    vocab_dir = os.path.join(config.DATA_PROCESSED_DIR, "vocabs")
    os.makedirs(vocab_dir, exist_ok=True)
    vocab.save(os.path.join(vocab_dir, "bilstm_attention_weighted_ce_vocab.json"))
    vocab.save(os.path.join(vocab_dir, "bilstm_attention_undersample_ce_vocab.json"))
    # Keep legacy path for backward compatibility with old scripts.
    vocab.save(os.path.join(config.DATA_PROCESSED_DIR, "vocab.json"))

    results = []

    # BiLSTM+Attention: weighted_ce
    logger.info("=" * 50 + " BiLSTM+Attention + weighted_ce " + "=" * 50)
    r1 = train_and_eval_rnn(train_df_orig, val_df, test_df, "weighted_ce", vocab,
                             "bilstm_attention_weighted_ce")
    results.append(r1)

    # BiLSTM+Attention: undersample_ce
    logger.info("=" * 50 + " BiLSTM+Attention + undersample_ce " + "=" * 50)
    r2 = train_and_eval_rnn(train_df_under, val_df, test_df, "undersample_ce", vocab,
                             "bilstm_attention_undersample_ce")
    results.append(r2)

    # Save comparison
    results_df = pd.DataFrame(results)
    save_path = os.path.join(config.TABLES_DIR, "imbalance_comparison.csv")
    results_df.to_csv(save_path, index=False)
    logger.info(f"[OK] Imbalance comparison saved to {save_path}")
    print("\n" + results_df.to_string(index=False))

    # Conclusions
    logger.info("\n=== CONCLUSIONS ===")
    if len(results) == 2:
        if results[0]["accuracy"] > results[1]["accuracy"]:
            logger.info("weighted_ce has HIGHER Accuracy")
        else:
            logger.info("undersample_ce has HIGHER Accuracy")
        if results[0]["macro_f1"] > results[1]["macro_f1"]:
            logger.info("weighted_ce has HIGHER Macro-F1")
        else:
            logger.info("undersample_ce has HIGHER Macro-F1 (likely better for rare classes)")


if __name__ == "__main__":
    main()


