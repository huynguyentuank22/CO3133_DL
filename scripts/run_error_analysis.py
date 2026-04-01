"""
Run error analysis on test set predictions.
Usage: python scripts/run_error_analysis.py
"""
import sys, os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from src import config
from src.utils import set_seed, get_logger, load_checkpoint
from src.data_utils import get_transformer_tokenizer, load_rnn_vocab
from src.datasets import RNNDataset, TransformerDataset
from src.rnn_models import BiLSTMAttention
from src.transformer_models import DistilBertClassifier
from src.trainer import Trainer
from src.error_analysis import (
    get_top_confusion_pairs, extract_misclassified,
    categorize_errors, compare_errors_by_strategy, generate_error_report,
)

logger = get_logger("error_analysis", os.path.join(config.LOGS_DIR, "error_analysis.log"))


def get_predictions(model, dataloader, model_type):
    """Get predictions from a model."""
    trainer = Trainer(model, None, torch.nn.CrossEntropyLoss(),
                      model_type=model_type, model_name="tmp")
    _, _, preds, labels = trainer.evaluate(dataloader)
    return np.array(preds), np.array(labels)


def main():
    set_seed()
    config.ensure_dirs()

    test_df = pd.read_csv(os.path.join(config.DATA_SPLITS_DIR, "test.csv"))
    report_dir = os.path.join(config.REPORTS_DIR, "error_analysis")
    os.makedirs(report_dir, exist_ok=True)

    results = {}

    # â”€â”€â”€ BiLSTM+Attention Analysis â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    for strategy in ["weighted_ce", "undersample_ce"]:
        ckpt_name = f"bilstm_attention_{strategy}"
        ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"{ckpt_name}_best.pt")
        if not os.path.exists(ckpt_path):
            logger.warning(f"Checkpoint not found: {ckpt_path}")
            continue

        vocab, _ = load_rnn_vocab(ckpt_name)
        if vocab is None:
            logger.warning(f"Skipping {ckpt_name}: no matching vocabulary")
            continue

        model = BiLSTMAttention(vocab_size=len(vocab))
        try:
            load_checkpoint(ckpt_path, model)
        except RuntimeError as e:
            logger.error(f"Failed to load {ckpt_name} with current vocab ({len(vocab)} tokens): {e}")
            continue
        model.to(config.DEVICE)

        test_ds = RNNDataset(test_df["full_text"].tolist(), test_df["label"].tolist(), vocab)
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

        preds, labels = get_predictions(model, test_loader, "rnn")
        results[f"bilstm_attn_{strategy}"] = {"y_true": labels, "y_pred": preds}

        # Confusion pairs
        pairs = get_top_confusion_pairs(labels, preds, top_k=10)
        logger.info(f"\nTop confusions ({ckpt_name}):")
        for p in pairs[:5]:
            logger.info(f"  {p['true_label']} â†’ {p['pred_label']}: {p['count']}")

        # Misclassified samples
        mis = extract_misclassified(test_df, labels, preds, n=50)
        mis_cat = categorize_errors(mis)
        mis_cat.to_csv(os.path.join(report_dir, f"{ckpt_name}_misclassified.csv"), index=False)

    # â”€â”€â”€ DistilBERT Analysis â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    tokenizer = get_transformer_tokenizer(config.DISTILBERT_MODEL_NAME)
    for strategy in ["weighted_ce", "undersample_ce"]:
        ckpt_name = f"distilbert_full_{strategy}"
        ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"{ckpt_name}_best.pt")
        if not os.path.exists(ckpt_path):
            logger.warning(f"Checkpoint not found: {ckpt_path}")
            continue

        model = DistilBertClassifier()
        load_checkpoint(ckpt_path, model)
        model.to(config.DEVICE)

        test_ds = TransformerDataset(test_df["full_text"].tolist(), test_df["label"].tolist(), tokenizer)
        test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

        preds, labels = get_predictions(model, test_loader, "transformer")
        results[f"distilbert_{strategy}"] = {"y_true": labels, "y_pred": preds}

        pairs = get_top_confusion_pairs(labels, preds, top_k=10)
        mis = extract_misclassified(test_df, labels, preds, n=50)
        mis_cat = categorize_errors(mis)
        mis_cat.to_csv(os.path.join(report_dir, f"{ckpt_name}_misclassified.csv"), index=False)

    # â”€â”€â”€ Cross-strategy comparison â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    comparison_df = None
    if ("bilstm_attn_weighted_ce" in results and "bilstm_attn_undersample_ce" in results):
        comparison_df = compare_errors_by_strategy(
            results["bilstm_attn_weighted_ce"],
            results["bilstm_attn_undersample_ce"],
            test_df,
        )
        comparison_df.to_csv(os.path.join(report_dir, "error_comparison_bilstm_attn.csv"), index=False)

    # Generate report
    if results:
        first_key = list(results.keys())[0]
        pairs = get_top_confusion_pairs(results[first_key]["y_true"], results[first_key]["y_pred"])
        mis = extract_misclassified(test_df, results[first_key]["y_true"], results[first_key]["y_pred"])
        mis_cat = categorize_errors(mis)
        report = generate_error_report(pairs, mis_cat, comparison_df)
        report_path = os.path.join(report_dir, "error_analysis_report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        logger.info(f"[OK] Error analysis report saved to {report_path}")

    logger.info("[OK] Error analysis complete")


if __name__ == "__main__":
    main()


