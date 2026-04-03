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
from collections import Counter
from src import config
from src.utils import set_seed, get_logger, load_checkpoint
from src.data_utils import get_transformer_tokenizer, load_rnn_vocab
from src.datasets import RNNDataset, TransformerDataset
from src.rnn_models import BiLSTM, BiLSTMAttention
from src.transformer_models import DistilBertClassifier, BertClassifier
from src.trainer import Trainer
from src.error_analysis import (
    get_top_confusion_pairs, extract_misclassified,
    categorize_errors,
)

logger = get_logger("error_analysis", os.path.join(config.LOGS_DIR, "error_analysis.log"))

# Best checkpoints by model family (from current evaluation summary).
BEST_MODELS = {
    "bilstm": "bilstm_weighted_ce",
    "bilstm_attn": "bilstm_attention_weighted_ce",
    "distilbert": "distilbert_full_weighted_ce",
    "bert": "bert_llrd_weighted_ce",
}


def get_predictions(model, dataloader, model_type):
    """Get predictions from a model."""
    trainer = Trainer(model, None, torch.nn.CrossEntropyLoss(),
                      model_type=model_type, model_name="tmp")
    _, _, preds, labels = trainer.evaluate(dataloader)
    return np.array(preds), np.array(labels)


def analyze_rnn_model(test_df, report_dir, ckpt_name, model_cls, result_key):
    """Run error analysis for a single RNN checkpoint."""
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"{ckpt_name}_best.pt")
    if not os.path.exists(ckpt_path):
        logger.warning(f"Checkpoint not found: {ckpt_path}")
        return None

    vocab, _ = load_rnn_vocab(ckpt_name)
    if vocab is None:
        logger.warning(f"Skipping {ckpt_name}: no matching vocabulary")
        return None

    model = model_cls(vocab_size=len(vocab))
    try:
        load_checkpoint(ckpt_path, model)
    except RuntimeError as e:
        logger.error(f"Failed to load {ckpt_name} with current vocab ({len(vocab)} tokens): {e}")
        return None
    model.to(config.DEVICE)

    test_ds = RNNDataset(test_df["full_text"].tolist(), test_df["label"].tolist(), vocab)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    preds, labels = get_predictions(model, test_loader, "rnn")
    pairs = get_top_confusion_pairs(labels, preds, top_k=10)
    logger.info(f"\nTop confusions ({ckpt_name}):")
    for p in pairs[:5]:
        logger.info(f"  {p['true_label']} -> {p['pred_label']}: {p['count']}")

    mis = extract_misclassified(test_df, labels, preds, n=50)
    mis_cat = categorize_errors(mis)
    mis_cat.to_csv(os.path.join(report_dir, f"{ckpt_name}_misclassified.csv"), index=False)

    return {
        "result_key": result_key,
        "ckpt_name": ckpt_name,
        "y_true": labels,
        "y_pred": preds,
        "pairs": pairs,
        "mis_cat": mis_cat,
    }


def analyze_transformer_model(test_df, report_dir, ckpt_name, model_cls, tokenizer, result_key):
    """Run error analysis for a single transformer checkpoint."""
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"{ckpt_name}_best.pt")
    if not os.path.exists(ckpt_path):
        logger.warning(f"Checkpoint not found: {ckpt_path}")
        return None

    model = model_cls()
    load_checkpoint(ckpt_path, model)
    model.to(config.DEVICE)

    test_ds = TransformerDataset(test_df["full_text"].tolist(), test_df["label"].tolist(), tokenizer)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    preds, labels = get_predictions(model, test_loader, "transformer")
    pairs = get_top_confusion_pairs(labels, preds, top_k=10)
    logger.info(f"\nTop confusions ({ckpt_name}):")
    for p in pairs[:5]:
        logger.info(f"  {p['true_label']} -> {p['pred_label']}: {p['count']}")

    mis = extract_misclassified(test_df, labels, preds, n=50)
    mis_cat = categorize_errors(mis)
    mis_cat.to_csv(os.path.join(report_dir, f"{ckpt_name}_misclassified.csv"), index=False)

    return {
        "result_key": result_key,
        "ckpt_name": ckpt_name,
        "y_true": labels,
        "y_pred": preds,
        "pairs": pairs,
        "mis_cat": mis_cat,
    }


def main():
    set_seed()
    config.ensure_dirs()

    test_df = pd.read_csv(os.path.join(config.DATA_SPLITS_DIR, "test.csv"))
    report_dir = os.path.join(config.REPORTS_DIR, "error_analysis")
    os.makedirs(report_dir, exist_ok=True)

    results = []

    # Analyze 2 best RNN-family models.
    rnn_specs = [
        (BEST_MODELS["bilstm"], BiLSTM, "bilstm"),
        (BEST_MODELS["bilstm_attn"], BiLSTMAttention, "bilstm_attn"),
    ]
    for ckpt_name, model_cls, result_key in rnn_specs:
        output = analyze_rnn_model(test_df, report_dir, ckpt_name, model_cls, result_key)
        if output is not None:
            results.append(output)

    # Analyze 2 best transformer-family models.
    distil_tokenizer = get_transformer_tokenizer(config.DISTILBERT_MODEL_NAME)
    bert_tokenizer = get_transformer_tokenizer(config.BERT_MODEL_NAME)
    transformer_specs = [
        (BEST_MODELS["distilbert"], DistilBertClassifier, distil_tokenizer, "distilbert"),
        (BEST_MODELS["bert"], BertClassifier, bert_tokenizer, "bert"),
    ]
    for ckpt_name, model_cls, tokenizer, result_key in transformer_specs:
        output = analyze_transformer_model(test_df, report_dir, ckpt_name, model_cls, tokenizer, result_key)
        if output is not None:
            results.append(output)

    # Generate combined report for all analyzed best models.
    if results:
        lines = ["# Error Analysis Report (Best Models)\n"]
        summary_rows = []

        for item in results:
            y_true = item["y_true"]
            y_pred = item["y_pred"]
            accuracy = float((y_true == y_pred).mean())
            summary_rows.append((item["result_key"], item["ckpt_name"], accuracy, 1.0 - accuracy))

            lines.append(f"\n## {item['result_key']} ({item['ckpt_name']})\n")

            lines.append("### Top Confusion Pairs\n")
            lines.append("| True Rating | Predicted Rating | Count |")
            lines.append("|-------------|------------------|-------|")
            for pair in item["pairs"]:
                lines.append(f"| {pair['true_label']} | {pair['pred_label']} | {pair['count']} |")

            lines.append("\n### Error Categories\n")
            lines.append("| Category | Count |")
            lines.append("|----------|-------|")
            cat_counts = Counter(item["mis_cat"]["error_category"])
            for cat, count in cat_counts.most_common():
                lines.append(f"| {cat} | {count} |")

        summary_df = pd.DataFrame(
            summary_rows,
            columns=["model_family", "checkpoint", "accuracy", "error_rate"],
        )
        summary_df.to_csv(os.path.join(report_dir, "error_summary_best_models.csv"), index=False)

        lines.append("\n## Summary (Best Models)\n")
        lines.append("| Model Family | Checkpoint | Accuracy | Error Rate |")
        lines.append("|--------------|------------|----------|------------|")
        for family, ckpt, acc, err in summary_rows:
            lines.append(f"| {family} | {ckpt} | {acc:.4f} | {err:.4f} |")

        report_path = os.path.join(report_dir, "error_analysis_report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        logger.info(f"[OK] Error analysis report saved to {report_path}")

    logger.info("[OK] Error analysis complete")


if __name__ == "__main__":
    main()


