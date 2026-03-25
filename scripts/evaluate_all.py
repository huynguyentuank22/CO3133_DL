"""
Evaluate all trained models on the test set.
Usage: python scripts/evaluate_all.py
"""
import sys, os, json
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src import config
from src.utils import set_seed, get_logger, load_checkpoint, count_parameters, model_size_mb
from src.data_utils import Vocabulary, get_transformer_tokenizer
from src.datasets import RNNDataset, TransformerDataset
from src.rnn_models import BiLSTM, BiLSTMAttention
from src.transformer_models import DistilBertClassifier, BertClassifier
from src.metrics import compute_metrics, plot_confusion_matrix, get_classification_report, save_metrics
from src.trainer import Trainer
from src.efficiency import measure_inference_time

logger = get_logger("evaluate_all", os.path.join(config.LOGS_DIR, "evaluate_all.log"))


def eval_rnn_model(model_class, model_name, checkpoint_name, test_df, vocab, batch_size=64):
    """Evaluate an RNN model checkpoint."""
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"{checkpoint_name}_best.pt")
    if not os.path.exists(ckpt_path):
        logger.warning(f"Checkpoint not found: {ckpt_path}")
        return None

    model = model_class(vocab_size=len(vocab))
    load_checkpoint(ckpt_path, model)
    model.to(config.DEVICE)

    test_ds = RNNDataset(test_df["full_text"].tolist(), test_df["label"].tolist(), vocab)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    trainer = Trainer(model, None, torch.nn.CrossEntropyLoss(), model_type="rnn", model_name=model_name)
    _, metrics, preds, labels = trainer.evaluate(test_loader)

    timing = measure_inference_time(model, test_loader, "rnn")

    # Save results
    metrics["model_name"] = model_name
    metrics["num_parameters"] = count_parameters(model)
    metrics["model_size_mb"] = round(model_size_mb(model), 2)
    metrics.update(timing)

    # Plots
    cm_path = os.path.join(config.FIGURES_DIR, f"{model_name}_confusion_matrix.png")
    plot_confusion_matrix(labels, preds, title=f"{model_name} â€” Confusion Matrix", save_path=cm_path)

    report = get_classification_report(labels, preds)
    logger.info(f"\n{model_name} Classification Report:\n{report}")

    save_metrics(metrics, os.path.join(config.REPORTS_DIR, f"{model_name}_metrics.json"))
    return metrics


def eval_transformer_model(model_class, model_name, checkpoint_name, test_df,
                            tokenizer_name, batch_size=16):
    """Evaluate a Transformer model checkpoint."""
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"{checkpoint_name}_best.pt")
    if not os.path.exists(ckpt_path):
        logger.warning(f"Checkpoint not found: {ckpt_path}")
        return None

    model = model_class()
    load_checkpoint(ckpt_path, model)
    model.to(config.DEVICE)

    tokenizer = get_transformer_tokenizer(tokenizer_name)
    test_ds = TransformerDataset(test_df["full_text"].tolist(), test_df["label"].tolist(), tokenizer)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    trainer = Trainer(model, None, torch.nn.CrossEntropyLoss(), model_type="transformer", model_name=model_name)
    _, metrics, preds, labels = trainer.evaluate(test_loader)

    timing = measure_inference_time(model, test_loader, "transformer")

    metrics["model_name"] = model_name
    metrics["num_parameters"] = count_parameters(model)
    metrics["model_size_mb"] = round(model_size_mb(model), 2)
    metrics.update(timing)

    cm_path = os.path.join(config.FIGURES_DIR, f"{model_name}_confusion_matrix.png")
    plot_confusion_matrix(labels, preds, title=f"{model_name} â€” Confusion Matrix", save_path=cm_path)

    report = get_classification_report(labels, preds)
    logger.info(f"\n{model_name} Classification Report:\n{report}")

    save_metrics(metrics, os.path.join(config.REPORTS_DIR, f"{model_name}_metrics.json"))
    return metrics


def main():
    set_seed()
    config.ensure_dirs()
    test_df = pd.read_csv(os.path.join(config.DATA_SPLITS_DIR, "test.csv"))

    # Load vocabulary for RNN
    vocab = Vocabulary()
    vocab_path = os.path.join(config.DATA_PROCESSED_DIR, "vocab.json")
    if os.path.exists(vocab_path):
        vocab.load(vocab_path)
    else:
        logger.warning("Vocabulary file not found. Please train an RNN model first.")
        return

    all_results = []

    # RNN models
    rnn_configs = [
        (BiLSTM, "bilstm_weighted_ce", "bilstm_weighted_ce"),
        (BiLSTM, "bilstm_undersample_ce", "bilstm_undersample_ce"),
        (BiLSTMAttention, "bilstm_attention_weighted_ce", "bilstm_attention_weighted_ce"),
        (BiLSTMAttention, "bilstm_attention_undersample_ce", "bilstm_attention_undersample_ce"),
    ]
    for model_class, name, ckpt in rnn_configs:
        result = eval_rnn_model(model_class, name, ckpt, test_df, vocab)
        if result:
            all_results.append(result)

    # Transformer models
    trans_configs = [
        (DistilBertClassifier, "distilbert_full_weighted_ce", "distilbert_full_weighted_ce", config.DISTILBERT_MODEL_NAME),
        (DistilBertClassifier, "distilbert_full_undersample_ce", "distilbert_full_undersample_ce", config.DISTILBERT_MODEL_NAME),
        (DistilBertClassifier, "distilbert_freeze_weighted_ce", "distilbert_freeze_weighted_ce", config.DISTILBERT_MODEL_NAME),
        (DistilBertClassifier, "distilbert_llrd_weighted_ce", "distilbert_llrd_weighted_ce", config.DISTILBERT_MODEL_NAME),
        (BertClassifier, "bert_full_weighted_ce", "bert_full_weighted_ce", config.BERT_MODEL_NAME),
        (BertClassifier, "bert_full_undersample_ce", "bert_full_undersample_ce", config.BERT_MODEL_NAME),
    ]
    for model_class, name, ckpt, tok_name in trans_configs:
        result = eval_transformer_model(model_class, name, ckpt, test_df, tok_name)
        if result:
            all_results.append(result)

    # Save summary table
    if all_results:
        summary_df = pd.DataFrame(all_results)
        summary_path = os.path.join(config.TABLES_DIR, "model_comparison.csv")
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"[OK] Summary table saved to {summary_path}")
        print("\n" + summary_df.to_string(index=False))


if __name__ == "__main__":
    main()


