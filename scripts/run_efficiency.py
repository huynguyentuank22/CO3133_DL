"""
Run efficiency comparison across all models.
Usage: python scripts/run_efficiency.py
"""
import sys, os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src import config
from src.utils import set_seed, get_logger, load_checkpoint
from src.data_utils import get_transformer_tokenizer, load_rnn_vocab
from src.datasets import RNNDataset, TransformerDataset
from src.rnn_models import BiLSTM, BiLSTMAttention
from src.transformer_models import DistilBertClassifier, BertClassifier
from src.efficiency import get_efficiency_stats, create_efficiency_table
from src.visualization import plot_accuracy_vs_inference, plot_f1_vs_size

logger = get_logger("efficiency", os.path.join(config.LOGS_DIR, "efficiency.log"))


def main():
    set_seed()
    config.ensure_dirs()

    test_df = pd.read_csv(os.path.join(config.DATA_SPLITS_DIR, "test.csv"))
    all_stats = []

    # RNN models
    for model_class, name, ckpt_name in [
        (BiLSTM, "BiLSTM", "bilstm_weighted_ce"),
        (BiLSTMAttention, "BiLSTM+Attention", "bilstm_attention_weighted_ce"),
    ]:
        ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"{ckpt_name}_best.pt")
        if not os.path.exists(ckpt_path):
            continue

        vocab, _ = load_rnn_vocab(ckpt_name)
        if vocab is None:
            logger.warning(f"Skipping {name}: no matching vocabulary for {ckpt_name}")
            continue

        model = model_class(vocab_size=len(vocab))
        try:
            load_checkpoint(ckpt_path, model)
        except RuntimeError as e:
            logger.error(f"Skipping {name}: checkpoint-vocab mismatch for {ckpt_name}: {e}")
            continue
        model.to(config.DEVICE)

        test_ds = RNNDataset(test_df["full_text"].tolist(), test_df["label"].tolist(), vocab)
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

        stats = get_efficiency_stats(model, name, test_loader, "rnn")
        all_stats.append(stats)
        logger.info(f"{name}: {stats}")

    # Transformer models
    for model_class, name, ckpt_name, tok_name in [
        (DistilBertClassifier, "DistilBERT", "distilbert_full_weighted_ce", config.DISTILBERT_MODEL_NAME),
        (BertClassifier, "BERT-base", "bert_full_weighted_ce", config.BERT_MODEL_NAME),
    ]:
        ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"{ckpt_name}_best.pt")
        if not os.path.exists(ckpt_path):
            continue
        model = model_class()
        load_checkpoint(ckpt_path, model)
        model.to(config.DEVICE)

        tokenizer = get_transformer_tokenizer(tok_name)
        test_ds = TransformerDataset(test_df["full_text"].tolist(), test_df["label"].tolist(), tokenizer)
        test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

        stats = get_efficiency_stats(model, name, test_loader, "transformer")
        all_stats.append(stats)
        logger.info(f"{name}: {stats}")

    if all_stats:
        save_path = os.path.join(config.TABLES_DIR, "efficiency_comparison.csv")
        df = create_efficiency_table(all_stats, save_path)

        # Add accuracy/macro_f1 from existing metrics for plotting
        for i, row in df.iterrows():
            metrics_path = os.path.join(config.REPORTS_DIR,
                                         f"{row['model_name'].lower().replace('+', '_').replace(' ', '_')}_metrics.json")
            # Try finding metrics
            import json
            for candidate in os.listdir(config.REPORTS_DIR):
                if candidate.endswith("_metrics.json"):
                    with open(os.path.join(config.REPORTS_DIR, candidate)) as f:
                        m = json.load(f)
                    if m.get("model_name", "").startswith(row["model_name"].lower().replace("+", "_").replace(" ", "_").split("_")[0]):
                        df.loc[i, "accuracy"] = m.get("accuracy", 0)
                        df.loc[i, "macro_f1"] = m.get("macro_f1", 0)
                        break

        if "accuracy" in df.columns:
            plot_accuracy_vs_inference(df)
        if "macro_f1" in df.columns:
            plot_f1_vs_size(df)

        print("\n" + df.to_string(index=False))

    logger.info("[OK] Efficiency comparison complete")


if __name__ == "__main__":
    main()


