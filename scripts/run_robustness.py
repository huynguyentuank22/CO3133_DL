"""
Run robustness evaluation: clean vs noisy test set.
Usage: python scripts/run_robustness.py
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
from src.metrics import compute_metrics
from src.robustness import create_noisy_test_set, compare_robustness

logger = get_logger("robustness", os.path.join(config.LOGS_DIR, "robustness.log"))


def eval_model(model, dataloader, model_type):
    trainer = Trainer(model, None, torch.nn.CrossEntropyLoss(),
                      model_type=model_type, model_name="tmp")
    _, metrics, _, _ = trainer.evaluate(dataloader)
    return metrics


def main():
    set_seed()
    config.ensure_dirs()

    test_df = pd.read_csv(os.path.join(config.DATA_SPLITS_DIR, "test.csv"))
    noisy_test_df = create_noisy_test_set(test_df)

    results = []

    # BiLSTM+Attention
    ckpt_name = "bilstm_attention_weighted_ce"
    vocab, _ = load_rnn_vocab(ckpt_name)
    ckpt = os.path.join(config.CHECKPOINT_DIR, f"{ckpt_name}_best.pt")
    if vocab is not None and os.path.exists(ckpt):
        model = BiLSTMAttention(vocab_size=len(vocab))
        load_checkpoint(ckpt, model)
        model.to(config.DEVICE)

        # Clean
        clean_ds = RNNDataset(test_df["full_text"].tolist(), test_df["label"].tolist(), vocab)
        clean_loader = DataLoader(clean_ds, batch_size=64, shuffle=False)
        clean_m = eval_model(model, clean_loader, "rnn")

        # Noisy
        noisy_ds = RNNDataset(noisy_test_df["full_text"].tolist(), noisy_test_df["label"].tolist(), vocab)
        noisy_loader = DataLoader(noisy_ds, batch_size=64, shuffle=False)
        noisy_m = eval_model(model, noisy_loader, "rnn")

        results.append(compare_robustness(clean_m, noisy_m, "BiLSTM+Attention"))

    # DistilBERT
    ckpt = os.path.join(config.CHECKPOINT_DIR, "distilbert_full_weighted_ce_best.pt")
    if os.path.exists(ckpt):
        model = DistilBertClassifier()
        load_checkpoint(ckpt, model)
        model.to(config.DEVICE)
        tokenizer = get_transformer_tokenizer(config.DISTILBERT_MODEL_NAME)

        clean_ds = TransformerDataset(test_df["full_text"].tolist(), test_df["label"].tolist(), tokenizer)
        clean_loader = DataLoader(clean_ds, batch_size=16, shuffle=False)
        clean_m = eval_model(model, clean_loader, "transformer")

        noisy_ds = TransformerDataset(noisy_test_df["full_text"].tolist(), noisy_test_df["label"].tolist(), tokenizer)
        noisy_loader = DataLoader(noisy_ds, batch_size=16, shuffle=False)
        noisy_m = eval_model(model, noisy_loader, "transformer")

        results.append(compare_robustness(clean_m, noisy_m, "DistilBERT"))

    if results:
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(config.TABLES_DIR, "robustness_results.csv"), index=False)
        print("\n" + df.to_string(index=False))

    logger.info("[OK] Robustness evaluation complete")


if __name__ == "__main__":
    main()


