"""
Run ensemble: weighted average of RNN and Transformer probabilities.
Usage: python scripts/run_ensemble.py
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
from src.data_utils import Vocabulary, get_transformer_tokenizer
from src.datasets import RNNDataset, TransformerDataset
from src.rnn_models import BiLSTMAttention
from src.transformer_models import DistilBertClassifier, BertClassifier
from src.trainer import Trainer
from src.ensemble import run_ensemble, save_ensemble_results

logger = get_logger("ensemble", os.path.join(config.LOGS_DIR, "ensemble.log"))


def get_probabilities(model, dataloader, model_type):
    trainer = Trainer(model, None, torch.nn.CrossEntropyLoss(),
                      model_type=model_type, model_name="tmp")
    return trainer.predict_proba(dataloader)


def main():
    set_seed()
    config.ensure_dirs()

    test_df = pd.read_csv(os.path.join(config.DATA_SPLITS_DIR, "test.csv"))
    y_true = test_df["label"].values

    # â”€â”€â”€ RNN probabilities â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    vocab = Vocabulary()
    vocab_path = os.path.join(config.DATA_PROCESSED_DIR, "vocab.json")
    if not os.path.exists(vocab_path):
        logger.error("Vocabulary not found. Train an RNN model first.")
        return
    vocab.load(vocab_path)

    rnn_ckpt = os.path.join(config.CHECKPOINT_DIR, "bilstm_attention_weighted_ce_best.pt")
    if not os.path.exists(rnn_ckpt):
        logger.error("BiLSTM+Attention checkpoint not found.")
        return

    rnn_model = BiLSTMAttention(vocab_size=len(vocab))
    load_checkpoint(rnn_ckpt, rnn_model)
    rnn_model.to(config.DEVICE)

    test_ds_rnn = RNNDataset(test_df["full_text"].tolist(), test_df["label"].tolist(), vocab)
    test_loader_rnn = DataLoader(test_ds_rnn, batch_size=64, shuffle=False)
    probs_rnn = get_probabilities(rnn_model, test_loader_rnn, "rnn")

    # â”€â”€â”€ Transformer probabilities â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Try DistilBERT first, then BERT-base
    trans_model = None
    tok_name = None
    for ckpt_name, model_class, tname in [
        ("distilbert_full_weighted_ce_best.pt", DistilBertClassifier, config.DISTILBERT_MODEL_NAME),
        ("bert_full_weighted_ce_best.pt", BertClassifier, config.BERT_MODEL_NAME),
    ]:
        ckpt_path = os.path.join(config.CHECKPOINT_DIR, ckpt_name)
        if os.path.exists(ckpt_path):
            trans_model = model_class()
            load_checkpoint(ckpt_path, trans_model)
            trans_model.to(config.DEVICE)
            tok_name = tname
            logger.info(f"Using {ckpt_name} for ensemble")
            break

    if trans_model is None:
        logger.error("No Transformer checkpoint found.")
        return

    tokenizer = get_transformer_tokenizer(tok_name)
    test_ds_trans = TransformerDataset(test_df["full_text"].tolist(), test_df["label"].tolist(), tokenizer)
    test_loader_trans = DataLoader(test_ds_trans, batch_size=16, shuffle=False)
    probs_trans = get_probabilities(trans_model, test_loader_trans, "transformer")

    # â”€â”€â”€ Ensemble â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    comparison, alpha_df, best_alpha = run_ensemble(
        probs_rnn, probs_trans, y_true, config.ENSEMBLE_ALPHAS
    )

    save_dir = os.path.join(config.TABLES_DIR)
    save_ensemble_results(comparison, alpha_df, save_dir)

    print("\n=== Ensemble Comparison ===")
    print(comparison.to_string(index=False))
    print(f"\nBest alpha: {best_alpha}")
    logger.info("[OK] Ensemble complete")


if __name__ == "__main__":
    main()


