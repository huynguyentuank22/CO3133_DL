"""
Run ensemble: weighted average of probabilities from two hard-coded models.
Usage: python scripts/run_ensemble.py
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
from src.trainer import Trainer
from src.ensemble import run_ensemble, save_ensemble_results

logger = get_logger("ensemble", os.path.join(config.LOGS_DIR, "ensemble.log"))

# Hard-coded pair for ensembling.
# You can set any two checkpoints here (same or different model families).
MODEL_A = {
    "display_name": "BERT-full-undersample",
    "model_type": "transformer",          # "rnn" | "transformer"
    "architecture": "bert",               # rnn: bilstm|bilstm_attn, transformer: distilbert|bert
    "checkpoint": "bert_full_undersample_ce",  # stem without _best.pt
}

MODEL_B = {
    "display_name": "BERT-full-weighted",
    "model_type": "transformer",
    "architecture": "bert",
    "checkpoint": "bert_full_weighted_ce",
}

RNN_CLASS_MAP = {
    "bilstm": BiLSTM,
    "bilstm_attn": BiLSTMAttention,
}

TRANSFORMER_CLASS_MAP = {
    "distilbert": (DistilBertClassifier, config.DISTILBERT_MODEL_NAME),
    "bert": (BertClassifier, config.BERT_MODEL_NAME),
}


def get_probabilities(model, dataloader, model_type):
    trainer = Trainer(model, None, torch.nn.CrossEntropyLoss(),
                      model_type=model_type, model_name="tmp")
    return trainer.predict_proba(dataloader)


def _get_checkpoint_path(checkpoint_stem: str) -> str:
    return os.path.join(config.CHECKPOINT_DIR, f"{checkpoint_stem}_best.pt")


def load_model_probabilities(spec: dict, test_df: pd.DataFrame):
    """Load one model from spec and return probability predictions on test set."""
    display_name = spec["display_name"]
    model_type = spec["model_type"]
    architecture = spec["architecture"]
    checkpoint_stem = spec["checkpoint"]
    ckpt_path = _get_checkpoint_path(checkpoint_stem)

    if not os.path.exists(ckpt_path):
        logger.error(f"Checkpoint not found for {display_name}: {ckpt_path}")
        return None

    if model_type == "rnn":
        model_class = RNN_CLASS_MAP.get(architecture)
        if model_class is None:
            logger.error(f"Unsupported RNN architecture for {display_name}: {architecture}")
            return None

        vocab, _ = load_rnn_vocab(checkpoint_stem)
        if vocab is None:
            logger.error(f"Vocabulary not found for {display_name}: {checkpoint_stem}")
            return None

        model = model_class(vocab_size=len(vocab))
        try:
            load_checkpoint(ckpt_path, model)
        except RuntimeError as e:
            logger.error(f"Failed to load {display_name} ({checkpoint_stem}): {e}")
            return None
        model.to(config.DEVICE)

        dataset = RNNDataset(test_df["full_text"].tolist(), test_df["label"].tolist(), vocab)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
        logger.info(f"Loaded {display_name} from {checkpoint_stem}")
        return get_probabilities(model, dataloader, "rnn")

    if model_type == "transformer":
        transformer_info = TRANSFORMER_CLASS_MAP.get(architecture)
        if transformer_info is None:
            logger.error(f"Unsupported transformer architecture for {display_name}: {architecture}")
            return None

        model_class, tokenizer_name = transformer_info
        model = model_class()
        load_checkpoint(ckpt_path, model)
        model.to(config.DEVICE)

        tokenizer = get_transformer_tokenizer(tokenizer_name)
        dataset = TransformerDataset(test_df["full_text"].tolist(), test_df["label"].tolist(), tokenizer)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
        logger.info(f"Loaded {display_name} from {checkpoint_stem}")
        return get_probabilities(model, dataloader, "transformer")

    logger.error(f"Unsupported model_type for {display_name}: {model_type}")
    return None


def main():
    set_seed()
    config.ensure_dirs()

    test_df = pd.read_csv(os.path.join(config.DATA_SPLITS_DIR, "test.csv"))
    y_true = test_df["label"].values

    probs_a = load_model_probabilities(MODEL_A, test_df)
    if probs_a is None:
        return

    probs_b = load_model_probabilities(MODEL_B, test_df)
    if probs_b is None:
        return

    if probs_a.shape != probs_b.shape:
        logger.error(
            "Probability shape mismatch between selected models: %s vs %s",
            probs_a.shape,
            probs_b.shape,
        )
        return

    comparison, alpha_df, best_alpha = run_ensemble(
        probs_a, probs_b, y_true, config.ENSEMBLE_ALPHAS
    )

    # Rename rows so outputs reflect selected hard-coded models.
    if "model" in comparison.columns and len(comparison) >= 3:
        comparison.loc[comparison.index[0], "model"] = MODEL_A["display_name"]
        comparison.loc[comparison.index[1], "model"] = MODEL_B["display_name"]
        comparison.loc[
            comparison.index[2],
            "model",
        ] = f"Ensemble ({MODEL_A['display_name']} + {MODEL_B['display_name']}, alpha={best_alpha})"

    alpha_df.insert(0, "model_a", MODEL_A["display_name"])
    alpha_df.insert(1, "model_b", MODEL_B["display_name"])

    save_dir = os.path.join(config.TABLES_DIR)
    save_ensemble_results(comparison, alpha_df, save_dir)

    print("\n=== Ensemble Comparison ===")
    print(comparison.to_string(index=False))
    print(f"\nBest alpha: {best_alpha}")
    logger.info("[OK] Ensemble complete")


if __name__ == "__main__":
    main()


