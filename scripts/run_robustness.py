"""
Run robustness evaluation: clean vs noisy test set.
Usage: python scripts/run_robustness.py
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
from src.robustness import create_noisy_test_set, compare_robustness

logger = get_logger("robustness", os.path.join(config.LOGS_DIR, "robustness.log"))

# Best checkpoints by model family (hard-coded from latest summary).
BEST_MODELS = {
    "bilstm": "bilstm_weighted_ce",
    "bilstm_attn": "bilstm_attention_weighted_ce",
    "distilbert": "distilbert_full_weighted_ce",
    "bert": "bert_llrd_weighted_ce",
}


def eval_model(model, dataloader, model_type):
    trainer = Trainer(model, None, torch.nn.CrossEntropyLoss(),
                      model_type=model_type, model_name="tmp")
    _, metrics, _, _ = trainer.evaluate(dataloader)
    return metrics


def evaluate_best_rnn_model(test_df, noisy_test_df, ckpt_name, model_class, display_name):
    """Evaluate clean vs noisy robustness for a best RNN-family model."""
    vocab, _ = load_rnn_vocab(ckpt_name)
    ckpt = os.path.join(config.CHECKPOINT_DIR, f"{ckpt_name}_best.pt")
    if vocab is None:
        logger.warning(f"Skipping {display_name}: no matching vocabulary for {ckpt_name}")
        return None
    if not os.path.exists(ckpt):
        logger.warning(f"Skipping {display_name}: checkpoint not found at {ckpt}")
        return None

    model = model_class(vocab_size=len(vocab))
    try:
        load_checkpoint(ckpt, model)
    except RuntimeError as e:
        logger.error(f"Skipping {display_name}: checkpoint-vocab mismatch for {ckpt_name}: {e}")
        return None
    model.to(config.DEVICE)

    clean_ds = RNNDataset(test_df["full_text"].tolist(), test_df["label"].tolist(), vocab)
    clean_loader = DataLoader(clean_ds, batch_size=64, shuffle=False)
    clean_m = eval_model(model, clean_loader, "rnn")

    noisy_ds = RNNDataset(noisy_test_df["full_text"].tolist(), noisy_test_df["label"].tolist(), vocab)
    noisy_loader = DataLoader(noisy_ds, batch_size=64, shuffle=False)
    noisy_m = eval_model(model, noisy_loader, "rnn")

    return compare_robustness(clean_m, noisy_m, display_name)


def evaluate_best_transformer_model(
    test_df,
    noisy_test_df,
    ckpt_name,
    model_class,
    tokenizer_name,
    display_name,
):
    """Evaluate clean vs noisy robustness for a best transformer-family model."""
    ckpt = os.path.join(config.CHECKPOINT_DIR, f"{ckpt_name}_best.pt")
    if not os.path.exists(ckpt):
        logger.warning(f"Skipping {display_name}: checkpoint not found at {ckpt}")
        return None

    model = model_class()
    load_checkpoint(ckpt, model)
    model.to(config.DEVICE)
    tokenizer = get_transformer_tokenizer(tokenizer_name)

    clean_ds = TransformerDataset(test_df["full_text"].tolist(), test_df["label"].tolist(), tokenizer)
    clean_loader = DataLoader(clean_ds, batch_size=16, shuffle=False)
    clean_m = eval_model(model, clean_loader, "transformer")

    noisy_ds = TransformerDataset(noisy_test_df["full_text"].tolist(), noisy_test_df["label"].tolist(), tokenizer)
    noisy_loader = DataLoader(noisy_ds, batch_size=16, shuffle=False)
    noisy_m = eval_model(model, noisy_loader, "transformer")

    return compare_robustness(clean_m, noisy_m, display_name)


def main():
    set_seed()
    config.ensure_dirs()

    test_df = pd.read_csv(os.path.join(config.DATA_SPLITS_DIR, "test.csv"))
    noisy_test_df = create_noisy_test_set(test_df)

    results = []

    rnn_specs = [
        (BEST_MODELS["bilstm"], BiLSTM, "BiLSTM"),
        (BEST_MODELS["bilstm_attn"], BiLSTMAttention, "BiLSTM+Attention"),
    ]
    for ckpt_name, model_class, display_name in rnn_specs:
        result = evaluate_best_rnn_model(test_df, noisy_test_df, ckpt_name, model_class, display_name)
        if result is not None:
            results.append(result)

    transformer_specs = [
        (BEST_MODELS["distilbert"], DistilBertClassifier, config.DISTILBERT_MODEL_NAME, "DistilBERT"),
        (BEST_MODELS["bert"], BertClassifier, config.BERT_MODEL_NAME, "BERT-base"),
    ]
    for ckpt_name, model_class, tokenizer_name, display_name in transformer_specs:
        result = evaluate_best_transformer_model(
            test_df,
            noisy_test_df,
            ckpt_name,
            model_class,
            tokenizer_name,
            display_name,
        )
        if result is not None:
            results.append(result)

    if results:
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(config.TABLES_DIR, "robustness_results.csv"), index=False)
        print("\n" + df.to_string(index=False))

    logger.info("[OK] Robustness evaluation complete")


if __name__ == "__main__":
    main()


