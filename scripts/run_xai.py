"""
Run XAI explainability analysis.
Usage: python scripts/run_xai.py
"""
import sys, os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import pandas as pd
import numpy as np
import torch
from src import config
from src.utils import set_seed, get_logger, load_checkpoint
from src.data_utils import get_transformer_tokenizer, load_rnn_vocab
from src.rnn_models import BiLSTMAttention
from src.transformer_models import DistilBertClassifier
from src.infer import predict_single_rnn_attention, predict_single_transformer
from src.xai_utils import (
    get_attention_explanation, attention_highlight_html,
    get_transformer_explanation, lime_explain, save_xai_html,
)

logger = get_logger("run_xai", os.path.join(config.LOGS_DIR, "run_xai.log"))


def main():
    set_seed()
    config.ensure_dirs()

    test_df = pd.read_csv(os.path.join(config.DATA_SPLITS_DIR, "test.csv"))
    xai_dir = os.path.join(config.REPORTS_DIR, "xai_results")
    os.makedirs(xai_dir, exist_ok=True)

    # â”€â”€â”€ BiLSTM + Attention XAI â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    vocab, _ = load_rnn_vocab("bilstm_attention_weighted_ce")
    rnn_ckpt = os.path.join(config.CHECKPOINT_DIR, "bilstm_attention_weighted_ce_best.pt")

    rnn_available = (vocab is not None) and os.path.exists(rnn_ckpt)
    if rnn_available:
        rnn_model = BiLSTMAttention(vocab_size=len(vocab))
        load_checkpoint(rnn_ckpt, rnn_model)
        rnn_model.to(config.DEVICE)
        logger.info("Loaded BiLSTM+Attention for XAI")

        # Select samples: 3 correct, 3 incorrect
        # Quick predictions on a subset
        rnn_htmls = ["<h1>BiLSTM + Attention â€” XAI Results</h1>"]
        correct_samples = []
        incorrect_samples = []

        for idx, row in test_df.iterrows():
            if len(correct_samples) >= 3 and len(incorrect_samples) >= 3:
                break
            text = row["full_text"]
            true_label = row["label"]
            explanation = get_attention_explanation(rnn_model, text, vocab)
            pred = explanation["pred"]
            confidence = float(explanation["probs"][pred])

            sample_info = {
                "idx": idx, "text": text,
                "true": config.LABEL_MAP_INV[true_label],
                "pred": config.LABEL_MAP_INV[pred],
                "confidence": confidence,
                "tokens": explanation["tokens"],
                "weights": explanation["weights"],
                "top_tokens": explanation["top_tokens"],
            }
            if pred == true_label and len(correct_samples) < 3:
                correct_samples.append(sample_info)
            elif pred != true_label and len(incorrect_samples) < 3:
                incorrect_samples.append(sample_info)

        rnn_htmls.append("<h2>Correct Predictions</h2>")
        for s in correct_samples:
            h = attention_highlight_html(
                s["tokens"], s["weights"], s["true"], s["pred"],
                s["confidence"], text_id=str(s["idx"]))
            top_str = ", ".join([f"{t}({w:.3f})" for t, w in s["top_tokens"][:5]])
            rnn_htmls.append(h)
            rnn_htmls.append(f"<p><em>Top tokens: {top_str}</em></p><hr>")

        rnn_htmls.append("<h2>Incorrect Predictions</h2>")
        for s in incorrect_samples:
            h = attention_highlight_html(
                s["tokens"], s["weights"], s["true"], s["pred"],
                s["confidence"], text_id=str(s["idx"]))
            top_str = ", ".join([f"{t}({w:.3f})" for t, w in s["top_tokens"][:5]])
            rnn_htmls.append(h)
            rnn_htmls.append(f"<p><em>Top tokens: {top_str}</em></p><hr>")

        save_xai_html("\n".join(rnn_htmls),
                      os.path.join(xai_dir, "bilstm_attention_xai.html"))
    else:
        logger.warning("BiLSTM+Attention checkpoint or vocab not found, skipping")

    # â”€â”€â”€ DistilBERT XAI (Captum) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    trans_ckpt = os.path.join(config.CHECKPOINT_DIR, "distilbert_full_weighted_ce_best.pt")
    if os.path.exists(trans_ckpt):
        trans_model = DistilBertClassifier()
        load_checkpoint(trans_ckpt, trans_model)
        trans_model.to(config.DEVICE)
        tokenizer = get_transformer_tokenizer(config.DISTILBERT_MODEL_NAME)
        logger.info("Loaded DistilBERT for XAI")

        trans_htmls = ["<h1>DistilBERT â€” XAI Results (Integrated Gradients)</h1>"]
        correct_samples = []
        incorrect_samples = []

        for idx, row in test_df.iterrows():
            if len(correct_samples) >= 3 and len(incorrect_samples) >= 3:
                break
            text = row["full_text"]
            true_label = row["label"]
            try:
                explanation = get_transformer_explanation(trans_model, text, tokenizer)
            except Exception as e:
                logger.warning(f"IG failed for sample {idx}: {e}")
                continue

            pred = explanation["pred"]
            confidence = float(explanation["probs"][pred])

            sample_info = {
                "idx": idx, "text": text,
                "true": config.LABEL_MAP_INV[true_label],
                "pred": config.LABEL_MAP_INV[pred],
                "confidence": confidence,
                "tokens": explanation["tokens"],
                "scores": explanation["scores"],
                "top_tokens": explanation["top_tokens"],
            }
            if pred == true_label and len(correct_samples) < 3:
                correct_samples.append(sample_info)
            elif pred != true_label and len(incorrect_samples) < 3:
                incorrect_samples.append(sample_info)

        trans_htmls.append("<h2>Correct Predictions</h2>")
        for s in correct_samples:
            h = attention_highlight_html(
                s["tokens"], s["scores"], s["true"], s["pred"],
                s["confidence"], text_id=str(s["idx"]))
            top_str = ", ".join([f"{t}({w:.3f})" for t, w in s["top_tokens"][:5]])
            trans_htmls.append(h)
            trans_htmls.append(f"<p><em>Top tokens: {top_str}</em></p><hr>")

        trans_htmls.append("<h2>Incorrect Predictions</h2>")
        for s in incorrect_samples:
            h = attention_highlight_html(
                s["tokens"], s["scores"], s["true"], s["pred"],
                s["confidence"], text_id=str(s["idx"]))
            top_str = ", ".join([f"{t}({w:.3f})" for t, w in s["top_tokens"][:5]])
            trans_htmls.append(h)
            trans_htmls.append(f"<p><em>Top tokens: {top_str}</em></p><hr>")

        save_xai_html("\n".join(trans_htmls),
                      os.path.join(xai_dir, "distilbert_xai.html"))
    else:
        logger.warning("DistilBERT checkpoint not found, skipping")

    # â”€â”€â”€ LIME explanation for a few samples â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if rnn_available:
        from src.infer import predict_batch_proba_rnn
        logger.info("Running LIME explanations...")
        lime_htmls = ["<h1>LIME Explanations (BiLSTM+Attention)</h1>"]

        def rnn_predict_fn(texts):
            return predict_batch_proba_rnn(rnn_model, texts, vocab)

        for i in range(min(3, len(test_df))):
            text = test_df.iloc[i]["full_text"]
            explanation = lime_explain(rnn_predict_fn, text, num_features=8, num_samples=200)
            if explanation:
                pred_label = explanation.available_labels()[0]
                lime_html = explanation.as_html()
                lime_htmls.append(f"<h3>Sample {i}</h3>{lime_html}<hr>")

        save_xai_html("\n".join(lime_htmls),
                      os.path.join(xai_dir, "lime_explanations.html"))

    logger.info(f"[OK] XAI results saved to {xai_dir}")


if __name__ == "__main__":
    main()


