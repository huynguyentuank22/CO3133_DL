"""
Run XAI explainability analysis.
Usage: python scripts/run_xai.py
"""
import sys, os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import pandas as pd
import numpy as np
from src import config
from src.utils import set_seed, get_logger, load_checkpoint
from src.data_utils import get_transformer_tokenizer, load_rnn_vocab
from src.rnn_models import BiLSTM, BiLSTMAttention
from src.transformer_models import DistilBertClassifier, BertClassifier
from src.infer import predict_batch_proba_rnn
from src.xai_utils import (
    get_attention_explanation, attention_highlight_html,
    get_transformer_explanation, lime_explain, save_xai_html,
)

logger = get_logger("run_xai", os.path.join(config.LOGS_DIR, "run_xai.log"))

LIME_TARGET_SAMPLES = 3
LIME_MAX_CANDIDATES = 200

# Hard-coded best checkpoints by group (based on current README summary).
BEST_MODELS = {
    "bilstm": "bilstm_weighted_ce",
    "bilstm_attn": "bilstm_attention_weighted_ce",
    "distilbert": "distilbert_full_weighted_ce",
    "bert": "bert_llrd_weighted_ce",
}


def run_attention_xai(model, vocab, test_df, output_path):
    """Generate attention-based XAI HTML for BiLSTM+Attention."""
    htmls = ["<h1>BiLSTM+Attention - XAI Results</h1>"]
    correct_samples = []
    incorrect_samples = []

    for idx, row in test_df.iterrows():
        if len(correct_samples) >= 3 and len(incorrect_samples) >= 3:
            break

        text = row["full_text"]
        true_label = row["label"]
        explanation = get_attention_explanation(model, text, vocab)
        pred = explanation["pred"]
        confidence = float(explanation["probs"][pred])

        sample_info = {
            "idx": idx,
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

    htmls.append("<h2>Correct Predictions</h2>")
    for s in correct_samples:
        h = attention_highlight_html(
            s["tokens"], s["weights"], s["true"], s["pred"],
            s["confidence"], text_id=str(s["idx"])
        )
        top_str = ", ".join([f"{t}({w:.3f})" for t, w in s["top_tokens"][:5]])
        htmls.append(h)
        htmls.append(f"<p><em>Top tokens: {top_str}</em></p><hr>")

    htmls.append("<h2>Incorrect Predictions</h2>")
    for s in incorrect_samples:
        h = attention_highlight_html(
            s["tokens"], s["weights"], s["true"], s["pred"],
            s["confidence"], text_id=str(s["idx"])
        )
        top_str = ", ".join([f"{t}({w:.3f})" for t, w in s["top_tokens"][:5]])
        htmls.append(h)
        htmls.append(f"<p><em>Top tokens: {top_str}</em></p><hr>")

    save_xai_html("\n".join(htmls), output_path)


def run_transformer_xai(model, tokenizer, test_df, model_title, output_path):
    """Generate IG-based XAI HTML for a transformer model."""
    htmls = [f"<h1>{model_title} - XAI Results (Integrated Gradients)</h1>"]
    correct_samples = []
    incorrect_samples = []

    for idx, row in test_df.iterrows():
        if len(correct_samples) >= 3 and len(incorrect_samples) >= 3:
            break

        text = row["full_text"]
        true_label = row["label"]

        try:
            explanation = get_transformer_explanation(model, text, tokenizer)
        except Exception as e:
            logger.warning(f"IG failed for sample {idx}: {e}")
            continue

        pred = explanation["pred"]
        confidence = float(explanation["probs"][pred])

        sample_info = {
            "idx": idx,
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

    htmls.append("<h2>Correct Predictions</h2>")
    for s in correct_samples:
        h = attention_highlight_html(
            s["tokens"], s["scores"], s["true"], s["pred"],
            s["confidence"], text_id=str(s["idx"])
        )
        top_str = ", ".join([f"{t}({w:.3f})" for t, w in s["top_tokens"][:5]])
        htmls.append(h)
        htmls.append(f"<p><em>Top tokens: {top_str}</em></p><hr>")

    htmls.append("<h2>Incorrect Predictions</h2>")
    for s in incorrect_samples:
        h = attention_highlight_html(
            s["tokens"], s["scores"], s["true"], s["pred"],
            s["confidence"], text_id=str(s["idx"])
        )
        top_str = ", ".join([f"{t}({w:.3f})" for t, w in s["top_tokens"][:5]])
        htmls.append(h)
        htmls.append(f"<p><em>Top tokens: {top_str}</em></p><hr>")

    save_xai_html("\n".join(htmls), output_path)


def run_lime_xai_for_rnn(model, vocab, test_df, title, output_path):
    """Generate LIME-based XAI HTML for an RNN model."""
    logger.info(f"Running LIME explanations for {title}...")
    htmls = [f"<h1>{title} - LIME Explanations</h1>"]
    skipped_samples = []
    selected_samples = []
    empty_perturbation_count = 0

    def rnn_predict_fn(texts):
        nonlocal empty_perturbation_count
        empty_positions = [idx for idx, t in enumerate(texts) if not str(t).strip()]
        safe_texts = list(texts)
        if empty_positions:
            empty_perturbation_count += len(empty_positions)
            # Replace empty perturbations with a safe token so LIME run can continue.
            for pos in empty_positions:
                safe_texts[pos] = "unk"
            logger.warning(
                "[LIME-EMPTY] Empty perturbations detected for %s at batch positions %s (total=%d). "
                "Replaced with fallback token.",
                title,
                empty_positions[:10],
                len(empty_positions),
            )

        probs = predict_batch_proba_rnn(model, safe_texts, vocab)
        probs = np.nan_to_num(
            probs,
            nan=1.0 / config.NUM_CLASSES,
            posinf=1.0 / config.NUM_CLASSES,
            neginf=0.0,
        )
        row_sums = probs.sum(axis=1, keepdims=True)
        row_sums[row_sums <= 0] = 1.0
        return probs / row_sums

    candidate_limit = min(len(test_df), LIME_MAX_CANDIDATES)
    for i in range(candidate_limit):
        if len(selected_samples) >= LIME_TARGET_SAMPLES:
            break

        text = test_df.iloc[i]["full_text"]
        try:
            explanation = lime_explain(rnn_predict_fn, text, num_features=8, num_samples=200)
            if explanation:
                selected_samples.append(i)
                htmls.append(f"<h3>Sample {i}</h3>{explanation.as_html()}<hr>")
        except Exception as e:
            preview = str(text).replace("\n", " ")[:140]
            skipped_samples.append((i, str(e), preview))
            logger.warning(
                f"[LIME-SKIP] Sample {i} skipped for {title}. "
                f"Reason: {e}. Text preview: {preview}"
            )

    logger.info(
        "LIME summary for %s: selected=%d, skipped=%d, empty_perturbations=%d",
        title,
        len(selected_samples),
        len(skipped_samples),
        empty_perturbation_count,
    )

    if selected_samples:
        htmls.append("<h2>Selected Samples</h2>")
        htmls.append(f"<p>{', '.join([str(s) for s in selected_samples])}</p>")
    else:
        htmls.append("<h2>Selected Samples</h2>")
        htmls.append("<p>None</p>")

    if skipped_samples:
        htmls.append("<h2>Skipped Samples</h2>")
        htmls.append("<ul>")
        for sample_idx, reason, preview in skipped_samples:
            htmls.append(
                f"<li><b>Sample {sample_idx}</b>: {reason}<br><em>{preview}</em></li>"
            )
        htmls.append("</ul>")

    save_xai_html("\n".join(htmls), output_path)


def main():
    set_seed()
    config.ensure_dirs()

    test_df = pd.read_csv(os.path.join(config.DATA_SPLITS_DIR, "test.csv"))
    xai_dir = os.path.join(config.REPORTS_DIR, "xai_results")
    os.makedirs(xai_dir, exist_ok=True)

    # Best BiLSTM (LIME only)
    bilstm_name = BEST_MODELS["bilstm"]
    bilstm_vocab, _ = load_rnn_vocab(bilstm_name)
    bilstm_ckpt = os.path.join(config.CHECKPOINT_DIR, f"{bilstm_name}_best.pt")
    if bilstm_vocab is not None and os.path.exists(bilstm_ckpt):
        bilstm_model = BiLSTM(vocab_size=len(bilstm_vocab))
        load_checkpoint(bilstm_ckpt, bilstm_model)
        bilstm_model.to(config.DEVICE)
        logger.info(f"Loaded {bilstm_name} for XAI")
        run_lime_xai_for_rnn(
            bilstm_model,
            bilstm_vocab,
            test_df,
            "BiLSTM (best)",
            os.path.join(xai_dir, "bilstm_xai.html"),
        )
    else:
        logger.warning(f"{bilstm_name} checkpoint or vocab not found, skipping")

    # Best BiLSTM+Attention (attention + LIME)
    attn_name = BEST_MODELS["bilstm_attn"]
    attn_vocab, _ = load_rnn_vocab(attn_name)
    attn_ckpt = os.path.join(config.CHECKPOINT_DIR, f"{attn_name}_best.pt")
    if attn_vocab is not None and os.path.exists(attn_ckpt):
        attn_model = BiLSTMAttention(vocab_size=len(attn_vocab))
        load_checkpoint(attn_ckpt, attn_model)
        attn_model.to(config.DEVICE)
        logger.info(f"Loaded {attn_name} for XAI")
        run_attention_xai(
            attn_model,
            attn_vocab,
            test_df,
            os.path.join(xai_dir, "bilstm_attention_xai.html"),
        )
        run_lime_xai_for_rnn(
            attn_model,
            attn_vocab,
            test_df,
            "BiLSTM+Attention (best)",
            os.path.join(xai_dir, "bilstm_attention_lime.html"),
        )
    else:
        logger.warning(f"{attn_name} checkpoint or vocab not found, skipping")

    # Best DistilBERT
    distil_name = BEST_MODELS["distilbert"]
    distil_ckpt = os.path.join(config.CHECKPOINT_DIR, f"{distil_name}_best.pt")
    if os.path.exists(distil_ckpt):
        distil_model = DistilBertClassifier()
        load_checkpoint(distil_ckpt, distil_model)
        distil_model.to(config.DEVICE)
        distil_tokenizer = get_transformer_tokenizer(config.DISTILBERT_MODEL_NAME)
        logger.info(f"Loaded {distil_name} for XAI")
        run_transformer_xai(
            distil_model,
            distil_tokenizer,
            test_df,
            "DistilBERT (best)",
            os.path.join(xai_dir, "distilbert_xai.html"),
        )
    else:
        logger.warning(f"{distil_name} checkpoint not found, skipping")

    # Best BERT-base
    bert_name = BEST_MODELS["bert"]
    bert_ckpt = os.path.join(config.CHECKPOINT_DIR, f"{bert_name}_best.pt")
    if os.path.exists(bert_ckpt):
        bert_model = BertClassifier()
        load_checkpoint(bert_ckpt, bert_model)
        bert_model.to(config.DEVICE)
        bert_tokenizer = get_transformer_tokenizer(config.BERT_MODEL_NAME)
        logger.info(f"Loaded {bert_name} for XAI")
        run_transformer_xai(
            bert_model,
            bert_tokenizer,
            test_df,
            "BERT-base (best)",
            os.path.join(xai_dir, "bert_xai.html"),
        )
    else:
        logger.warning(f"{bert_name} checkpoint not found, skipping")

    logger.info(f"[OK] XAI results saved to {xai_dir}")


if __name__ == "__main__":
    main()


