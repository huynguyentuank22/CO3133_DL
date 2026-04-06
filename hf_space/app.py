"""
Gradio Space app: multi-model rating prediction + explainability.

Hard-coded model source repository:
- MODEL_REPO_ID
"""
from __future__ import annotations

import html
import sys
from functools import lru_cache
from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

# Allow importing local project modules when this folder is in the same repo.
SPACE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SPACE_DIR.parent
for p in (SPACE_DIR, PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src import config
from src.data_utils import Vocabulary, get_transformer_tokenizer
from src.infer import predict_batch_proba_rnn, predict_single_rnn, predict_single_transformer
from src.rnn_models import BiLSTM, BiLSTMAttention
from src.transformer_models import BertClassifier, DistilBertClassifier
from src.utils import load_checkpoint, set_seed
from src.xai_utils import get_rnn_ig_explanation, get_transformer_explanation, lime_explain


set_seed()
config.ensure_dirs()

MODEL_REPO_ID = "huynguyentuan/DL-assignment-1"
MODEL_OPTIONS = {
    "BERT-base": {
        "model_type": "bert",
        "checkpoint": "bert_llrd_weighted_ce_best.pt",
        "explain": "ig",
        "vocab": None,
    },
    "DistilBERT": {
        "model_type": "distilbert",
        "checkpoint": "distilbert_full_weighted_ce_best.pt",
        "explain": "ig",
        "vocab": None,
    },
    "BiLSTM + Attention": {
        "model_type": "bilstm_attn",
        "checkpoint": "bilstm_attention_weighted_ce_best.pt",
        "explain": "ig",
        "vocab": "bilstm_attention_weighted_ce_vocab.json",
    },
    "BiLSTM": {
        "model_type": "bilstm",
        "checkpoint": "bilstm_weighted_ce_best.pt",
        "explain": "lime",
        "vocab": "bilstm_weighted_ce_vocab.json",
    },
}


def _download_from_repo(candidates: list[str]) -> str:
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            return hf_hub_download(repo_id=MODEL_REPO_ID, filename=candidate, repo_type="model")
        except Exception as exc:  # pragma: no cover - fallback path
            last_error = exc

    joined = ", ".join(candidates)
    raise FileNotFoundError(f"Cannot find any of these files in repo '{MODEL_REPO_ID}': {joined}") from last_error


def resolve_checkpoint_path(filename: str) -> str:
    return _download_from_repo(
        [
            filename,
            f"checkpoints/{filename}",
            f"outputs/checkpoints/{filename}",
        ]
    )


def resolve_vocab_path(filename: str) -> str:
    local_candidates = [
        SPACE_DIR / "data" / "processed" / "vocabs" / filename,
        SPACE_DIR / "vocabs" / filename,
    ]
    for p in local_candidates:
        if p.exists():
            return str(p)

    return _download_from_repo(
        [
            filename,
            f"vocabs/{filename}",
            f"data/processed/vocabs/{filename}",
        ]
    )


@lru_cache(maxsize=4)
def load_model_bundle(model_name: str):
    if model_name not in MODEL_OPTIONS:
        raise ValueError(f"Unknown model option: {model_name}")

    spec = MODEL_OPTIONS[model_name]
    model_type = spec["model_type"]
    checkpoint_filename = spec["checkpoint"]
    explain_method = spec["explain"]

    ckpt_path = resolve_checkpoint_path(checkpoint_filename)

    if model_type == "bilstm":
        vocab_filename = spec["vocab"]
        if not vocab_filename:
            raise RuntimeError("Vocabulary filename missing for BiLSTM")
        vocab_path = resolve_vocab_path(vocab_filename)
        vocab = Vocabulary()
        vocab.load(vocab_path)

        model = BiLSTM(vocab_size=len(vocab))
        load_checkpoint(ckpt_path, model)
        model.to(config.DEVICE)
        model.eval()

        return {
            "model_name": model_name,
            "model_type": model_type,
            "explain_method": explain_method,
            "checkpoint_filename": checkpoint_filename,
            "checkpoint_path": ckpt_path,
            "vocab_path": vocab_path,
            "model": model,
            "vocab": vocab,
            "tokenizer": None,
        }

    if model_type == "bilstm_attn":
        vocab_filename = spec["vocab"]
        if not vocab_filename:
            raise RuntimeError("Vocabulary filename missing for BiLSTM + Attention")
        vocab_path = resolve_vocab_path(vocab_filename)
        vocab = Vocabulary()
        vocab.load(vocab_path)

        model = BiLSTMAttention(vocab_size=len(vocab))
        load_checkpoint(ckpt_path, model)
        model.to(config.DEVICE)
        model.eval()

        return {
            "model_name": model_name,
            "model_type": model_type,
            "explain_method": explain_method,
            "checkpoint_filename": checkpoint_filename,
            "checkpoint_path": ckpt_path,
            "vocab_path": vocab_path,
            "model": model,
            "vocab": vocab,
            "tokenizer": None,
        }

    if model_type == "distilbert":
        model = DistilBertClassifier(
            model_name=config.DISTILBERT_MODEL_NAME,
            num_classes=config.NUM_CLASSES,
            finetune_strategy="full",
        )
        load_checkpoint(ckpt_path, model)
        model.to(config.DEVICE)
        model.eval()

        tokenizer = get_transformer_tokenizer(config.DISTILBERT_MODEL_NAME)
        return {
            "model_name": model_name,
            "model_type": model_type,
            "explain_method": explain_method,
            "checkpoint_filename": checkpoint_filename,
            "checkpoint_path": ckpt_path,
            "vocab_path": None,
            "model": model,
            "vocab": None,
            "tokenizer": tokenizer,
        }

    if model_type == "bert":
        model = BertClassifier(
            model_name=config.BERT_MODEL_NAME,
            num_classes=config.NUM_CLASSES,
            finetune_strategy="llrd",
        )
        load_checkpoint(ckpt_path, model)
        model.to(config.DEVICE)
        model.eval()

        tokenizer = get_transformer_tokenizer(config.BERT_MODEL_NAME)
        return {
            "model_name": model_name,
            "model_type": model_type,
            "explain_method": explain_method,
            "checkpoint_filename": checkpoint_filename,
            "checkpoint_path": ckpt_path,
            "vocab_path": None,
            "model": model,
            "vocab": None,
            "tokenizer": tokenizer,
        }

    raise RuntimeError(f"Unsupported model type: {model_type}")


def format_sentiment(rating: int) -> str:
    sentiment = {
        1: "Very Negative",
        2: "Negative",
        3: "Neutral",
        4: "Positive",
        5: "Very Positive",
    }
    return sentiment.get(rating, "Unknown")


def build_probability_plot(probs: np.ndarray):
    fig, ax = plt.subplots(figsize=(8, 3.2))
    labels = [f"Rating {r}" for r in range(1, 6)]
    colors = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#27ae60"]

    bars = ax.barh(labels, probs, color=colors)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Probability")

    for bar, p in zip(bars, probs):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2, f"{p:.3f}", va="center", fontsize=9)

    plt.tight_layout()
    return fig


def build_ig_highlight_html(tokens: np.ndarray, scores: np.ndarray) -> str:
    s_min, s_max = float(scores.min()), float(scores.max())
    if s_max - s_min > 1e-8:
        s_norm = (scores - s_min) / (s_max - s_min)
    else:
        s_norm = np.zeros_like(scores)

    spans = []
    for tok, s in zip(tokens, s_norm):
        rgba = f"rgba(255,{int(255 * (1 - s))},0,{0.30 + 0.70 * float(s):.3f})"
        spans.append(
            f"<span style='background-color:{rgba};padding:2px 4px;margin:1px;border-radius:3px;' title='ig={float(s):.4f}'>{html.escape(str(tok))}</span>"
        )

    return "<div style='line-height:2.0;font-family:Consolas,monospace;'>" + " ".join(spans) + "</div>"


def predict_and_explain(model_name: str, title: str, review_text: str):
    review_text = (review_text or "").strip()
    title = (title or "").strip()

    if not review_text:
        return (
            "Please enter review text.",
            "Please enter review text.",
            "-",
            "-",
            None,
            pd.DataFrame(columns=["Token", "IG Score"]),
            "<p>No explanation available.</p>",
            "-",
        )

    full_text = f"{title} [SEP] {review_text}" if title else review_text

    try:
        bundle = load_model_bundle(model_name)
    except Exception as exc:
        return (
            f"Failed to load model: {exc}",
            "-",
            "-",
            "-",
            None,
            pd.DataFrame(columns=["Token", "IG Score"]),
            "<p>Model loading failed. Please check model files in Hugging Face repo.</p>",
            "-",
        )

    model = bundle["model"]
    model_type = bundle["model_type"]
    explain_method = bundle["explain_method"]
    tokenizer = bundle["tokenizer"]
    vocab = bundle["vocab"]

    if model_type in {"bilstm", "bilstm_attn"}:
        pred, probs = predict_single_rnn(model, full_text, vocab)
    else:
        pred, probs = predict_single_transformer(model, full_text, tokenizer)

    predicted_rating = config.LABEL_MAP_INV[pred]
    confidence = float(probs[pred])

    prob_fig = build_probability_plot(probs)

    if explain_method == "lime":
        def rnn_predict_fn(texts):
            prob_batch = predict_batch_proba_rnn(model, texts, vocab)
            prob_batch = np.nan_to_num(
                prob_batch,
                nan=1.0 / config.NUM_CLASSES,
                posinf=1.0 / config.NUM_CLASSES,
                neginf=0.0,
            )
            row_sums = prob_batch.sum(axis=1, keepdims=True)
            row_sums[row_sums <= 0] = 1.0
            return prob_batch / row_sums

        try:
            lime_exp = lime_explain(rnn_predict_fn, full_text, num_features=10, num_samples=200)
        except Exception:
            lime_exp = None

        if lime_exp is None:
            top_df = pd.DataFrame(columns=["Token", "IG Score"])
            highlight_html = "<p>LIME explanation is unavailable for this input.</p>"
        else:
            try:
                contributions = lime_exp.as_list(label=pred)
            except Exception:
                contributions = lime_exp.as_list()

            if contributions:
                contrib_df = pd.DataFrame(contributions, columns=["Token", "IG Score"])
                contrib_df["IG Score"] = contrib_df["IG Score"].astype(float)
                contrib_df["abs_score"] = contrib_df["IG Score"].abs()
                contrib_df = contrib_df.sort_values("abs_score", ascending=False).head(15)
                top_df = contrib_df[["Token", "IG Score"]].reset_index(drop=True)

                def _render(row):
                    score = float(row["IG Score"])
                    token = html.escape(str(row["Token"]))
                    if score >= 0:
                        return f"<li><b style='color:#1e8449'>+{score:.4f}</b> {token}</li>"
                    return f"<li><b style='color:#b03a2e'>{score:.4f}</b> {token}</li>"

                items = "".join(_render(row) for _, row in top_df.iterrows())
                highlight_html = (
                    "<div><p><b>LIME token contributions</b></p>"
                    f"<ul style='margin:0 0 0 16px;padding:0'>{items}</ul></div>"
                )
            else:
                top_df = pd.DataFrame(columns=["Token", "IG Score"])
                highlight_html = "<p>No token contributions returned by LIME.</p>"
    else:
        try:
            if model_type == "bilstm_attn":
                ig = get_rnn_ig_explanation(model, full_text, vocab)
            else:
                ig = get_transformer_explanation(model, full_text, tokenizer)
        except Exception:
            ig = None

        tokens = np.array(ig["tokens"]) if ig else np.array([])
        scores = np.array(ig["scores"], dtype=float) if ig else np.array([])

        if tokens.size and scores.size:
            valid_len = min(len(tokens), len(scores))
            tokens = tokens[:valid_len]
            scores = scores[:valid_len]
            top_k = min(15, valid_len)
            top_idx = np.argsort(scores)[-top_k:][::-1]

            top_df = pd.DataFrame(
                {
                    "Token": [tokens[i] for i in top_idx],
                    "IG Score": [float(scores[i]) for i in top_idx],
                }
            )
            highlight_html = build_ig_highlight_html(tokens, scores)
        else:
            top_df = pd.DataFrame(columns=["Token", "IG Score"])
            highlight_html = "<p>No token-level IG attribution available.</p>"

    pred_md = f"### Predicted Rating: {predicted_rating} / 5"
    conf_md = f"### Confidence: {confidence:.1%}"
    senti_md = f"### Sentiment: {format_sentiment(predicted_rating)}"
    info_md = (
        f"- MODEL_REPO_ID: `{MODEL_REPO_ID}`\n"
        f"- MODEL: `{bundle['model_name']}` ({bundle['model_type']})\n"
        f"- CHECKPOINT: `{bundle['checkpoint_filename']}`\n"
        f"- EXPLAIN: `{bundle['explain_method'].upper()}`\n"
        f"- DEVICE: `{config.DEVICE}`"
    )

    return "", pred_md, conf_md, senti_md, prob_fig, top_df, highlight_html, info_md


DEFAULT_MODEL = "DistilBERT"


with gr.Blocks(title="Multi-Model Review Rating Predictor") as demo:
    gr.Markdown("# Multi-Model Review Rating Predictor")
    gr.Markdown(
        "Predict ratings (1-5) with multiple models and inspect explainability (IG/LIME). "
        "Model files are loaded from Hugging Face Hub."
    )

    with gr.Accordion("Runtime Config", open=False):
        runtime_info = gr.Markdown(
            f"- MODEL_REPO_ID: `{MODEL_REPO_ID}`\n"
            f"- MODEL: `{DEFAULT_MODEL}`\n"
            f"- DEVICE: `{config.DEVICE}`"
        )

    model_name_in = gr.Dropdown(
        label="Model",
        choices=list(MODEL_OPTIONS.keys()),
        value=DEFAULT_MODEL,
    )

    status_out = gr.Markdown()

    with gr.Row():
        title_in = gr.Textbox(label="Title (optional)", placeholder="Review title")
    review_in = gr.Textbox(
        label="Review Text",
        lines=7,
        placeholder="Type your review here...",
    )

    with gr.Row():
        predict_btn = gr.Button("Predict", variant="primary")
        clear_btn = gr.Button("Clear")

    with gr.Row():
        pred_out = gr.Markdown()
        conf_out = gr.Markdown()
        senti_out = gr.Markdown()

    prob_plot = gr.Plot(label="Class Probabilities")
    top_tokens_table = gr.Dataframe(label="Top IG Tokens", headers=["Token", "IG Score"], interactive=False)
    highlight_out = gr.HTML(label="Token Highlight")

    examples = [
        [DEFAULT_MODEL, "", "Absolutely love this dress! Perfect fit, gorgeous color, and so comfortable."],
        [DEFAULT_MODEL, "", "It is okay. Material is decent but the cut is a bit boxy."],
        [DEFAULT_MODEL, "", "Terrible quality. Size was wrong and stitching came apart quickly."],
    ]
    gr.Examples(examples=examples, inputs=[model_name_in, title_in, review_in])

    predict_btn.click(
        fn=predict_and_explain,
        inputs=[model_name_in, title_in, review_in],
        outputs=[status_out, pred_out, conf_out, senti_out, prob_plot, top_tokens_table, highlight_out, runtime_info],
    )

    clear_btn.click(
        fn=lambda: (
            "",
            "",
            "",
            "",
            None,
            pd.DataFrame(columns=["Token", "IG Score"]),
            "",
            f"- MODEL_REPO_ID: `{MODEL_REPO_ID}`\n- MODEL: `{DEFAULT_MODEL}`\n- DEVICE: `{config.DEVICE}`",
        ),
        inputs=None,
        outputs=[status_out, pred_out, conf_out, senti_out, prob_plot, top_tokens_table, highlight_out, runtime_info],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
