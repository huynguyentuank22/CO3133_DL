"""
Streamlit demo app for Rating Prediction.
Usage: streamlit run app/streamlit_app.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import streamlit as st
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src import config
from src.utils import set_seed, load_checkpoint
from src.data_utils import Vocabulary, get_transformer_tokenizer
from src.rnn_models import BiLSTMAttention
from src.transformer_models import DistilBertClassifier, BertClassifier
from src.infer import (
    predict_single_rnn_attention, predict_single_transformer,
)

set_seed()
config.ensure_dirs()

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Rating Predictor", page_icon="⭐", layout="wide")
st.title("⭐ Review Rating Predictor")
st.markdown("Predict ratings (1-5) from clothing reviews using multiple ML models.")

# ─── Sidebar ─────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Configuration")

MODEL_OPTIONS = {
    "BiLSTM + Attention (weighted_ce)": ("rnn", "bilstm_attention_weighted_ce"),
    "BiLSTM + Attention (undersample_ce)": ("rnn", "bilstm_attention_undersample_ce"),
    "DistilBERT (full, weighted_ce)": ("distilbert", "distilbert_full_weighted_ce"),
    "DistilBERT (freeze, weighted_ce)": ("distilbert", "distilbert_freeze_weighted_ce"),
    "DistilBERT (llrd, weighted_ce)": ("distilbert", "distilbert_llrd_weighted_ce"),
    "BERT-base (full, weighted_ce)": ("bert", "bert_full_weighted_ce"),
    "BERT-base (full, undersample_ce)": ("bert", "bert_full_undersample_ce"),
}

# Filter to only available checkpoints
available_models = {}
for display_name, (model_type, ckpt_name) in MODEL_OPTIONS.items():
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"{ckpt_name}_best.pt")
    if os.path.exists(ckpt_path):
        available_models[display_name] = (model_type, ckpt_name)

if not available_models:
    st.error("No model checkpoints found. Please train at least one model first.")
    st.stop()

selected_model = st.sidebar.selectbox("Select Model", list(available_models.keys()))
model_type, ckpt_name = available_models[selected_model]

# ─── Example Reviews ─────────────────────────────────────────────────────────
st.sidebar.header("📝 Example Reviews")
examples = {
    "Very Positive (5)": "Absolutely love this dress! Perfect fit, gorgeous color, "
                         "and so comfortable. I've gotten tons of compliments!",
    "Positive (4)": "Nice quality top, fits well. The color was slightly different "
                    "from the photo but still pretty. Would recommend.",
    "Neutral (3)": "It's okay. Nothing special about this shirt. The material is "
                   "decent but the cut is a bit boxy for my taste.",
    "Negative (2)": "Disappointed with the quality. The fabric feels cheap and the "
                    "stitching is already coming apart after one wash.",
    "Very Negative (1)": "Terrible! The sizing is completely wrong, the material "
                         "is see-through, and it fell apart immediately. Total waste of money.",
}
selected_example = st.sidebar.selectbox("Or try an example:", ["(none)"] + list(examples.keys()))


# ─── Load Model ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(model_type, ckpt_name):
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"{ckpt_name}_best.pt")

    if model_type == "rnn":
        vocab = Vocabulary()
        vocab_path = os.path.join(config.DATA_PROCESSED_DIR, "vocab.json")
        vocab.load(vocab_path)
        model = BiLSTMAttention(vocab_size=len(vocab))
        load_checkpoint(ckpt_path, model)
        model.to(config.DEVICE)
        model.eval()
        return model, vocab, None

    elif model_type == "distilbert":
        model = DistilBertClassifier()
        load_checkpoint(ckpt_path, model)
        model.to(config.DEVICE)
        model.eval()
        tokenizer = get_transformer_tokenizer(config.DISTILBERT_MODEL_NAME)
        return model, None, tokenizer

    elif model_type == "bert":
        model = BertClassifier()
        load_checkpoint(ckpt_path, model)
        model.to(config.DEVICE)
        model.eval()
        tokenizer = get_transformer_tokenizer(config.BERT_MODEL_NAME)
        return model, None, tokenizer


model, vocab, tokenizer = load_model(model_type, ckpt_name)

# ─── Main Area ────────────────────────────────────────────────────────────────
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Enter your review")
    if selected_example != "(none)":
        default_text = examples[selected_example]
    else:
        default_text = ""

    review_text = st.text_area("Review Text", value=default_text, height=150,
                               placeholder="Type your review here...")
    title = st.text_input("Title (optional)", value="",
                          placeholder="Review title...")

    predict_btn = st.button("🔮 Predict Rating", type="primary", use_container_width=True)

with col2:
    st.subheader("Model Info")
    st.info(f"**Model:** {selected_model}\n\n"
            f"**Device:** {config.DEVICE}")

# ─── Prediction ───────────────────────────────────────────────────────────────
if predict_btn and review_text:
    # Build full_text
    full_text = f"{title} [SEP] {review_text}" if title else review_text

    with st.spinner("Predicting..."):
        if model_type == "rnn":
            pred, probs, tokens, weights = predict_single_rnn_attention(
                model, full_text, vocab)
        else:
            pred, probs = predict_single_transformer(model, full_text, tokenizer)
            tokens, weights = None, None

    predicted_rating = config.LABEL_MAP_INV[pred]
    confidence = float(probs[pred])

    # ─── Results ──────────────────────────────────────────────────────────
    st.divider()
    res_col1, res_col2, res_col3 = st.columns(3)
    with res_col1:
        stars = "⭐" * predicted_rating
        st.metric("Predicted Rating", f"{predicted_rating} / 5 {stars}")
    with res_col2:
        st.metric("Confidence", f"{confidence:.1%}")
    with res_col3:
        sentiment = {1: "😡 Very Negative", 2: "😞 Negative", 3: "😐 Neutral",
                     4: "😊 Positive", 5: "😍 Very Positive"}
        st.metric("Sentiment", sentiment[predicted_rating])

    # ─── Probability Distribution ─────────────────────────────────────────
    st.subheader("Class Probabilities")
    prob_df = pd.DataFrame({
        "Rating": [f"⭐ {r}" for r in range(1, 6)],
        "Probability": probs,
    })

    fig, ax = plt.subplots(figsize=(8, 3))
    colors = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#27ae60"]
    bars = ax.barh(prob_df["Rating"], prob_df["Probability"], color=colors)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability")
    for bar, p in zip(bars, probs):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{p:.3f}", va="center", fontsize=10)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ─── Token Importance (Attention) ─────────────────────────────────────
    if tokens is not None and weights is not None:
        st.subheader("🔍 Token Importance (Attention)")

        # Show top tokens
        valid_len = min(len(tokens), len(weights))
        tokens = tokens[:valid_len]
        weights = weights[:valid_len]

        top_k = min(15, valid_len)
        top_idx = np.argsort(weights)[-top_k:][::-1]

        top_df = pd.DataFrame({
            "Token": [tokens[i] for i in top_idx],
            "Attention Weight": [f"{weights[i]:.4f}" for i in top_idx],
        })
        st.table(top_df)

        # Highlighted text
        w_norm = (weights - weights.min()) / (weights.max() - weights.min() + 1e-8)
        highlighted = " ".join(
            f'<span style="background-color:rgba(255,{int(255*(1-w))},0,{0.3+0.7*w});'
            f'padding:2px 4px;border-radius:3px;">{tok}</span>'
            for tok, w in zip(tokens, w_norm)
        )
        st.markdown(f"<div style='line-height:2;'>{highlighted}</div>",
                    unsafe_allow_html=True)

elif predict_btn and not review_text:
    st.warning("Please enter a review text to predict.")
