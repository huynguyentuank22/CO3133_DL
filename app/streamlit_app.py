"""
Streamlit demo app for Rating Prediction.
Usage: streamlit run app/streamlit_app.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import streamlit as st
import streamlit.components.v1 as components
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src import config
from src.utils import set_seed, load_checkpoint
from src.data_utils import get_transformer_tokenizer, load_rnn_vocab, resolve_rnn_vocab_path
from src.rnn_models import BiLSTM, BiLSTMAttention
from src.transformer_models import DistilBertClassifier, BertClassifier
from src.infer import (
    predict_single_rnn, predict_batch_proba_rnn, predict_single_transformer,
)
from src.xai_utils import get_rnn_ig_explanation, get_transformer_explanation, lime_explain

set_seed()
config.ensure_dirs()

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Rating Predictor", page_icon="⭐", layout="wide")
st.title("⭐ Review Rating Predictor")
st.markdown("Predict ratings (1-5) from clothing reviews using multiple models.")

# ─── Sidebar ─────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Configuration")

MODEL_OPTIONS = {
    "BERT-base": ("bert", "bert_llrd_weighted_ce", "ig"),
    "DistilBERT": ("distilbert", "distilbert_full_weighted_ce", "ig"),
    "BiLSTM + Attention": ("bilstm_attn", "bilstm_attention_weighted_ce", "ig"),
    "BiLSTM": ("bilstm", "bilstm_weighted_ce", "lime"),
}

# Filter to only available checkpoints
available_models = {}
for display_name, (model_type, ckpt_name, explain_method) in MODEL_OPTIONS.items():
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"{ckpt_name}_best.pt")
    if not os.path.exists(ckpt_path):
        continue
    if model_type in {"bilstm", "bilstm_attn"} and resolve_rnn_vocab_path(ckpt_name) is None:
        continue
    if os.path.exists(ckpt_path):
        available_models[display_name] = (model_type, ckpt_name, explain_method)

if not available_models:
    st.error("No model checkpoints found. Please train at least one model first.")
    st.stop()

selected_model = st.sidebar.selectbox("Select Model", list(available_models.keys()))
model_type, ckpt_name, explain_method = available_models[selected_model]

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

    if model_type == "bilstm":
        vocab, vocab_path = load_rnn_vocab(ckpt_name)
        if vocab is None:
            raise FileNotFoundError(f"No vocabulary found for {ckpt_name}")
        model = BiLSTM(vocab_size=len(vocab))
        load_checkpoint(ckpt_path, model)
        model.to(config.DEVICE)
        model.eval()
        return model, vocab, None

    elif model_type == "bilstm_attn":
        vocab, vocab_path = load_rnn_vocab(ckpt_name)
        if vocab is None:
            raise FileNotFoundError(f"No vocabulary found for {ckpt_name}")
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
            f"**Explain:** {explain_method.upper()}\n\n"
            f"**Device:** {config.DEVICE}")

# ─── Prediction ───────────────────────────────────────────────────────────────
if predict_btn and review_text:
    # Build full_text
    full_text = f"{title} [SEP] {review_text}" if title else review_text

    with st.spinner("Predicting..."):
        if model_type in {"bilstm", "bilstm_attn"}:
            pred, probs = predict_single_rnn(model, full_text, vocab)
        else:
            pred, probs = predict_single_transformer(model, full_text, tokenizer)

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

    # ─── Explainability ─────────────────────────────────────────────────────
    st.subheader("🔍 Explainability")

    if explain_method == "lime":
        with st.spinner("Generating LIME explanation..."):
            try:
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

                lime_exp = lime_explain(rnn_predict_fn, full_text, num_features=8, num_samples=200)
            except Exception as e:
                lime_exp = None
                st.warning(f"LIME explanation failed: {e}")

        if lime_exp is None:
            st.info("LIME explanation is unavailable for this input.")
        else:
            try:
                contributions = lime_exp.as_list(label=pred)
            except Exception:
                contributions = lime_exp.as_list()

            if not contributions:
                st.info("No token contributions returned by LIME.")
            else:
                contrib_df = pd.DataFrame(contributions, columns=["Token", "Contribution"])
                contrib_df["Abs Contribution"] = contrib_df["Contribution"].abs()
                contrib_df = contrib_df.sort_values("Abs Contribution", ascending=False)

                pos_df = contrib_df[contrib_df["Contribution"] > 0].head(8)
                neg_df = contrib_df[contrib_df["Contribution"] < 0].head(8)

                c_pos, c_neg = st.columns(2)
                with c_pos:
                    st.markdown("**Positive toward predicted class**")
                    st.table(pos_df[["Token", "Contribution"]] if not pos_df.empty else pd.DataFrame({"Token": [], "Contribution": []}))
                with c_neg:
                    st.markdown("**Negative toward predicted class**")
                    st.table(neg_df[["Token", "Contribution"]] if not neg_df.empty else pd.DataFrame({"Token": [], "Contribution": []}))

                with st.expander("Show full LIME HTML"):
                    components.html(lime_exp.as_html(), height=620, scrolling=True)

    else:
        with st.spinner("Generating IG explanation..."):
            try:
                if model_type == "bilstm_attn":
                    ig_exp = get_rnn_ig_explanation(model, full_text, vocab)
                else:
                    ig_exp = get_transformer_explanation(model, full_text, tokenizer)
            except Exception as e:
                ig_exp = None
                st.warning(f"IG explanation failed: {e}")

        if ig_exp is None:
            st.info("IG explanation is unavailable for this input.")
        else:
            tokens = np.array(ig_exp["tokens"])
            scores = np.array(ig_exp["scores"], dtype=float)

            if tokens.size == 0 or scores.size == 0:
                st.info("No token-level attribution available for this input.")
            else:
                valid_len = min(len(tokens), len(scores))
                tokens = tokens[:valid_len]
                scores = scores[:valid_len]

                top_k = min(15, valid_len)
                top_idx = np.argsort(scores)[-top_k:][::-1]

                top_df = pd.DataFrame({
                    "Token": [tokens[i] for i in top_idx],
                    "IG Score": [f"{scores[i]:.4f}" for i in top_idx],
                })
                st.table(top_df)

                s_min, s_max = scores.min(), scores.max()
                if s_max - s_min > 1e-8:
                    s_norm = (scores - s_min) / (s_max - s_min)
                else:
                    s_norm = np.zeros_like(scores)

                highlighted = " ".join(
                    f'<span style="background-color:rgba(255,{int(255*(1-s))},0,{0.3+0.7*s});'
                    f'padding:2px 4px;border-radius:3px;">{tok}</span>'
                    for tok, s in zip(tokens, s_norm)
                )
                st.markdown(f"<div style='line-height:2;'>{highlighted}</div>", unsafe_allow_html=True)

elif predict_btn and not review_text:
    st.warning("Please enter a review text to predict.")
