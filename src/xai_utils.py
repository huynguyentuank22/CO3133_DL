"""
XAI utilities: Attention visualization, Captum attribution, LIME explanations.
"""
import os
import html
import numpy as np
import torch
import torch.nn.functional as F
from src import config
from src.data_utils import simple_tokenize, Vocabulary
from src.utils import get_logger

logger = get_logger(__name__)


# ─── Attention Visualization (BiLSTM + Attention) ────────────────────────────

def attention_highlight_html(tokens, weights, true_label, pred_label, confidence,
                             text_id: str = ""):
    """Generate HTML with tokens colored by attention weight."""
    if len(weights) > len(tokens):
        weights = weights[:len(tokens)]
    # Normalize
    w_min, w_max = weights.min(), weights.max()
    if w_max - w_min > 1e-8:
        norm_w = (weights - w_min) / (w_max - w_min)
    else:
        norm_w = np.zeros_like(weights)

    spans = []
    for tok, w in zip(tokens, norm_w):
        r = int(255 * w)
        g = int(255 * (1 - w * 0.5))
        b = int(100)
        bg = f"rgb({r},{g},{b})"
        spans.append(f'<span style="background-color:{bg};padding:2px 4px;'
                     f'margin:1px;border-radius:3px;" '
                     f'title="weight={w:.4f}">{html.escape(tok)}</span>')

    body = " ".join(spans)
    header = (f"<p><b>ID:</b> {text_id} | "
              f"<b>True:</b> {true_label} | <b>Pred:</b> {pred_label} | "
              f"<b>Confidence:</b> {confidence:.4f}</p>")
    return f"<div style='margin:10px;font-family:monospace;'>{header}{body}</div>"


def get_attention_explanation(model, text, vocab, device=config.DEVICE,
                              max_length=config.RNN_MAX_LENGTH):
    """Get attention-based explanation for BiLSTM+Attention model."""
    model.eval()
    ids = vocab.encode(text, max_length)
    x = torch.tensor([ids], dtype=torch.long).to(device)
    with torch.no_grad():
        logits, attn_weights = model(x, return_attention=True)
    probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    pred = int(np.argmax(probs))
    tokens = simple_tokenize(text)[:max_length]
    weights = attn_weights.cpu().numpy()[0][:len(tokens)]
    top_k = min(10, len(tokens))
    top_indices = np.argsort(weights)[-top_k:][::-1]
    top_tokens = [(tokens[i], float(weights[i])) for i in top_indices]
    return {
        "pred": pred, "probs": probs, "tokens": tokens,
        "weights": weights, "top_tokens": top_tokens,
    }


def captum_integrated_gradients_rnn(model, input_ids, target_class,
                                    device=config.DEVICE, n_steps=50):
    """Compute Integrated Gradients attribution for RNN embedding inputs."""
    try:
        from captum.attr import LayerIntegratedGradients
    except ImportError:
        logger.warning("Captum not installed. Falling back to simple gradient for RNN.")
        return _simple_gradient_attribution_rnn(model, input_ids, target_class, device)

    model.eval()
    model.zero_grad()

    def forward_func(input_ids):
        return model(input_ids)

    if not hasattr(model, "embedding"):
        raise ValueError("RNN model does not expose an embedding layer")

    lig = LayerIntegratedGradients(forward_func, model.embedding)
    input_ids = input_ids.to(device)
    baseline = torch.full_like(input_ids, fill_value=getattr(model, "pad_idx", 0)).to(device)

    attributions, _ = lig.attribute(
        inputs=input_ids,
        baselines=baseline,
        target=target_class,
        n_steps=n_steps,
        return_convergence_delta=True,
    )

    attr_scores = attributions.sum(dim=-1).squeeze(0).cpu().detach().numpy()
    return attr_scores


def _simple_gradient_attribution_rnn(model, input_ids, target_class, device):
    """Fallback saliency for RNN models when Captum is unavailable."""
    model.eval()
    model.zero_grad()

    if not hasattr(model, "embedding"):
        return np.zeros(input_ids.shape[1])

    input_ids = input_ids.to(device)
    emb_out = model.embedding(input_ids)
    emb_out.retain_grad()

    embedded = model.dropout(emb_out) if hasattr(model, "dropout") else emb_out
    lstm_out, (hidden, _) = model.lstm(embedded)

    if hasattr(model, "attention"):
        pad_idx = getattr(model, "pad_idx", 0)
        mask = (input_ids != pad_idx).float()
        context, _ = model.attention(lstm_out, mask)
        features = model.dropout(context) if hasattr(model, "dropout") else context
    else:
        hidden_fwd = hidden[-2]
        hidden_bwd = hidden[-1]
        hidden_cat = torch.cat([hidden_fwd, hidden_bwd], dim=1)
        features = model.dropout(hidden_cat) if hasattr(model, "dropout") else hidden_cat

    logits = model.fc(features)
    loss = logits[0, target_class]
    loss.backward()

    if emb_out.grad is None:
        return np.zeros(input_ids.shape[1])
    return emb_out.grad.sum(dim=-1).squeeze(0).cpu().detach().numpy()


def get_rnn_ig_explanation(model, text, vocab, device=config.DEVICE,
                           max_length=config.RNN_MAX_LENGTH):
    """Get IG-based token attribution for an RNN model."""
    model.eval()
    ids = vocab.encode(text, max_length)
    input_ids = torch.tensor([ids], dtype=torch.long)

    with torch.no_grad():
        logits = model(input_ids.to(device))
    probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    pred = int(np.argmax(probs))

    attr_scores = captum_integrated_gradients_rnn(model, input_ids, pred, device)

    tokens = simple_tokenize(text)[:max_length]
    scores = attr_scores[:len(tokens)]
    abs_scores = np.abs(scores)
    if abs_scores.size > 0 and abs_scores.max() > 1e-8:
        norm_scores = abs_scores / abs_scores.max()
    else:
        norm_scores = np.zeros_like(abs_scores)

    top_k = min(10, len(tokens))
    top_indices = np.argsort(norm_scores)[-top_k:][::-1] if top_k > 0 else []
    top_tokens = [(tokens[i], float(norm_scores[i])) for i in top_indices]

    return {
        "pred": pred,
        "probs": probs,
        "tokens": tokens,
        "scores": norm_scores,
        "top_tokens": top_tokens,
    }


# ─── Captum Attribution (Transformer) ────────────────────────────────────────

def captum_integrated_gradients(model, input_ids, attention_mask, target_class,
                                 device=config.DEVICE, n_steps=50):
    """Compute Integrated Gradients attribution using Captum."""
    try:
        from captum.attr import LayerIntegratedGradients
    except ImportError:
        logger.warning("Captum not installed. Falling back to simple gradient.")
        return _simple_gradient_attribution(model, input_ids, attention_mask,
                                            target_class, device)

    model.eval()
    model.zero_grad()

    def forward_func(input_ids, attention_mask):
        logits = model(input_ids, attention_mask=attention_mask)
        return logits

    # Get embedding layer
    if hasattr(model, "backbone"):
        if hasattr(model.backbone, "embeddings"):
            emb_layer = model.backbone.embeddings
        else:
            emb_layer = model.backbone.distilbert.embeddings if hasattr(model.backbone, "distilbert") else model.backbone.embeddings
    else:
        raise ValueError("Cannot find embedding layer")

    lig = LayerIntegratedGradients(forward_func, emb_layer)

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    # Baseline: PAD token
    baseline = torch.zeros_like(input_ids).to(device)

    attributions, delta = lig.attribute(
        inputs=input_ids,
        baselines=baseline,
        target=target_class,
        additional_forward_args=(attention_mask,),
        n_steps=n_steps,
        return_convergence_delta=True,
    )
    # Sum across embedding dim
    attr_scores = attributions.sum(dim=-1).squeeze(0).cpu().detach().numpy()
    return attr_scores


def _simple_gradient_attribution(model, input_ids, attention_mask,
                                  target_class, device):
    """Fallback: simple gradient-based saliency."""
    model.eval()
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    if hasattr(model, "backbone"):
        emb = model.backbone.embeddings if hasattr(model.backbone, "embeddings") else None
    if emb is None:
        return np.zeros(input_ids.shape[1])

    emb_out = emb(input_ids)
    emb_out.retain_grad()
    logits = model.backbone(inputs_embeds=emb_out, attention_mask=attention_mask).last_hidden_state[:, 0, :]
    logits = model.dropout(logits)
    logits = model.classifier(logits)
    loss = logits[0, target_class]
    loss.backward()

    if emb_out.grad is not None:
        attr = emb_out.grad.sum(dim=-1).squeeze(0).cpu().detach().numpy()
    else:
        attr = np.zeros(input_ids.shape[1])
    return attr


def get_transformer_explanation(model, text, tokenizer, device=config.DEVICE,
                                 max_length=config.TRANSFORMER_MAX_LENGTH):
    """Get attribution-based explanation for Transformer model."""
    model.eval()
    encoding = tokenizer(text, max_length=max_length, padding="max_length",
                         truncation=True, return_tensors="pt")
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]

    with torch.no_grad():
        logits = model(input_ids.to(device), attention_mask=attention_mask.to(device))
    probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    pred = int(np.argmax(probs))

    # Get attribution scores
    attr_scores = captum_integrated_gradients(model, input_ids, attention_mask,
                                               pred, device)

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    # Only keep non-padding tokens
    mask = attention_mask[0].numpy().astype(bool)
    tokens = [t for t, m in zip(tokens, mask) if m]
    scores = attr_scores[mask]

    # Normalize for display
    abs_scores = np.abs(scores)
    if abs_scores.max() > 1e-8:
        norm_scores = abs_scores / abs_scores.max()
    else:
        norm_scores = np.zeros_like(abs_scores)

    top_k = min(10, len(tokens))
    top_indices = np.argsort(norm_scores)[-top_k:][::-1]
    top_tokens = [(tokens[i], float(norm_scores[i])) for i in top_indices]

    return {
        "pred": pred, "probs": probs, "tokens": tokens,
        "scores": norm_scores, "top_tokens": top_tokens,
    }


# ─── LIME Explanation ────────────────────────────────────────────────────────

def lime_explain(predict_fn, text: str, num_features: int = 10,
                 num_samples: int = 500):
    """Generate LIME explanation for a text sample.
    
    predict_fn: function that takes list of strings and returns 2D array of probabilities
    """
    try:
        from lime.lime_text import LimeTextExplainer
    except ImportError:
        logger.warning("LIME not installed")
        return None

    explainer = LimeTextExplainer(class_names=[f"Rating {r}" for r in range(1, 6)])
    explanation = explainer.explain_instance(
        text, predict_fn, num_features=num_features,
        num_samples=num_samples, top_labels=config.NUM_CLASSES,
    )
    return explanation


def save_xai_html(html_content: str, filepath: str):
    """Save XAI visualization to HTML file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    full_html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>XAI Results</title>
<style>body{{font-family:Arial,sans-serif;margin:20px;}}</style>
</head><body>{html_content}</body></html>"""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(full_html)
    logger.info(f"XAI HTML saved to {filepath}")
