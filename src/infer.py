"""
Inference utilities: single sample and batch prediction.
"""
import torch
import torch.nn.functional as F
import numpy as np
from src import config
from src.data_utils import Vocabulary, simple_tokenize, get_transformer_tokenizer


def predict_single_rnn(model, text: str, vocab: Vocabulary,
                       device=config.DEVICE, max_length=config.RNN_MAX_LENGTH):
    """Predict class and probabilities for a single text (RNN model)."""
    model.eval()
    ids = vocab.encode(text, max_length)
    x = torch.tensor([ids], dtype=torch.long).to(device)
    with torch.no_grad():
        logits = model(x)
    probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    pred = int(np.argmax(probs))
    return pred, probs


def predict_single_rnn_attention(model, text: str, vocab: Vocabulary,
                                 device=config.DEVICE, max_length=config.RNN_MAX_LENGTH):
    """Predict with attention weights for BiLSTM+Attention."""
    model.eval()
    ids = vocab.encode(text, max_length)
    x = torch.tensor([ids], dtype=torch.long).to(device)
    with torch.no_grad():
        logits, attn_weights = model(x, return_attention=True)
    probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    pred = int(np.argmax(probs))
    weights = attn_weights.cpu().numpy()[0]
    tokens = simple_tokenize(text)[:max_length]
    return pred, probs, tokens, weights[:len(tokens)]


def predict_single_transformer(model, text: str, tokenizer,
                               device=config.DEVICE,
                               max_length=config.TRANSFORMER_MAX_LENGTH):
    """Predict class and probabilities for a single text (Transformer model)."""
    model.eval()
    encoding = tokenizer(text, max_length=max_length, padding="max_length",
                         truncation=True, return_tensors="pt")
    input_ids = encoding["input_ids"].to(device)
    attn_mask = encoding["attention_mask"].to(device)
    with torch.no_grad():
        logits = model(input_ids, attention_mask=attn_mask)
    probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    pred = int(np.argmax(probs))
    return pred, probs


def predict_batch_proba_rnn(model, texts, vocab: Vocabulary,
                            device=config.DEVICE, batch_size=64,
                            max_length=config.RNN_MAX_LENGTH) -> np.ndarray:
    """Get softmax probabilities for a batch of texts (RNN)."""
    model.eval()
    all_probs = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        ids = [vocab.encode(t, max_length) for t in batch_texts]
        x = torch.tensor(ids, dtype=torch.long).to(device)
        with torch.no_grad():
            logits = model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
    return np.vstack(all_probs)


def predict_batch_proba_transformer(model, texts, tokenizer,
                                    device=config.DEVICE, batch_size=16,
                                    max_length=config.TRANSFORMER_MAX_LENGTH) -> np.ndarray:
    """Get softmax probabilities for a batch of texts (Transformer)."""
    model.eval()
    all_probs = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        encoding = tokenizer(batch_texts, max_length=max_length, padding="max_length",
                             truncation=True, return_tensors="pt")
        input_ids = encoding["input_ids"].to(device)
        attn_mask = encoding["attention_mask"].to(device)
        with torch.no_grad():
            logits = model(input_ids, attention_mask=attn_mask)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
    return np.vstack(all_probs)
