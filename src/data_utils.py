"""
Data utilities: vocabulary building for RNN, tokenizer wrappers for Transformers.
"""
import os
import json
import re
from collections import Counter
from typing import List, Dict
import pandas as pd
from transformers import AutoTokenizer
from src import config
from src.utils import get_logger

logger = get_logger(__name__)

# ─── RNN Vocabulary ──────────────────────────────────────────────────────────
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_IDX = 0
UNK_IDX = 1


def simple_tokenize(text: str, lowercase: bool = config.LOWERCASE_RNN) -> List[str]:
    """Simple whitespace + punctuation tokenizer."""
    if lowercase:
        text = text.lower()
    # Split on whitespace and basic punctuation
    tokens = re.findall(r"\b\w+\b", text)
    return tokens


class Vocabulary:
    """Word-to-index vocabulary for RNN models."""

    def __init__(self):
        self.word2idx: Dict[str, int] = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
        self.idx2word: Dict[int, str] = {PAD_IDX: PAD_TOKEN, UNK_IDX: UNK_TOKEN}
        self.word_freq: Counter = Counter()

    def build_from_texts(self, texts: List[str],
                         max_vocab_size: int = config.RNN_MAX_VOCAB_SIZE):
        """Build vocabulary from a list of texts (training set only)."""
        for text in texts:
            tokens = simple_tokenize(text)
            self.word_freq.update(tokens)

        most_common = self.word_freq.most_common(max_vocab_size - 2)  # -2 for PAD, UNK
        for word, _ in most_common:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

        logger.info(f"Vocabulary built: {len(self.word2idx)} tokens")

    def encode(self, text: str, max_length: int = config.RNN_MAX_LENGTH) -> List[int]:
        """Encode text to sequence of indices with padding/truncation."""
        tokens = simple_tokenize(text)
        ids = [self.word2idx.get(t, UNK_IDX) for t in tokens]
        # Truncate
        if len(ids) > max_length:
            ids = ids[:max_length]
        # Pad
        while len(ids) < max_length:
            ids.append(PAD_IDX)
        return ids

    def __len__(self):
        return len(self.word2idx)

    def save(self, filepath: str):
        """Save vocabulary to JSON."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.word2idx, f, ensure_ascii=False)

    def load(self, filepath: str):
        """Load vocabulary from JSON."""
        with open(filepath, "r", encoding="utf-8") as f:
            self.word2idx = json.load(f)
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        logger.info(f"Vocabulary loaded: {len(self.word2idx)} tokens")


# ─── Transformer Tokenizer ───────────────────────────────────────────────────

def get_transformer_tokenizer(model_name: str):
    """Get a HuggingFace tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer


def tokenize_for_transformer(texts: List[str], tokenizer,
                             max_length: int = config.TRANSFORMER_MAX_LENGTH):
    """Tokenize texts for transformer models."""
    encoding = tokenizer(
        texts,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return encoding
