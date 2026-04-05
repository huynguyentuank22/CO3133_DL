"""
Robustness evaluation: noise injection into test set.
"""
import random
import string
from typing import List
import pandas as pd
from src import config
from src.utils import get_logger

logger = get_logger(__name__)


def inject_typos(text: str, prob: float = 0.05, seed: int | None = None) -> str:
    """Inject character-level typos."""
    if seed is not None:
        random.seed(seed)
    chars = list(text)
    for i in range(len(chars)):
        if random.random() < prob and chars[i].isalpha():
            op = random.choice(["swap", "delete", "insert"])
            if op == "swap" and i > 0:
                chars[i], chars[i - 1] = chars[i - 1], chars[i]
            elif op == "delete":
                chars[i] = ""
            elif op == "insert":
                chars[i] = chars[i] + random.choice(string.ascii_lowercase)
    return "".join(chars)


def random_case_change(text: str, prob: float = 0.1, seed: int | None = None) -> str:
    """Randomly change case of characters."""
    if seed is not None:
        random.seed(seed)
    chars = list(text)
    for i in range(len(chars)):
        if random.random() < prob and chars[i].isalpha():
            chars[i] = chars[i].swapcase()
    return "".join(chars)


def add_punctuation_noise(text: str, prob: float = 0.05, seed: int | None = None) -> str:
    """Add random punctuation."""
    if seed is not None:
        random.seed(seed)
    words = text.split()
    new_words = []
    for word in words:
        if random.random() < prob:
            punct = random.choice([".", ",", "!", "?", ";", "..."])
            word = word + punct
        new_words.append(word)
    return " ".join(new_words)


def drop_tokens(text: str, prob: float = 0.05, seed: int | None = None) -> str:
    """Drop random tokens from text."""
    if seed is not None:
        random.seed(seed)
    words = text.split()
    if len(words) <= 3:
        return text
    remaining = [w for w in words if random.random() > prob]
    if not remaining:
        remaining = [words[0]]
    return " ".join(remaining)


def create_noisy_test_set(df: pd.DataFrame, text_col: str = "full_text",
                          seed: int = config.SEED) -> pd.DataFrame:
    """Create a noisy version of a test set."""
    noisy_df = df.copy()
    noisy_texts = []
    for idx, text in enumerate(noisy_df[text_col]):
        s = seed + idx
        text = inject_typos(str(text), prob=0.03, seed=s)
        text = random_case_change(text, prob=0.08, seed=s + 1)
        text = add_punctuation_noise(text, prob=0.03, seed=s + 2)
        text = drop_tokens(text, prob=0.03, seed=s + 3)
        noisy_texts.append(text)
    noisy_df[text_col] = noisy_texts
    logger.info(f"Created noisy test set with {len(noisy_df)} samples")
    return noisy_df


def compare_robustness(clean_metrics: dict, noisy_metrics: dict,
                        model_name: str) -> dict:
    """Compare clean vs noisy metrics.

    The drop_* columns are defined so a positive value means degradation.
    """
    result = {"model_name": model_name}
    error_metrics = {"mae", "mse", "rmse"}
    for key in clean_metrics:
        result[f"clean_{key}"] = clean_metrics[key]
        result[f"noisy_{key}"] = noisy_metrics[key]
        if key in error_metrics:
            # For error metrics, larger is worse.
            drop = noisy_metrics[key] - clean_metrics[key]
        else:
            # For score metrics, smaller is worse.
            drop = clean_metrics[key] - noisy_metrics[key]
        result[f"drop_{key}"] = round(drop, 4)
    return result
