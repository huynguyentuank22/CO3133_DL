"""
Text augmentation: synonym replacement and random deletion.
"""
import random
import re
import nltk
from typing import List
from src import config
from src.utils import get_logger

logger = get_logger(__name__)

# Download WordNet if not present
try:
    from nltk.corpus import wordnet
    wordnet.synsets("test")
except LookupError:
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    from nltk.corpus import wordnet


def get_synonyms(word: str) -> List[str]:
    """Get synonyms for a word from WordNet."""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            if lemma.name().lower() != word.lower() and "_" not in lemma.name():
                synonyms.add(lemma.name().lower())
    return list(synonyms)


def synonym_replacement(text: str, prob: float = config.AUGMENTATION_SYNONYM_PROB,
                         seed: int | None = None) -> str:
    """Replace words with synonyms with given probability."""
    if seed is not None:
        random.seed(seed)
    words = text.split()
    new_words = []
    for word in words:
        if random.random() < prob and word.isalpha():
            synonyms = get_synonyms(word.lower())
            if synonyms:
                new_words.append(random.choice(synonyms))
            else:
                new_words.append(word)
        else:
            new_words.append(word)
    return " ".join(new_words)


def random_deletion(text: str, prob: float = config.AUGMENTATION_DELETE_PROB,
                     seed: int | None = None) -> str:
    """Randomly delete words with given probability (keep at least 1 word)."""
    if seed is not None:
        random.seed(seed)
    words = text.split()
    if len(words) <= 1:
        return text
    remaining = [w for w in words if random.random() > prob]
    if not remaining:
        remaining = [random.choice(words)]
    return " ".join(remaining)


def augment_text(text: str, methods: List[str] = None,
                 seed: int | None = None) -> str:
    """Apply augmentation pipeline to a text sample."""
    if methods is None:
        methods = ["synonym", "deletion"]
    result = text
    for method in methods:
        if method == "synonym":
            result = synonym_replacement(result, seed=seed)
        elif method == "deletion":
            result = random_deletion(result, seed=seed)
    return result


def augment_dataframe(df, text_col: str = "full_text", label_col: str = "label",
                       methods: List[str] = None, seed: int = config.SEED):
    """Augment texts in a DataFrame and return combined dataset."""
    import pandas as pd
    augmented_rows = []
    for idx, row in df.iterrows():
        aug_text = augment_text(row[text_col], methods=methods,
                                seed=seed + idx)
        new_row = row.copy()
        new_row[text_col] = aug_text
        augmented_rows.append(new_row)

    aug_df = pd.DataFrame(augmented_rows)
    combined = pd.concat([df, aug_df], ignore_index=True)
    combined = combined.sample(frac=1, random_state=seed).reset_index(drop=True)
    logger.info(f"Augmentation: {len(df)} → {len(combined)} samples")
    return combined
