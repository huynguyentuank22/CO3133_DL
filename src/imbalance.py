"""
Class imbalance handling: weighted CE, undersampling, distribution visualization.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from collections import Counter
from src import config
from src.utils import get_logger

logger = get_logger(__name__)


def compute_class_weights(labels, num_classes: int = config.NUM_CLASSES) -> torch.Tensor:
    """Compute inverse-frequency class weights for CrossEntropyLoss."""
    counter = Counter(labels)
    total = sum(counter.values())
    weights = []
    for c in range(num_classes):
        count = counter.get(c, 1)
        weights.append(total / (num_classes * count))
    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    logger.info(f"Class weights: {weights_tensor.tolist()}")
    return weights_tensor


def random_undersample(df: pd.DataFrame, label_col: str = "label",
                       seed: int = config.SEED) -> pd.DataFrame:
    """Undersample majority classes to match the smallest class count."""
    counter = Counter(df[label_col])
    min_count = min(counter.values())
    logger.info(f"Undersampling to {min_count} samples per class (min class count)")

    dfs = []
    for label in sorted(counter.keys()):
        class_df = df[df[label_col] == label]
        sampled = class_df.sample(n=min_count, random_state=seed)
        dfs.append(sampled)

    result = pd.concat(dfs, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    logger.info(f"After undersampling: {len(result)} total samples")
    return result


def get_loss_function(strategy: str = config.IMBALANCE_STRATEGY,
                      train_labels=None, device=config.DEVICE):
    """Return CrossEntropyLoss based on imbalance strategy."""
    if strategy == "weighted_ce" and train_labels is not None:
        weights = compute_class_weights(train_labels).to(device)
        return torch.nn.CrossEntropyLoss(weight=weights)
    else:
        return torch.nn.CrossEntropyLoss()


def log_class_distribution(labels, name: str = ""):
    """Log and return class distribution."""
    counter = Counter(labels)
    dist = {config.LABEL_MAP_INV.get(k, k): v for k, v in sorted(counter.items())}
    logger.info(f"Class distribution [{name}]: {dist}")
    return dist


def plot_class_distributions(distributions: dict, title: str = "Class Distribution",
                             save_path: str | None = None):
    """Plot bar charts comparing class distributions.
    
    distributions: dict of {name: {class_label: count}}
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(config.NUM_CLASSES)
    width = 0.8 / len(distributions)

    for i, (name, dist) in enumerate(distributions.items()):
        counts = [dist.get(r, 0) for r in range(1, 6)]
        ax.bar(x + i * width, counts, width, label=name, alpha=0.85)

    ax.set_xlabel("Rating")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.set_xticks(x + width * (len(distributions) - 1) / 2)
    ax.set_xticklabels([str(r) for r in range(1, 6)])
    ax.legend()
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Distribution plot saved to {save_path}")
    plt.close()
