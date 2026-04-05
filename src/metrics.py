"""
Evaluation metrics for classification.
"""
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, mean_absolute_error,
)
import matplotlib.pyplot as plt
import seaborn as sns
from src import config
from src.utils import get_logger

logger = get_logger(__name__)


def compute_metrics(y_true, y_pred) -> dict:
    """Compute classification metrics plus ordinal-aware MAE."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def get_classification_report(y_true, y_pred, target_names=None) -> str:
    """Get text classification report."""
    if target_names is None:
        target_names = [f"Rating {r}" for r in range(1, 6)]
    return classification_report(y_true, y_pred, target_names=target_names, zero_division=0)


def get_confusion_matrix(y_true, y_pred) -> np.ndarray:
    """Compute confusion matrix."""
    return confusion_matrix(y_true, y_pred)


def plot_confusion_matrix(y_true, y_pred, title: str = "Confusion Matrix",
                          save_path: str | None = None, labels=None):
    """Plot and optionally save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    if labels is None:
        labels = [str(r) for r in range(1, 6)]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Confusion matrix saved to {save_path}")
    plt.close()
    return cm


def save_metrics(metrics: dict, filepath: str):
    """Save metrics dict to JSON."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {filepath}")


def create_summary_table(all_results: list) -> pd.DataFrame:
    """Create summary comparison table from list of result dicts.
    
    Each dict should have keys like: model_name, model_family,
    imbalance_strategy, fine_tune_strategy, accuracy, macro_f1, mae, etc.
    """
    df = pd.DataFrame(all_results)
    return df
