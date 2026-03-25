"""
Visualization utilities: training curves, confusion matrix, efficiency charts.
"""
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from src import config
from src.utils import get_logger

logger = get_logger(__name__)


def plot_training_curves(history: dict, model_name: str,
                         save_dir: str = config.FIGURES_DIR):
    """Plot training and validation loss/metric curves."""
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    axes[0].plot(epochs, history["train_loss"], "b-o", label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], "r-o", label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{model_name} — Loss")
    axes[0].legend()

    # Macro-F1
    axes[1].plot(epochs, history["val_accuracy"], "g-o", label="Val Accuracy")
    axes[1].plot(epochs, history["val_macro_f1"], "m-o", label="Val Macro-F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_title(f"{model_name} — Accuracy & Macro-F1")
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(save_dir, f"{model_name}_training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Training curves saved to {path}")


def plot_accuracy_vs_inference(df: pd.DataFrame,
                               save_dir: str = config.FIGURES_DIR):
    """Scatter plot: accuracy vs inference time per sample."""
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    for _, row in df.iterrows():
        ax.scatter(row["inference_time_per_sample_ms"], row["accuracy"],
                   s=120, zorder=5)
        ax.annotate(row["model_name"],
                    (row["inference_time_per_sample_ms"], row["accuracy"]),
                    textcoords="offset points", xytext=(5, 5), fontsize=9)
    ax.set_xlabel("Inference Time per Sample (ms)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs Inference Time")
    plt.tight_layout()
    path = os.path.join(save_dir, "accuracy_vs_inference_time.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Efficiency chart saved to {path}")


def plot_f1_vs_size(df: pd.DataFrame, save_dir: str = config.FIGURES_DIR):
    """Scatter plot: Macro-F1 vs model size."""
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    for _, row in df.iterrows():
        ax.scatter(row["model_size_mb"], row["macro_f1"], s=120, zorder=5)
        ax.annotate(row["model_name"],
                    (row["model_size_mb"], row["macro_f1"]),
                    textcoords="offset points", xytext=(5, 5), fontsize=9)
    ax.set_xlabel("Model Size (MB)")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Macro-F1 vs Model Size")
    plt.tight_layout()
    path = os.path.join(save_dir, "macro_f1_vs_model_size.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Efficiency chart saved to {path}")


def plot_comparison_bar(df: pd.DataFrame, metric_col: str = "macro_f1",
                        group_col: str = "model_name",
                        title: str = "Model Comparison",
                        save_path: str | None = None):
    """Bar chart comparing models on a specific metric."""
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(df[group_col], df[metric_col], color=sns.color_palette("Set2", len(df)))
    ax.set_ylabel(metric_col)
    ax.set_title(title)
    ax.set_xticklabels(df[group_col], rotation=30, ha="right")
    for bar, val in zip(bars, df[metric_col]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
