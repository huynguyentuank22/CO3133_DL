"""
Error analysis: confusion pairs, misclassified samples, error categorization.
"""
import os
import pandas as pd
import numpy as np
from collections import Counter
from src import config
from src.metrics import get_confusion_matrix
from src.utils import get_logger

logger = get_logger(__name__)


def get_top_confusion_pairs(y_true, y_pred, top_k: int = 5):
    """Find top confused class pairs (true→pred) with counts."""
    cm = get_confusion_matrix(y_true, y_pred)
    pairs = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i != j and cm[i, j] > 0:
                pairs.append({
                    "true_label": config.LABEL_MAP_INV.get(i, i),
                    "pred_label": config.LABEL_MAP_INV.get(j, j),
                    "count": int(cm[i, j]),
                })
    pairs.sort(key=lambda x: x["count"], reverse=True)
    return pairs[:top_k]


def extract_misclassified(df: pd.DataFrame, y_true, y_pred, n: int = 50):
    """Extract misclassified samples with metadata."""
    df = df.copy()
    df["true_label"] = [config.LABEL_MAP_INV.get(y, y) for y in y_true]
    df["pred_label"] = [config.LABEL_MAP_INV.get(y, y) for y in y_pred]
    df["correct"] = df["true_label"] == df["pred_label"]
    misclassified = df[~df["correct"]].copy()
    # Sort by most common confusion pair
    misclassified = misclassified.sort_values(by=["true_label", "pred_label"])
    return misclassified.head(n)


def categorize_errors(misclassified_df: pd.DataFrame) -> pd.DataFrame:
    """Heuristic error categorization."""
    df = misclassified_df.copy()
    categories = []
    for _, row in df.iterrows():
        text = str(row.get("full_text", ""))
        true_l = row["true_label"]
        pred_l = row["pred_label"]
        word_count = len(text.split())

        if abs(true_l - pred_l) == 1:
            cat = "subtle_rating_difference"
        elif word_count < 15:
            cat = "short_review"
        elif word_count > 200:
            cat = "long_noisy_review"
        elif any(w in text.lower() for w in ["but", "however", "although", "except", "yet"]):
            cat = "mixed_sentiment"
        else:
            cat = "ambiguous_review"
        categories.append(cat)

    df["error_category"] = categories
    return df


def compare_errors_by_strategy(results_weighted: dict, results_undersample: dict,
                                test_df: pd.DataFrame):
    """Compare misclassification patterns between two imbalance strategies."""
    comparisons = []
    for label_idx in range(config.NUM_CLASSES):
        rating = config.LABEL_MAP_INV[label_idx]
        mask_true = np.array(results_weighted["y_true"]) == label_idx

        # Weighted CE errors
        weighted_wrong = np.sum(
            (np.array(results_weighted["y_true"]) == label_idx) &
            (np.array(results_weighted["y_pred"]) != label_idx)
        )
        weighted_total = np.sum(mask_true)

        # Undersample CE errors
        under_wrong = np.sum(
            (np.array(results_undersample["y_true"]) == label_idx) &
            (np.array(results_undersample["y_pred"]) != label_idx)
        )

        comparisons.append({
            "rating": rating,
            "total_samples": int(weighted_total),
            "errors_weighted_ce": int(weighted_wrong),
            "error_rate_weighted_ce": round(weighted_wrong / max(weighted_total, 1), 4),
            "errors_undersample_ce": int(under_wrong),
            "error_rate_undersample_ce": round(under_wrong / max(weighted_total, 1), 4),
        })

    return pd.DataFrame(comparisons)


def generate_error_report(confusion_pairs, error_categories, comparison_df=None):
    """Generate markdown error analysis report."""
    lines = ["# Error Analysis Report\n"]

    lines.append("## Top Confusion Pairs\n")
    lines.append("| True Rating | Predicted Rating | Count |")
    lines.append("|-------------|------------------|-------|")
    for pair in confusion_pairs:
        lines.append(f"| {pair['true_label']} | {pair['pred_label']} | {pair['count']} |")

    lines.append("\n## Error Categories\n")
    cat_counts = Counter(error_categories["error_category"])
    lines.append("| Category | Count |")
    lines.append("|----------|-------|")
    for cat, count in cat_counts.most_common():
        lines.append(f"| {cat} | {count} |")

    if comparison_df is not None:
        lines.append("\n## Error Comparison: Weighted CE vs Undersample CE\n")
        lines.append(comparison_df.to_markdown(index=False))

    return "\n".join(lines)
