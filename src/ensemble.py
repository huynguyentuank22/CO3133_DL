"""
Ensemble: weighted probability averaging between RNN and Transformer models.
"""
import os
import numpy as np
import pandas as pd
from src import config
from src.metrics import compute_metrics
from src.utils import get_logger

logger = get_logger(__name__)


def weighted_ensemble(probs_rnn: np.ndarray, probs_transformer: np.ndarray,
                       alpha: float = 0.5) -> np.ndarray:
    """Compute weighted average of probability distributions.
    
    p_final = alpha * p_rnn + (1 - alpha) * p_transformer
    """
    return alpha * probs_rnn + (1 - alpha) * probs_transformer


def search_best_alpha(probs_rnn: np.ndarray, probs_transformer: np.ndarray,
                       y_true, alphas: list = config.ENSEMBLE_ALPHAS,
                       metric: str = "macro_f1") -> tuple:
    """Search for best alpha on validation/test set."""
    results = []
    for alpha in alphas:
        p_final = weighted_ensemble(probs_rnn, probs_transformer, alpha)
        preds = np.argmax(p_final, axis=1)
        metrics = compute_metrics(y_true, preds)
        metrics["alpha"] = alpha
        results.append(metrics)
        logger.info(f"Alpha {alpha:.2f}: Accuracy={metrics['accuracy']:.4f}, "
                    f"Macro-F1={metrics['macro_f1']:.4f}")

    results_df = pd.DataFrame(results)
    best_idx = results_df[metric].idxmax()
    best_alpha = results_df.loc[best_idx, "alpha"]
    best_metrics = results_df.loc[best_idx].to_dict()
    logger.info(f"Best alpha: {best_alpha} ({metric}={best_metrics[metric]:.4f})")
    return best_alpha, best_metrics, results_df


def run_ensemble(probs_rnn, probs_transformer, y_true,
                  alphas=config.ENSEMBLE_ALPHAS):
    """Run ensemble and return comparison table."""
    # Individual model performance
    rnn_preds = np.argmax(probs_rnn, axis=1)
    trans_preds = np.argmax(probs_transformer, axis=1)
    rnn_metrics = compute_metrics(y_true, rnn_preds)
    trans_metrics = compute_metrics(y_true, trans_preds)

    rnn_metrics["model"] = "RNN (BiLSTM+Attention)"
    trans_metrics["model"] = "Transformer (best)"

    # Ensemble results
    best_alpha, best_metrics, alpha_df = search_best_alpha(
        probs_rnn, probs_transformer, y_true, alphas
    )

    best_metrics["model"] = f"Ensemble (alpha={best_alpha})"

    comparison = pd.DataFrame([rnn_metrics, trans_metrics, best_metrics])
    return comparison, alpha_df, best_alpha


def save_ensemble_results(comparison_df, alpha_df, save_dir):
    """Save ensemble results to CSV."""
    os.makedirs(save_dir, exist_ok=True)
    comparison_df.to_csv(os.path.join(save_dir, "ensemble_comparison.csv"), index=False)
    alpha_df.to_csv(os.path.join(save_dir, "ensemble_alpha_search.csv"), index=False)
    logger.info(f"Ensemble results saved to {save_dir}")
