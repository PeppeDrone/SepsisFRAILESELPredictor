"""
Evaluation metrics module.

Comprehensive metrics for binary classification:
- Accuracy, Sensitivity, Specificity, F1-score
- ROC-AUC, PR-AUC
- Brier score
- Confusion matrix
"""

import logging
from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from sepsis_ml import CLASSIFICATION_THRESHOLD

logger = logging.getLogger(__name__)


def calculate_sensitivity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate sensitivity (recall for positive class).

    Sensitivity = TP / (TP + FN)

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels

    Returns
    -------
    float
        Sensitivity score
    """
    return recall_score(y_true, y_pred, pos_label=1, zero_division=0)


def calculate_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate specificity (recall for negative class).

    Specificity = TN / (TN + FP)

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels

    Returns
    -------
    float
        Specificity score
    """
    return recall_score(y_true, y_pred, pos_label=0, zero_division=0)


def calculate_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = CLASSIFICATION_THRESHOLD,
) -> Dict[str, float]:
    """
    Calculate all evaluation metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels (or will be thresholded from y_proba)
    y_proba : np.ndarray
        Predicted probabilities for positive class
    threshold : float
        Classification threshold (default: 0.5)

    Returns
    -------
    dict
        Dictionary of metric names and values
    """
    # Ensure predictions are thresholded consistently
    if y_pred is None or len(y_pred) == 0:
        y_pred = (y_proba >= threshold).astype(int)

    metrics = {}

    # Basic metrics
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["sensitivity"] = calculate_sensitivity(y_true, y_pred)
    metrics["specificity"] = calculate_specificity(y_true, y_pred)
    metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)

    # Probabilistic metrics
    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
    except ValueError as e:
        logger.warning(f"Could not calculate ROC-AUC: {e}")
        metrics["roc_auc"] = np.nan

    try:
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        metrics["pr_auc"] = auc(recall, precision)
    except ValueError as e:
        logger.warning(f"Could not calculate PR-AUC: {e}")
        metrics["pr_auc"] = np.nan

    try:
        metrics["brier"] = brier_score_loss(y_true, y_proba)
    except ValueError as e:
        logger.warning(f"Could not calculate Brier score: {e}")
        metrics["brier"] = np.nan

    # Confusion matrix elements
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics["tn"] = tn
        metrics["fp"] = fp
        metrics["fn"] = fn
        metrics["tp"] = tp
    else:
        # Handle edge case where only one class present
        logger.warning(f"Confusion matrix has unexpected shape: {cm.shape}")
        metrics["tn"] = 0
        metrics["fp"] = 0
        metrics["fn"] = 0
        metrics["tp"] = 0

    metrics["threshold"] = threshold

    return metrics


def calculate_roc_curve_data(
    y_true: np.ndarray,
    y_proba: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate ROC curve data.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_proba : np.ndarray
        Predicted probabilities

    Returns
    -------
    fpr : np.ndarray
        False positive rates
    tpr : np.ndarray
        True positive rates
    thresholds : np.ndarray
        Thresholds
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    return fpr, tpr, thresholds


def calculate_pr_curve_data(
    y_true: np.ndarray,
    y_proba: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate Precision-Recall curve data.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_proba : np.ndarray
        Predicted probabilities

    Returns
    -------
    precision : np.ndarray
        Precision values
    recall : np.ndarray
        Recall values
    thresholds : np.ndarray
        Thresholds
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    return precision, recall, thresholds


def calculate_calibration_curve_data(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate calibration curve data (fraction of positives vs mean predicted probability).

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_proba : np.ndarray
        Predicted probabilities
    n_bins : int
        Number of bins for calibration curve

    Returns
    -------
    prob_true : np.ndarray
        Fraction of positives in each bin
    prob_pred : np.ndarray
        Mean predicted probability in each bin
    """
    from sklearn.calibration import calibration_curve

    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy="uniform")
    return prob_true, prob_pred


class MajorityClassBaseline:
    """
    Baseline classifier that always predicts the majority class.

    Used to contextualize model performance relative to a trivial strategy.
    """

    def __init__(self):
        self.majority_class_ = None

    def fit(self, y: np.ndarray):
        """Fit by finding the majority class."""
        classes, counts = np.unique(y, return_counts=True)
        self.majority_class_ = classes[np.argmax(counts)]
        logger.debug(f"Majority class baseline: always predict class {self.majority_class_}")
        return self

    def predict(self, n_samples: int) -> np.ndarray:
        """Predict majority class for all samples."""
        if self.majority_class_ is None:
            raise ValueError("Baseline not fitted yet.")
        return np.full(n_samples, self.majority_class_)

    def predict_proba(self, n_samples: int) -> np.ndarray:
        """
        Return majority class probability.

        For compatibility with metrics that require probabilities.
        """
        if self.majority_class_ is None:
            raise ValueError("Baseline not fitted yet.")

        # Return probability 1.0 for majority class
        if self.majority_class_ == 1:
            return np.ones(n_samples)
        else:
            return np.zeros(n_samples)


def aggregate_metrics_across_folds(
    fold_metrics: list,
    ci: float = 0.95,
) -> Dict[str, Tuple[float, float, float]]:
    """
    Aggregate metrics across folds with confidence intervals.

    Parameters
    ----------
    fold_metrics : list
        List of metric dictionaries from each fold
    ci : float
        Confidence interval level (default: 0.95 for 95% CI)

    Returns
    -------
    dict
        Dictionary with metric_name: (mean, lower_ci, upper_ci)
    """
    import scipy.stats as stats

    aggregated = {}

    # Get all metric names
    metric_names = fold_metrics[0].keys()

    for metric_name in metric_names:
        if metric_name in ["tn", "fp", "fn", "tp", "threshold"]:
            # Skip confusion matrix elements and threshold for aggregation
            continue

        values = [fold[metric_name] for fold in fold_metrics if not np.isnan(fold[metric_name])]

        if not values:
            aggregated[metric_name] = (np.nan, np.nan, np.nan)
            continue

        mean_val = np.mean(values)
        std_val = np.std(values, ddof=1) if len(values) > 1 else 0

        # Calculate confidence interval
        if len(values) > 1:
            sem = std_val / np.sqrt(len(values))
            ci_delta = sem * stats.t.ppf((1 + ci) / 2, len(values) - 1)
            lower_ci = mean_val - ci_delta
            upper_ci = mean_val + ci_delta
        else:
            lower_ci = mean_val
            upper_ci = mean_val

        aggregated[metric_name] = (mean_val, lower_ci, upper_ci)

    return aggregated

