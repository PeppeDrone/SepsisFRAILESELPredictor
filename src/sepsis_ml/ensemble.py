"""
Ensemble methods for combining model predictions.

Implements majority-vote (mode) ensemble with tie-breaking.
"""

import logging
from typing import Dict, List

import numpy as np
from scipy import stats

from sepsis_ml import CLASSIFICATION_THRESHOLD
from sepsis_ml.metrics import calculate_all_metrics

logger = logging.getLogger(__name__)


def create_majority_vote_ensemble(
    model_predictions: Dict[str, np.ndarray],
    model_probabilities: Dict[str, np.ndarray],
    threshold: float = CLASSIFICATION_THRESHOLD,
) -> tuple:
    """
    Create ensemble predictions via majority vote.

    Ties are broken by selecting the class with highest average probability
    across all models.

    Parameters
    ----------
    model_predictions : dict
        Dictionary of model_name: predictions (0/1)
    model_probabilities : dict
        Dictionary of model_name: probabilities for positive class
    threshold : float
        Classification threshold (default: 0.5)

    Returns
    -------
    y_pred_ensemble : np.ndarray
        Ensemble predictions
    y_proba_ensemble : np.ndarray
        Ensemble probabilities (average across models)
    """
    # Stack predictions and probabilities
    pred_matrix = np.column_stack(list(model_predictions.values()))
    proba_matrix = np.column_stack(list(model_probabilities.values()))

    n_samples = pred_matrix.shape[0]

    # Average probability across models
    y_proba_ensemble = proba_matrix.mean(axis=1)

    # Majority vote with tie-breaking
    y_pred_ensemble = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):
        votes = pred_matrix[i, :]

        # Count votes
        mode_result = stats.mode(votes, keepdims=True)
        majority_class = mode_result.mode[0]
        vote_count = mode_result.count[0]

        # Check for tie
        n_models = len(votes)
        if vote_count == n_models / 2 and n_models % 2 == 0:
            # Tie: use average probability
            if y_proba_ensemble[i] >= threshold:
                y_pred_ensemble[i] = 1
            else:
                y_pred_ensemble[i] = 0
        else:
            # No tie: use majority
            y_pred_ensemble[i] = majority_class

    logger.debug(
        f"Ensemble: {len(model_predictions)} models combined via majority vote"
    )

    return y_pred_ensemble, y_proba_ensemble


def evaluate_ensemble_across_folds(
    model_results: Dict[str, List[Dict]],
    threshold: float = CLASSIFICATION_THRESHOLD,
) -> List[Dict]:
    """
    Evaluate ensemble performance across all CV folds.

    Parameters
    ----------
    model_results : dict
        Dictionary of model_type: list of fold results
    threshold : float
        Classification threshold

    Returns
    -------
    list
        List of ensemble results per fold
    """
    # Determine number of folds
    first_model = list(model_results.keys())[0]
    n_folds = len(model_results[first_model])

    ensemble_results = []

    for fold_idx in range(n_folds):
        # Collect predictions and probabilities from all models for this fold
        model_predictions = {}
        model_probabilities = {}
        y_true = None

        for model_type, fold_results in model_results.items():
            fold_result = fold_results[fold_idx]

            model_predictions[model_type] = fold_result["y_pred"]
            model_probabilities[model_type] = fold_result["y_proba"]

            if y_true is None:
                y_true = fold_result["y_true"]

        # Create ensemble predictions
        y_pred_ensemble, y_proba_ensemble = create_majority_vote_ensemble(
            model_predictions=model_predictions,
            model_probabilities=model_probabilities,
            threshold=threshold,
        )

        # Calculate metrics
        metrics = calculate_all_metrics(
            y_true=y_true,
            y_pred=y_pred_ensemble,
            y_proba=y_proba_ensemble,
            threshold=threshold,
        )

        ensemble_results.append(
            {
                "fold": fold_idx + 1,
                "model_type": "ensemble",
                "y_true": y_true,
                "y_pred": y_pred_ensemble,
                "y_proba": y_proba_ensemble,
                "metrics": metrics,
            }
        )

        logger.info(
            f"Fold {fold_idx+1} Ensemble results: "
            f"Acc={metrics['accuracy']:.3f}, "
            f"Sens={metrics['sensitivity']:.3f}, "
            f"Spec={metrics['specificity']:.3f}, "
            f"F1={metrics['f1']:.3f}, "
            f"ROC-AUC={metrics['roc_auc']:.3f}"
        )

    return ensemble_results

