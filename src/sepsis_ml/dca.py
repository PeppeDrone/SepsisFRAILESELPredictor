"""
Decision Curve Analysis (DCA) utilities.

DCA evaluates the clinical utility of predictive models across different
probability thresholds by calculating net benefit.

Net Benefit = (TP/N) - (FP/N) * (pt/(1-pt))

where:
- TP = True Positives
- FP = False Positives
- N = Total samples
- pt = Threshold probability
"""

import logging
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def calculate_net_benefit(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
) -> float:
    """
    Calculate net benefit for a given threshold.

    Net Benefit = (TP/N) - (FP/N) * (pt/(1-pt))

    Parameters
    ----------
    y_true : np.ndarray
        True labels (binary)
    y_proba : np.ndarray
        Predicted probabilities for positive class
    threshold : float
        Probability threshold (0 < threshold < 1)

    Returns
    -------
    float
        Net benefit value
    """
    if threshold <= 0 or threshold >= 1:
        return 0.0

    n = len(y_true)
    y_pred = (y_proba >= threshold).astype(int)

    # Calculate TP and FP
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))

    # Net benefit formula
    net_benefit = (tp / n) - (fp / n) * (threshold / (1 - threshold))

    return net_benefit


def calculate_treat_all_net_benefit(
    y_true: np.ndarray,
    threshold: float,
) -> float:
    """
    Calculate net benefit for "treat all" strategy.

    Assumes all patients are predicted positive.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    threshold : float
        Probability threshold

    Returns
    -------
    float
        Net benefit for treat-all strategy
    """
    if threshold <= 0 or threshold >= 1:
        return 0.0

    n = len(y_true)
    prevalence = np.mean(y_true)

    # All samples treated: TP = all positives, FP = all negatives
    net_benefit = prevalence - (1 - prevalence) * (threshold / (1 - threshold))

    return net_benefit


def calculate_treat_none_net_benefit() -> float:
    """
    Calculate net benefit for "treat none" strategy.

    Always returns 0 (no intervention = no benefit or harm).

    Returns
    -------
    float
        Net benefit (always 0.0)
    """
    return 0.0


def calculate_dca_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate complete Decision Curve Analysis.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_proba : np.ndarray
        Predicted probabilities
    thresholds : np.ndarray, optional
        Array of thresholds to evaluate. If None, uses 100 points from 0.01 to 0.99.

    Returns
    -------
    thresholds : np.ndarray
        Threshold probabilities
    net_benefit_model : np.ndarray
        Net benefit for the model
    net_benefit_all : np.ndarray
        Net benefit for treat-all strategy
    net_benefit_none : np.ndarray
        Net benefit for treat-none strategy (all zeros)
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 100)

    net_benefit_model = np.array(
        [calculate_net_benefit(y_true, y_proba, t) for t in thresholds]
    )

    net_benefit_all = np.array(
        [calculate_treat_all_net_benefit(y_true, t) for t in thresholds]
    )

    net_benefit_none = np.zeros_like(thresholds)

    return thresholds, net_benefit_model, net_benefit_all, net_benefit_none


def aggregate_dca_curves(
    dca_curves: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Aggregate DCA curves across multiple folds.

    Parameters
    ----------
    dca_curves : list
        List of (thresholds, nb_model, nb_all, nb_none) tuples from each fold

    Returns
    -------
    thresholds : np.ndarray
        Common threshold array
    mean_nb_model : np.ndarray
        Mean net benefit for model
    std_nb_model : np.ndarray
        Std dev of net benefit for model
    mean_nb_all : np.ndarray
        Mean net benefit for treat-all
    std_nb_all : np.ndarray
        Std dev of net benefit for treat-all
    mean_nb_none : np.ndarray
        Mean net benefit for treat-none (all zeros)
    """
    # Ensure all folds use same thresholds
    thresholds = dca_curves[0][0]

    nb_model_list = [curve[1] for curve in dca_curves]
    nb_all_list = [curve[2] for curve in dca_curves]
    nb_none_list = [curve[3] for curve in dca_curves]

    mean_nb_model = np.mean(nb_model_list, axis=0)
    std_nb_model = np.std(nb_model_list, axis=0)

    mean_nb_all = np.mean(nb_all_list, axis=0)
    std_nb_all = np.std(nb_all_list, axis=0)

    mean_nb_none = np.mean(nb_none_list, axis=0)

    return thresholds, mean_nb_model, std_nb_model, mean_nb_all, std_nb_all, mean_nb_none

