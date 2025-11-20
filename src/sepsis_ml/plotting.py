"""
Plotting module for publication-quality figures.

Includes:
- Missing data barplot
- ROC curves with 95% CI
- Precision-Recall curves with AUPRC
- Calibration curves with Brier scores
- Decision Curve Analysis
- Confusion matrices
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.metrics import auc, confusion_matrix

from sepsis_ml.dca import aggregate_dca_curves, calculate_dca_curve
from sepsis_ml.metrics import (
    calculate_calibration_curve_data,
    calculate_pr_curve_data,
    calculate_roc_curve_data,
)
from sepsis_ml.models import get_model_name

logger = logging.getLogger(__name__)

# Plot styling
plt.rcParams.update(
    {
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
        "figure.titlesize": 15,
        "figure.dpi": 150,
    }
)


def _save_figure(fig: plt.Figure, output_path: Optional[Path], description: str):
    """Save PNG and SVG variants for a figure."""
    if not output_path:
        return

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    svg_path = output_path.with_suffix(".svg")
    fig.savefig(svg_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved {description} to {output_path} and {svg_path}")


def plot_missing_data(
    completion_rates: Dict[str, float],
    retained_features: List[str],
    threshold: float = 0.90,
    output_path: Optional[Path] = None,
    title: str = "Missing Data Percentage by Feature",
):
    """
    Plot barplot showing percentage of missing data for each feature.
    Gray bars indicate retained features, red bars indicate dropped features.

    Parameters
    ----------
    completion_rates : dict
        Dictionary mapping feature names to their completion rates (0-1)
    retained_features : list
        List of feature names that were retained (passed the threshold)
    threshold : float
        Completion threshold used for filtering (default: 0.90)
    output_path : Path, optional
        Path to save figure
    title : str
        Figure title
    """
    # Calculate missing percentages
    features = list(completion_rates.keys())
    missing_pct = [(1 - completion_rates[f]) * 100 for f in features]
    
    # Determine which features are retained vs dropped
    is_retained = [f in retained_features for f in features]
    
    # Sort by missing percentage (descending) for better visualization
    sorted_indices = np.argsort(missing_pct)[::-1]
    features_sorted = [features[i] for i in sorted_indices]
    missing_pct_sorted = [missing_pct[i] for i in sorted_indices]
    is_retained_sorted = [is_retained[i] for i in sorted_indices]
    
    # Create figure
    n_features = len(features)
    fig_height = max(8, n_features * 0.3)  # Adjust height based on number of features
    fig, ax = plt.subplots(figsize=(10, fig_height))
    
    # Create color arrays: gray for retained, red for dropped
    # Use darker colors for borders
    bar_colors = ['gray' if retained else 'red' for retained in is_retained_sorted]
    edge_colors = ['#404040' if retained else '#8B0000' for retained in is_retained_sorted]  # Dark gray and dark red
    
    # Create horizontal barplot with darker borders
    y_pos = np.arange(len(features_sorted))
    bars = ax.barh(y_pos, missing_pct_sorted, color=bar_colors, alpha=0.7, linewidth=1.0)
    
    # Set darker border colors for each bar
    for bar, edge_color in zip(bars, edge_colors):
        bar.set_edgecolor(edge_color)
    
    # Add threshold line for inclusion percentage (completion threshold)
    threshold_pct = (1 - threshold) * 100
    inclusion_pct = threshold * 100  # Inclusion percentage
    ax.axvline(x=threshold_pct, color='black', linestyle='--', linewidth=2.0, 
               label=f'Inclusion threshold ({inclusion_pct:.1f}% completion, {threshold_pct:.1f}% missing)')
    
    # Customize axes
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features_sorted, fontsize=14)
    ax.tick_params(axis='x', labelsize=14)
    ax.set_xlabel('Missing Data (%)', fontsize=16)
    ax.set_ylabel('Feature', fontsize=16)
    ax.set_title(title, fontsize=15, fontweight='bold')
    ax.set_xlim([0, 40])  # Cut x-axis to 40%
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Make bottom and left spines thicker
    ax.spines['bottom'].set_linewidth(2.0)
    ax.spines['left'].set_linewidth(2.0)
    
    # Add legend with darker border colors (top right)
    gray_patch = Patch(facecolor='gray', alpha=0.7, edgecolor='#404040', linewidth=1.0, label='Retained features')
    red_patch = Patch(facecolor='red', alpha=0.7, edgecolor='#8B0000', linewidth=1.0, label='Dropped features')
    threshold_line = plt.Line2D([0], [0], color='black', linestyle='--', linewidth=2.0, 
                                label=f'Inclusion threshold ({inclusion_pct:.1f}% completion)')
    ax.legend(handles=[gray_patch, red_patch, threshold_line], loc='upper right', fontsize=15)
    
    # Add percentage labels on bars (larger font)
    for i, (bar, pct) in enumerate(zip(bars, missing_pct_sorted)):
        if pct > 0.5:  # Only label if missing % is significant
            width = bar.get_width()
            # Clamp width to xlim if it exceeds 40%
            display_width = min(width, 40)
            ax.text(display_width + 0.5, bar.get_y() + bar.get_height()/2, 
                   f'{pct:.1f}%', ha='left', va='center', fontsize=13)
    
    plt.tight_layout()
    
    _save_figure(fig, output_path, "missing data plot")
    
    plt.close(fig)


def plot_roc_curves(
    results: Dict[str, List[Dict]],
    output_path: Optional[Path] = None,
    title: str = "ROC Curves (Mean ± 95% CI across folds)",
):
    """
    Plot ROC curves for all models with confidence intervals.

    Parameters
    ----------
    results : dict
        Dictionary of model_type: list of fold results
    output_path : Path, optional
        Path to save figure
    title : str
        Figure title
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    for (model_type, fold_results), color in zip(results.items(), colors):
        # Collect ROC curves from all folds
        fpr_list = []
        tpr_list = []
        auc_list = []

        for fold_result in fold_results:
            y_true = fold_result["y_true"]
            y_proba = fold_result["y_proba"]

            fpr, tpr, _ = calculate_roc_curve_data(y_true, y_proba)
            fpr_list.append(fpr)
            tpr_list.append(tpr)
            auc_list.append(fold_result["metrics"]["roc_auc"])

        # Interpolate to common FPR grid
        mean_fpr = np.linspace(0, 1, 100)
        tpr_interp_list = []

        for fpr, tpr in zip(fpr_list, tpr_list):
            tpr_interp = np.interp(mean_fpr, fpr, tpr)
            tpr_interp[0] = 0.0
            tpr_interp_list.append(tpr_interp)

        # Calculate mean and std
        mean_tpr = np.mean(tpr_interp_list, axis=0)
        mean_tpr[-1] = 1.0
        std_tpr = np.std(tpr_interp_list, axis=0)

        mean_auc = np.mean(auc_list)
        std_auc = np.std(auc_list)

        # Plot mean curve
        label = f"{get_model_name(model_type)} (AuROC = {mean_auc:.3f} ± {std_auc:.3f})"
        ax.plot(mean_fpr, mean_tpr, color=color, lw=2, label=label)

        # Plot confidence interval
        tpr_upper = np.minimum(mean_tpr + 1.96 * std_tpr, 1)
        tpr_lower = np.maximum(mean_tpr - 1.96 * std_tpr, 0)
        ax.fill_between(
            mean_fpr, tpr_lower, tpr_upper, color=color, alpha=0.2
        )

    # Plot diagonal reference
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance (AuROC = 0.500)")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    plt.tight_layout()

    _save_figure(fig, output_path, "ROC curves")

    plt.close(fig)


def plot_pr_curves(
    results: Dict[str, List[Dict]],
    output_path: Optional[Path] = None,
    title: str = "Precision-Recall Curves (Mean ± 95% CI across folds)",
):
    """
    Plot Precision-Recall curves with AUPRC.

    Parameters
    ----------
    results : dict
        Dictionary of model_type: list of fold results
    output_path : Path, optional
        Path to save figure
    title : str
        Figure title
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    for (model_type, fold_results), color in zip(results.items(), colors):
        # Collect PR curves from all folds
        recall_list = []
        precision_list = []
        auprc_list = []

        for fold_result in fold_results:
            y_true = fold_result["y_true"]
            y_proba = fold_result["y_proba"]

            precision, recall, _ = calculate_pr_curve_data(y_true, y_proba)
            recall_list.append(recall)
            precision_list.append(precision)
            auprc_list.append(fold_result["metrics"]["pr_auc"])

        # Interpolate to common recall grid
        mean_recall = np.linspace(0, 1, 100)
        precision_interp_list = []

        for recall, precision in zip(recall_list, precision_list):
            # Reverse for interpolation (recall must be increasing)
            precision_interp = np.interp(
                mean_recall, recall[::-1], precision[::-1]
            )
            precision_interp_list.append(precision_interp)

        # Calculate mean and std
        mean_precision = np.mean(precision_interp_list, axis=0)
        std_precision = np.std(precision_interp_list, axis=0)

        mean_auprc = np.mean(auprc_list)
        std_auprc = np.std(auprc_list)

        # Plot mean curve
        label = f"{get_model_name(model_type)} (AuPRC = {mean_auprc:.3f} ± {std_auprc:.3f})"
        ax.plot(mean_recall, mean_precision, color=color, lw=2, label=label)

        # Plot confidence interval
        precision_upper = np.minimum(mean_precision + 1.96 * std_precision, 1)
        precision_lower = np.maximum(mean_precision - 1.96 * std_precision, 0)
        ax.fill_between(
            mean_recall, precision_lower, precision_upper, color=color, alpha=0.2
        )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(alpha=0.3)

    plt.tight_layout()

    _save_figure(fig, output_path, "precision-recall curves")

    plt.close(fig)


def plot_calibration_curves(
    results: Dict[str, List[Dict]],
    output_path: Optional[Path] = None,
    n_bins: int = 10,
    title: str = "Calibration Curves",
):
    """
    Plot calibration curves with Brier scores.

    Parameters
    ----------
    results : dict
        Dictionary of model_type: list of fold results
    output_path : Path, optional
        Path to save figure
    n_bins : int
        Number of bins for calibration
    title : str
        Figure title
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    for (model_type, fold_results), color in zip(results.items(), colors):
        # Aggregate predictions across folds
        y_true_all = []
        y_proba_all = []
        brier_scores = []

        for fold_result in fold_results:
            y_true_all.extend(fold_result["y_true"])
            y_proba_all.extend(fold_result["y_proba"])
            brier_scores.append(fold_result["metrics"]["brier"])

        y_true_all = np.array(y_true_all)
        y_proba_all = np.array(y_proba_all)

        # Calculate calibration curve
        prob_true, prob_pred = calculate_calibration_curve_data(
            y_true_all, y_proba_all, n_bins=n_bins
        )

        mean_brier = np.mean(brier_scores)
        std_brier = np.std(brier_scores)

        # Plot
        label = (
            f"{get_model_name(model_type)} "
            f"(Brier = {mean_brier:.3f} ± {std_brier:.3f})"
        )
        ax.plot(prob_pred, prob_true, marker="o", color=color, lw=2, label=label)

    # Plot perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")

    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(alpha=0.3)

    plt.tight_layout()

    _save_figure(fig, output_path, "calibration curves")

    plt.close(fig)


def plot_dca_curves(
    results: Dict[str, List[Dict]],
    output_path: Optional[Path] = None,
    title: str = "Decision Curve Analysis",
):
    """
    Plot Decision Curve Analysis comparing models to treat-all/treat-none.

    Parameters
    ----------
    results : dict
        Dictionary of model_type: list of fold results
    output_path : Path, optional
        Path to save figure
    title : str
        Figure title
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    # Calculate DCA curves for each model
    for (model_type, fold_results), color in zip(results.items(), colors):
        dca_curves = []

        for fold_result in fold_results:
            y_true = fold_result["y_true"]
            y_proba = fold_result["y_proba"]

            dca_curve = calculate_dca_curve(y_true, y_proba)
            dca_curves.append(dca_curve)

        # Aggregate across folds
        (
            thresholds,
            mean_nb_model,
            std_nb_model,
            mean_nb_all,
            std_nb_all,
            mean_nb_none,
        ) = aggregate_dca_curves(dca_curves)

        # Plot model net benefit
        label = get_model_name(model_type)
        ax.plot(thresholds, mean_nb_model, color=color, lw=2, label=label)

        # Plot confidence interval
        nb_upper = mean_nb_model + 1.96 * std_nb_model
        nb_lower = mean_nb_model - 1.96 * std_nb_model
        ax.fill_between(thresholds, nb_lower, nb_upper, color=color, alpha=0.2)

    # Plot treat-all and treat-none (use first model's data as they're the same)
    first_model = list(results.keys())[0]
    dca_curves = []
    for fold_result in results[first_model]:
        y_true = fold_result["y_true"]
        y_proba = fold_result["y_proba"]
        dca_curve = calculate_dca_curve(y_true, y_proba)
        dca_curves.append(dca_curve)

    (
        thresholds,
        _,
        _,
        mean_nb_all,
        std_nb_all,
        mean_nb_none,
    ) = aggregate_dca_curves(dca_curves)

    ax.plot(thresholds, mean_nb_all, "k--", lw=1.5, label="Treat All")
    ax.plot(thresholds, mean_nb_none, "k:", lw=1.5, label="Treat None")

    ax.set_xlabel("Threshold Probability")
    ax.set_ylabel("Net Benefit")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([-0.5, 0.5])  # Fixed y-axis limits for better comparison

    plt.tight_layout()

    _save_figure(fig, output_path, "decision curve analysis plot")

    plt.close(fig)


def plot_confusion_matrices(
    results: Dict[str, List[Dict]],
    output_path: Optional[Path] = None,
):
    """
    Plot confusion matrices for all models (aggregated across folds).

    Parameters
    ----------
    results : dict
        Dictionary of model_type: list of fold results
    output_path : Path, optional
        Path to save figure
    """
    n_models = len(results)
    n_cols = min(3, n_models)
    n_rows = int(np.ceil(n_models / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))

    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, (model_type, fold_results) in enumerate(results.items()):
        # Aggregate predictions across all folds
        y_true_all = []
        y_pred_all = []

        for fold_result in fold_results:
            y_true_all.extend(fold_result["y_true"])
            y_pred_all.extend(fold_result["y_pred"])

        y_true_all = np.array(y_true_all)
        y_pred_all = np.array(y_pred_all)

        # Calculate confusion matrix
        cm = confusion_matrix(y_true_all, y_pred_all)

        # Plot with fixed color scale for comparison across panels
        ax = axes[idx]
        im = ax.imshow(
            cm, 
            cmap="Blues", 
            interpolation="nearest",
            vmin=0,
            vmax=2250
        )

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Count", fontsize=15)
        cbar.ax.tick_params(labelsize=13)

        # Add labels
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["qSOFA < 2", "qSOFA>=2 (sepsis)"], fontsize=13)
        ax.set_yticklabels(["qSOFA < 2", "qSOFA>=2 (sepsis)"], fontsize=13)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(get_model_name(model_type))

        # Add text annotations
        # Use fixed threshold (half of vmax) for text color to ensure readability
        text_threshold = 2250 / 2
        for i in range(2):
            for j in range(2):
                text = ax.text(
                    j,
                    i,
                    f"{cm[i, j]}",
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > text_threshold else "black",
                    fontsize=16,
                )

    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()

    _save_figure(fig, output_path, "confusion matrices")

    plt.close(fig)


def plot_all_figures(
    results: Dict[str, List[Dict]],
    output_dir: Path,
    original_df: Optional[pd.DataFrame] = None,
    feature_names: Optional[List[str]] = None,
):
    """
    Generate all figures and save to output directory.

    Parameters
    ----------
    results : dict
        Dictionary of model_type: list of fold results
    output_dir : Path
        Directory to save figures
    original_df : pd.DataFrame, optional
        Original dataframe with IDs and all columns (for misclassification analysis)
    feature_names : list, optional
        List of feature column names (for misclassification analysis)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating all figures...")

    # ROC curves
    plot_roc_curves(results, output_path=output_dir / "roc_curves.png")

    # PR curves
    plot_pr_curves(results, output_path=output_dir / "pr_curves.png")

    # Calibration curves
    plot_calibration_curves(results, output_path=output_dir / "calibration_curves.png")

    # DCA curves
    plot_dca_curves(results, output_path=output_dir / "dca_curves.png")

    # Confusion matrices (this will also trigger misclassification analysis if data provided)
    plot_confusion_matrices(results, output_path=output_dir / "confusion_matrices.png")

    logger.info(f"All figures saved to {output_dir}")

