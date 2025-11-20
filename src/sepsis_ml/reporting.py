"""
Reporting module for generating tables and summaries.

Includes:
- Aggregated metrics tables (CSV, Markdown)
- JSON artifacts
- Pretty console summaries
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from sepsis_ml import CLASSIFICATION_THRESHOLD
from sepsis_ml.metrics import aggregate_metrics_across_folds
from sepsis_ml.models import get_model_name

logger = logging.getLogger(__name__)


def create_metrics_summary_table(
    results: Dict[str, List[Dict]],
    baseline_results: Optional[List[Dict]] = None,
    ci: float = 0.95,
) -> pd.DataFrame:
    """
    Create summary table with aggregated metrics across folds.

    Parameters
    ----------
    results : dict
        Dictionary of model_type: list of fold results
    baseline_results : list, optional
        Baseline model results
    ci : float
        Confidence interval level

    Returns
    -------
    pd.DataFrame
        Summary table with mean ± CI for all metrics
    """
    rows = []

    # Add baseline if provided
    if baseline_results:
        baseline_metrics = [r["metrics"] for r in baseline_results]
        agg_metrics = aggregate_metrics_across_folds(baseline_metrics, ci=ci)

        row = {"Model": "Baseline (Majority Class)"}
        for metric_name, (mean_val, lower_ci, upper_ci) in agg_metrics.items():
            row[metric_name.upper()] = f"{mean_val:.3f} ({lower_ci:.3f}-{upper_ci:.3f})"

        rows.append(row)

    # Add each model
    for model_type, fold_results in results.items():
        fold_metrics = [r["metrics"] for r in fold_results]
        agg_metrics = aggregate_metrics_across_folds(fold_metrics, ci=ci)

        row = {"Model": get_model_name(model_type)}

        for metric_name, (mean_val, lower_ci, upper_ci) in agg_metrics.items():
            row[metric_name.upper()] = f"{mean_val:.3f} ({lower_ci:.3f}-{upper_ci:.3f})"

        rows.append(row)

    df = pd.DataFrame(rows)

    # Reorder columns to match manuscript style
    metric_order = [
        "Model",
        "ACCURACY",
        "SENSITIVITY",
        "SPECIFICITY",
        "F1",
        "ROC_AUC",
        "PR_AUC",
        "BRIER",
    ]

    # Keep only columns that exist
    cols = [c for c in metric_order if c in df.columns]
    df = df[cols]

    return df


def save_summary_table(
    df: pd.DataFrame,
    output_dir: Path,
    filename_stem: str = "summary",
):
    """
    Save summary table to CSV and Markdown.

    Parameters
    ----------
    df : pd.DataFrame
        Summary table
    output_dir : Path
        Output directory
    filename_stem : str
        Filename stem (without extension)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save CSV
    csv_path = output_dir / f"{filename_stem}.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved summary table (CSV) to {csv_path}")

    # Save Markdown
    md_path = output_dir / f"{filename_stem}.md"
    with open(md_path, "w") as f:
        f.write("# Sepsis Prediction Model Performance Summary\n\n")
        f.write("## Aggregated Metrics Across 5-Fold Cross-Validation\n\n")
        f.write(
            f"**Classification Threshold**: {CLASSIFICATION_THRESHOLD} "
            "(explicitly enforced for sensitivity/specificity)\n\n"
        )
        f.write(df.to_markdown(index=False))
        f.write("\n\n")
        f.write("## Interpretation\n\n")
        f.write(
            "Values shown as: **Mean (95% CI Lower - 95% CI Upper)** across folds.\n\n"
        )
        f.write("**Metrics:**\n")
        f.write("- **ACCURACY**: Overall classification accuracy\n")
        f.write("- **SENSITIVITY**: True positive rate (recall for sepsis class)\n")
        f.write("- **SPECIFICITY**: True negative rate\n")
        f.write("- **F1**: Harmonic mean of precision and recall\n")
        f.write(
            "- **ROC_AUC**: Area under the ROC curve (primary metric for imbalanced data)\n"
        )
        f.write("- **PR_AUC**: Area under the Precision-Recall curve\n")
        f.write("- **BRIER**: Brier score (lower is better; measures calibration)\n\n")
        f.write("## Key Findings\n\n")
        f.write(
            "1. **Baseline context**: The majority-class baseline provides context "
            "for interpreting accuracy in the presence of class imbalance.\n"
        )
        f.write(
            "2. **Primary metrics**: ROC-AUC and F1-score are emphasized as they are "
            "more informative for imbalanced classification tasks.\n"
        )
        f.write(
            "3. **Threshold**: All sensitivity/specificity values are calculated "
            f"using a fixed threshold of {CLASSIFICATION_THRESHOLD}.\n"
        )
        f.write(
            "4. **Ensemble**: The ensemble combines all individual models via "
            "majority vote (mode), with tie-breaking by average probability.\n\n"
        )
        f.write("## Extended Analyses per Reviewer Feedback\n\n")
        f.write(
            "This analysis extends the original manuscript with:\n"
            "- **Expanded Random Forest hyperparameter ranges** (deeper trees, "
            "more estimators)\n"
            "- **XGBoost** as an additional competitive gradient boosting model\n"
            "- **Precision-Recall curves** with AUPRC for better evaluation of "
            "imbalanced data\n"
            "- **Calibration analysis** with Brier scores to assess probability "
            "quality\n"
            "- **Decision Curve Analysis (DCA)** to evaluate clinical utility "
            "across thresholds\n"
            "- **Majority-class baseline** to contextualize performance\n\n"
        )

    logger.info(f"Saved summary report (Markdown) to {md_path}")


def save_best_params_summary(
    best_params: Dict[str, List[Dict]],
    output_dir: Path,
):
    """
    Save best hyperparameters from each fold.

    Parameters
    ----------
    best_params : dict
        Dictionary of model_type: list of best params per fold
    output_dir : Path
        Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_type, params_list in best_params.items():
        output_path = output_dir / f"best_params_{model_type}.json"

        params_data = {
            "model": get_model_name(model_type),
            "model_type": model_type,
            "folds": [
                {"fold": i + 1, "params": params}
                for i, params in enumerate(params_list)
            ],
        }

        with open(output_path, "w") as f:
            json.dump(params_data, f, indent=2)

        logger.info(
            f"Saved best params for {get_model_name(model_type)} to {output_path}"
        )


def print_console_summary(
    df: pd.DataFrame,
):
    """
    Print pretty summary to console.

    Parameters
    ----------
    df : pd.DataFrame
        Summary table
    """
    print("\n" + "=" * 100)
    print("SEPSIS PREDICTION MODEL PERFORMANCE SUMMARY")
    print("=" * 100)
    print(f"\nClassification Threshold: {CLASSIFICATION_THRESHOLD}")
    print(f"Cross-Validation: 5-fold stratified")
    print("\nResults (Mean ± 95% CI):\n")
    print(df.to_string(index=False))
    print("\n" + "=" * 100)
    print("\nKey Observations:")
    print("  - ROC-AUC and F1 are primary metrics for imbalanced classification")
    print("  - Sensitivity and specificity computed at threshold = 0.5")
    print("  - Baseline (majority class) provides context for class imbalance")
    print("  - See reports/tables/summary.md for detailed interpretation")
    print("=" * 100 + "\n")


def save_fold_level_predictions(
    results: Dict[str, List[Dict]],
    output_dir: Path,
):
    """
    Save fold-level predictions for all models.

    Parameters
    ----------
    results : dict
        Dictionary of model_type: list of fold results
    output_dir : Path
        Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_type, fold_results in results.items():
        for fold_result in fold_results:
            fold_idx = fold_result["fold"]
            fold_dir = output_dir / f"fold_{fold_idx}" / model_type
            fold_dir.mkdir(parents=True, exist_ok=True)

            # Save predictions
            pred_df = pd.DataFrame(
                {
                    "y_true": fold_result["y_true"],
                    "y_pred": fold_result["y_pred"],
                    "y_proba": fold_result["y_proba"],
                }
            )
            pred_df.to_csv(fold_dir / "predictions.csv", index=False)

            # Save metrics
            with open(fold_dir / "metrics.json", "w") as f:
                json.dump(fold_result["metrics"], f, indent=2, default=float)

    logger.info(f"Saved fold-level predictions to {output_dir}")


def identify_and_save_misclassifications(
    results: Dict[str, List[Dict]],
    original_df: pd.DataFrame,
    feature_names: List[str],
    output_dir: Path,
):
    """
    Identify misclassified patients and save their IDs and original data.

    Parameters
    ----------
    results : dict
        Dictionary of model_type: list of fold results
    original_df : pd.DataFrame
        Original dataframe with all columns including IDs
    feature_names : list
        List of feature column names
    output_dir : Path
        Output directory for misclassification files
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Try to find ID column (common names)
    id_columns = ["id", "ID", "Id", "patient_id", "PatientID", "PATIENT_ID", 
                  "subject_id", "SubjectID", "SUBJECT_ID", "index"]
    id_col = None
    for col in id_columns:
        if col in original_df.columns:
            id_col = col
            break
    
    if id_col is None:
        # Use index as ID
        logger.warning("No ID column found. Using row index as patient ID.")
        id_col = "__index__"
        original_df = original_df.copy()
        original_df[id_col] = original_df.index
    
    logger.info(f"Using '{id_col}' as patient ID column")
    
    for model_type, fold_results in results.items():
        # Aggregate all misclassifications across folds
        misclassified_data = []
        
        for fold_result in fold_results:
            y_true = np.array(fold_result["y_true"])
            y_pred = np.array(fold_result["y_pred"])
            test_indices = fold_result.get("test_indices", None)
            X_test_original = fold_result.get("X_test_original", None)
            
            if test_indices is None or X_test_original is None:
                logger.warning(
                    f"Missing test_indices or X_test_original for {model_type}. "
                    "Skipping misclassification analysis for this model."
                )
                continue
            
            # Identify misclassifications
            misclassified_mask = (y_true != y_pred)
            n_misclassified = misclassified_mask.sum()
            
            if n_misclassified == 0:
                continue
            
            misclassified_indices = test_indices[misclassified_mask]
            misclassified_y_true = y_true[misclassified_mask]
            misclassified_y_pred = y_pred[misclassified_mask]
            misclassified_proba = np.array(fold_result["y_proba"])[misclassified_mask]
            X_test_misclassified = X_test_original[misclassified_mask]
            
            # Get original data for misclassified patients
            for idx in range(n_misclassified):
                orig_idx = misclassified_indices[idx]
                if orig_idx >= len(original_df):
                    continue
                
                patient_data = {
                    "model": get_model_name(model_type),
                    "fold": fold_result["fold"],
                    "patient_id": original_df.iloc[orig_idx][id_col],
                    "true_label": int(misclassified_y_true[idx]),
                    "predicted_label": int(misclassified_y_pred[idx]),
                    "predicted_probability": float(misclassified_proba[idx]),
                    "error_type": "False Positive" if misclassified_y_true[idx] == 0 else "False Negative",
                }
                
                # Add all original feature values
                for feat_idx, feat_name in enumerate(feature_names):
                    if feat_idx < X_test_misclassified.shape[1]:
                        patient_data[feat_name] = float(X_test_misclassified[idx, feat_idx])
                    else:
                        patient_data[feat_name] = np.nan
                
                # Add any other columns from original dataframe
                for col in original_df.columns:
                    if col not in feature_names and col != id_col and col != "sepsis":
                        patient_data[col] = original_df.iloc[orig_idx][col]
                
                misclassified_data.append(patient_data)
        
        if misclassified_data:
            # Create DataFrame
            misclassified_df = pd.DataFrame(misclassified_data)
            
            # Save to CSV
            output_file = output_dir / f"misclassified_patients_{model_type}.csv"
            misclassified_df.to_csv(output_file, index=False)
            
            # Summary statistics
            n_fp = (misclassified_df["error_type"] == "False Positive").sum()
            n_fn = (misclassified_df["error_type"] == "False Negative").sum()
            
            logger.info(
                f"Saved {len(misclassified_df)} misclassified patients for "
                f"{get_model_name(model_type)}: {n_fp} FP, {n_fn} FN -> {output_file}"
            )
            
            # Also save summary
            summary = {
                "model": get_model_name(model_type),
                "total_misclassified": len(misclassified_df),
                "false_positives": int(n_fp),
                "false_negatives": int(n_fn),
                "false_positive_rate": float(n_fp / len(misclassified_df)) if len(misclassified_df) > 0 else 0.0,
            }
            
            summary_file = output_dir / f"misclassification_summary_{model_type}.json"
            with open(summary_file, "w") as f:
                json.dump(summary, f, indent=2)
        else:
            logger.info(
                f"No misclassifications found for {get_model_name(model_type)} "
                "(or missing original data)"
            )


def generate_all_reports(
    results: Dict[str, List[Dict]],
    baseline_results: Optional[List[Dict]],
    output_dir: Path,
    best_params: Optional[Dict[str, List[Dict]]] = None,
    original_df: Optional[pd.DataFrame] = None,
    feature_names: Optional[List[str]] = None,
):
    """
    Generate all reports and save to output directory.

    Parameters
    ----------
    results : dict
        Dictionary of model_type: list of fold results
    baseline_results : list, optional
        Baseline results
    output_dir : Path
        Output directory
    best_params : dict, optional
        Best hyperparameters per model
    original_df : pd.DataFrame, optional
        Original dataframe with IDs and all columns
    feature_names : list, optional
        List of feature column names
    """
    logger.info("Generating reports...")

    # Create summary table
    df = create_metrics_summary_table(results, baseline_results)

    # Print to console
    print_console_summary(df)

    # Save tables
    tables_dir = output_dir / "tables"
    save_summary_table(df, tables_dir)

    # Save best params
    if best_params:
        artifacts_dir = output_dir / "artifacts"
        save_best_params_summary(best_params, artifacts_dir)

    # Save fold-level predictions
    artifacts_dir = output_dir / "artifacts"
    save_fold_level_predictions(results, artifacts_dir)

    # Identify and save misclassifications
    if original_df is not None and feature_names is not None:
        logger.info("Identifying misclassified patients...")
        misclass_dir = output_dir / "misclassifications"
        identify_and_save_misclassifications(
            results=results,
            original_df=original_df,
            feature_names=feature_names,
            output_dir=misclass_dir,
        )
    else:
        logger.warning(
            "Original dataframe or feature names not provided. "
            "Skipping misclassification analysis."
        )

    # Save merged predictions with original data if requested
    if original_df is not None:
        logger.info("Merging predictions with original data...")
        merged_dir = output_dir / "tables"
        save_merged_predictions_with_data(
            results=results,
            original_df=original_df,
            feature_names=feature_names,
            output_path=merged_dir / "data_predictions.csv",
        )

    logger.info(f"All reports saved to {output_dir}")


def save_merged_predictions_with_data(
    results: Dict[str, List[Dict]],
    original_df: pd.DataFrame,
    feature_names: Optional[List[str]],
    output_path: Path,
):
    """
    Merge per-fold predictions with the original dataset and save as a single CSV.

    Parameters
    ----------
    results : dict
        Model results keyed by model type.
    original_df : pd.DataFrame
        Original dataset with all columns.
    feature_names : list, optional
        Ordered list of feature names used during modeling.
    output_path : Path
        Path to the output CSV file.
    """
    if not results:
        logger.warning("No model results provided; skipping merged predictions.")
        return

    reference_df = original_df.copy()
    reference_df["_row_index"] = reference_df.index

    merged_dfs = []

    for model_type, fold_results in results.items():
        model_name = get_model_name(model_type)

        for fold_result in fold_results:
            test_indices = fold_result.get("test_indices")
            if test_indices is None:
                logger.warning(
                    "Missing test_indices for %s fold %s; skipping merge.",
                    model_name,
                    fold_result.get("fold"),
                )
                continue

            y_true = pd.Series(fold_result["y_true"], name="true_label")
            y_pred = pd.Series(fold_result["y_pred"], name=f"{model_type}_pred")
            y_proba = pd.Series(
                fold_result["y_proba"],
                name=f"{model_type}_proba",
                dtype=float,
            )

            fold_df = pd.DataFrame(
                {
                    "_row_index": test_indices,
                    "fold": fold_result["fold"],
                    "model_type": model_type,
                    "model_name": model_name,
                }
            )
            fold_df = pd.concat([fold_df, y_true, y_pred, y_proba], axis=1)
            merged_dfs.append(fold_df)

    if not merged_dfs:
        logger.warning(
            "No fold predictions available to merge; skipping merged predictions."
        )
        return

    merged_predictions = pd.concat(merged_dfs, ignore_index=True)
    merged_predictions = merged_predictions.merge(
        reference_df,
        on="_row_index",
        how="left",
        validate="m:1",
    )

    merged_predictions.drop(columns=["_row_index"], inplace=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_predictions.to_csv(output_path, index=False)

    logger.info("Saved merged predictions to %s", output_path)

