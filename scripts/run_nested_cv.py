#!/usr/bin/env python
"""
Main CLI entrypoint for running nested cross-validation.

Usage:
    python scripts/run_nested_cv.py --data data.xlsx --config configs/default.yaml --seed 42
"""

import argparse
import json
import logging
import random
import sys
import warnings
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")

warnings.filterwarnings(
    "ignore",
    message=r"resource_tracker: .*",
    category=UserWarning,
)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sepsis_ml.ensemble import evaluate_ensemble_across_folds
from sepsis_ml.io import prepare_dataset
from sepsis_ml.nested_cv import run_nested_cv
from sepsis_ml.plotting import plot_all_figures, plot_missing_data
from sepsis_ml.reporting import generate_all_reports

# Setup logging with UTF-8 encoding to handle special characters
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("sepsis_ml.log", encoding="utf-8"),
    ],
)

logger = logging.getLogger(__name__)


def set_random_seeds(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    logger.info(f"Set random seeds to {seed}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run nested cross-validation for sepsis prediction"
    )

    parser.add_argument(
        "--data",
        type=str,
        default="data.xlsx",
        help="Path to input Excel data file (default: data.xlsx)",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration YAML file (default: configs/default.yaml)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)",
    )

    parser.add_argument(
        "--opt-metric",
        type=str,
        choices=["accuracy", "roc_auc", "balanced_accuracy", "f1"],
        default=None,
        help="Optimization metric for inner CV (overrides config). Options: accuracy, roc_auc, balanced_accuracy, f1",
    )

    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help="Comma-separated list of models to train (en,knn,dtc,rf,xgb) or 'all'. Overrides config file if provided. (default: use config file or 'all')",
    )

    parser.add_argument(
        "--no-ensemble",
        action="store_true",
        help="Skip ensemble creation",
    )

    parser.add_argument(
        "--save-artifacts",
        dest="save_artifacts",
        action="store_true",
        help="(Default) Save per-fold artifacts for exact reproducibility",
    )
    parser.add_argument(
        "--no-artifacts",
        dest="save_artifacts",
        action="store_false",
        help="Skip saving per-fold artifacts (not recommended)",
    )
    parser.set_defaults(save_artifacts=True)

    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Number of parallel jobs for Optuna (default: -1 = all cores)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="Output directory for reports and figures (default: reports)",
    )

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    config_file = Path(config_path)

    if not config_file.exists():
        logger.warning(f"Config file {config_path} not found. Using defaults.")
        return {}

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded configuration from {config_path}")
    return config


def main():
    """Main execution function."""
    args = parse_args()

    logger.info("="*100)
    logger.info("SEPSIS ML: Nested Cross-Validation Pipeline")
    logger.info("="*100)

    # Load configuration
    config = load_config(args.config)

    # Override config with command-line arguments
    if args.seed is not None:
        config["random_state"] = args.seed
    if args.opt_metric is not None:
        config["optimization_metric"] = args.opt_metric
    if args.n_jobs is not None:
        config["n_jobs"] = args.n_jobs

    # Set defaults if not in config
    config.setdefault("random_state", 42)
    config.setdefault("optimization_metric", "accuracy")
    config.setdefault("outer_cv_folds", 5)
    config.setdefault("inner_cv_folds", 5)
    config.setdefault("n_trials", 50)
    config.setdefault("n_jobs", -1)
    config.setdefault("completion_threshold", 0.90)
    config.setdefault("outcome_column", "qSOFA")
    config.setdefault("outcome_threshold", 2)

    # Set random seeds
    set_random_seeds(config["random_state"])

    # Parse model selection
    # Priority: command-line argument > config file > default (all models)
    if args.models.lower() != "all":
        # Command-line argument overrides config
        model_types = [m.strip() for m in args.models.split(",")]
    elif "models" in config and config["models"]:
        # Use models from config file
        model_types = config["models"]
        # Filter out commented/None entries (in case YAML has nulls)
        model_types = [m for m in model_types if m is not None]
    else:
        # Default to all models if nothing specified
        model_types = ["en", "knn", "dtc", "rf", "xgb"]

    logger.info(f"Selected models: {model_types}")

    # Generate unique run ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_uuid = str(uuid4())[:8]
    run_id = f"{timestamp}_{short_uuid}"
    
    # Prepare output directories with run ID
    base_output_dir = Path(args.output_dir)
    output_dir = base_output_dir / f"run_{run_id}"
    artifacts_dir = output_dir / "artifacts" if args.save_artifacts else None
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save run metadata
    run_info = {
        "run_id": run_id,
        "timestamp": timestamp,
        "config_file": args.config,
        "data_file": args.data,
        "models": model_types,
        "random_state": config.get("random_state", 42),
        "outer_cv_folds": config.get("outer_cv_folds", 5),
        "inner_cv_folds": config.get("inner_cv_folds", 5),
        "n_trials": config.get("n_trials", 50),
        "optimization_metric": config.get("optimization_metric", "accuracy"),
        "save_artifacts": args.save_artifacts,
        "artifacts_dir": str(artifacts_dir) if artifacts_dir else None,
    }
    
    with open(output_dir / "run_info.json", "w") as f:
        json.dump(run_info, f, indent=2)
    
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Output directory: {output_dir}")

    # Step 1: Load and prepare data
    logger.info(f"\nStep 1: Loading data from {args.data}")
    data_path = Path(args.data)
    
    if not data_path.exists():
        logger.error(f"Data file not found: {args.data}")
        sys.exit(1)

    df_clean, feature_names, completion_rates = prepare_dataset(
        filepath=data_path,
        config=config,
    )

    X = df_clean[feature_names]
    y = df_clean["sepsis"].values
    
    # Keep original dataframe for misclassification analysis
    original_df = df_clean.copy()

    logger.info(
        f"Dataset prepared: {X.shape[0]} samples, {X.shape[1]} features, "
        f"sepsis prevalence: {y.mean()*100:.1f}%"
    )

    # Step 0: Save missing data analysis and plot (at the beginning before everything)
    logger.info("\nStep 0: Saving missing data analysis")
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Save completion rates to JSON
    completion_rates_pct = {k: v * 100 for k, v in completion_rates.items()}
    missing_rates_pct = {k: (1 - v) * 100 for k, v in completion_rates.items()}
    
    # Calculate statistics for included (retained) features
    retained_missing_pct = [missing_rates_pct[f] for f in feature_names]
    median_missing = np.median(retained_missing_pct)
    q25 = np.percentile(retained_missing_pct, 25)
    q75 = np.percentile(retained_missing_pct, 75)
    iqr_missing = q75 - q25
    
    # Log statistics
    n_total_initial = len(completion_rates)
    n_total_included = len(feature_names)
    logger.info(f"Missing data summary:")
    logger.info(f"  Total initial features: {n_total_initial}")
    logger.info(f"  Total included features: {n_total_included}")
    logger.info(f"  Dropped features: {n_total_initial - n_total_included}")
    logger.info(f"  Missing % for included features - Median: {median_missing:.2f}%, IQR: [{q25:.2f}%, {q75:.2f}%]")
    
    missing_data_info = {
        "completion_rates_percent": completion_rates_pct,
        "missing_rates_percent": missing_rates_pct,
        "threshold_percent": config.get("completion_threshold", 0.90) * 100,
        "retained_features": feature_names,
        "dropped_features": [f for f in completion_rates.keys() if f not in feature_names],
        "n_retained": len(feature_names),
        "n_dropped": len(completion_rates) - len(feature_names),
        "n_total": len(completion_rates),
        "included_features_missing_stats": {
            "median": float(median_missing),
            "q25": float(q25),
            "q75": float(q75),
            "iqr": float(iqr_missing),
        },
    }
    
    with open(output_dir / "missing_data_info.json", "w") as f:
        json.dump(missing_data_info, f, indent=2)
    
    logger.info(f"Saved missing data info to {output_dir / 'missing_data_info.json'}")
    
    # Plot missing data barplot
    plot_missing_data(
        completion_rates=completion_rates,
        retained_features=feature_names,
        threshold=config.get("completion_threshold", 0.90),
        output_path=figures_dir / "missing_data_barplot.png",
    )

    # Step 2: Run nested cross-validation
    logger.info(f"\nStep 2: Running nested {config['outer_cv_folds']}×{config['inner_cv_folds']} CV")
    logger.info(f"Optimization metric: {config['optimization_metric']}")
    logger.info(f"Optuna trials per model: {config['n_trials']}")

    cv_results = run_nested_cv(
        X=X,
        y=y,
        model_types=model_types,
        config=config,
        artifacts_dir=artifacts_dir,
        original_df=original_df,
    )

    model_results = cv_results["models"]
    baseline_results = cv_results["baseline"]

    # Step 3: Create ensemble (if requested)
    if not args.no_ensemble:
        logger.info("\nStep 3: Creating ensemble via majority vote")
        ensemble_results = evaluate_ensemble_across_folds(model_results)
        model_results["ensemble"] = ensemble_results

    # Step 4: Generate reports
    logger.info("\nStep 4: Generating reports and tables")

    # Collect best params
    from sepsis_ml.nested_cv import NestedCVOrchestrator
    # Note: best_params are stored in the results already, so we extract them
    best_params = {}
    for model_type, fold_results in model_results.items():
        if model_type == "ensemble":
            continue
        best_params[model_type] = [
            fr.get("best_params", {}) for fr in fold_results if "best_params" in fr
        ]

    generate_all_reports(
        results=model_results,
        baseline_results=baseline_results,
        output_dir=output_dir,
        best_params=best_params if best_params else None,
        original_df=original_df,
        feature_names=feature_names,
    )

    # Step 5: Generate figures
    logger.info("\nStep 5: Generating figures")
    plot_all_figures(
        model_results, 
        figures_dir,
        original_df=original_df,
        feature_names=feature_names,
    )

    logger.info("\n" + "="*100)
    logger.info("Pipeline complete!")
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Reports saved to: {output_dir / 'tables'}")
    logger.info(f"Figures saved to: {figures_dir}")
    logger.info(f"Misclassifications saved to: {output_dir / 'misclassifications'}")
    if args.save_artifacts:
        logger.info(f"Artifacts saved to: {artifacts_dir}")
    logger.info(f"All outputs saved to: {output_dir}")
    logger.info("="*100 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)

