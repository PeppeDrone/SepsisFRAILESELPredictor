"""
Nested cross-validation orchestrator with Optuna hyperparameter optimization.

Implements nested 5×5 CV:
- Outer loop: 5-fold stratified CV for evaluation
- Inner loop: 5-fold stratified CV for hyperparameter optimization with Optuna

All preprocessing, dimensionality reduction, and model fitting are done
ONLY on training data to prevent leakage.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import optuna
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from tqdm import tqdm

from sepsis_ml import CLASSIFICATION_THRESHOLD
from sepsis_ml.metrics import MajorityClassBaseline, calculate_all_metrics
from sepsis_ml.models import create_model, get_model_name
from sepsis_ml.pca_rfecv import create_dimensionality_reduction
from sepsis_ml.preprocessing import create_preprocessing_pipeline
from sepsis_ml.search_space import get_search_space

logger = logging.getLogger(__name__)


def _to_serializable(obj: Any) -> Any:
    """Recursively convert numpy/pandas objects into JSON-serializable types."""
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, (np.generic,)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


class NestedCVOrchestrator:
    """
    Orchestrates nested cross-validation with Optuna HPO.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : np.ndarray
        Target labels
    model_types : list
        List of model type codes to train ('en', 'knn', 'dtc', 'rf', 'xgb')
    outer_cv_folds : int
        Number of outer CV folds (default: 5)
    inner_cv_folds : int
        Number of inner CV folds (default: 5)
    n_trials : int
        Number of Optuna trials per model (default: 50)
    optimization_metric : str
        Metric to optimize ('accuracy', 'roc_auc', 'balanced_accuracy', or 'f1')
    random_state : int
        Random seed for reproducibility
    n_jobs : int
        Number of parallel jobs for Optuna
    artifacts_dir : Path, optional
        Directory to save per-fold artifacts
    """

    def __init__(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        model_types: List[str],
        config: Dict[str, Any],
        outer_cv_folds: int = 5,
        inner_cv_folds: int = 5,
        n_trials: int = 50,
        optimization_metric: str = "accuracy",
        random_state: int = 42,
        n_jobs: int = -1,
        artifacts_dir: Optional[Path] = None,
        original_df: Optional[pd.DataFrame] = None,
    ):
        self.X = X
        self.y = y
        self.model_types = model_types
        self.config = config
        self.original_df = original_df  # Original dataframe with IDs and all columns
        self.outer_cv_folds = outer_cv_folds
        self.inner_cv_folds = inner_cv_folds
        self.n_trials = n_trials
        self.optimization_metric = optimization_metric
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.artifacts_dir = artifacts_dir
        self.apply_smote = self.config.get("apply_smote", True)
        self.smote_k_neighbors = self.config.get("smote_k_neighbors", 5)

        # Results storage
        self.results_ = {model_type: [] for model_type in model_types}
        self.best_params_ = {model_type: [] for model_type in model_types}

        # Baseline results
        self.baseline_results_ = []

        # Setup outer CV
        self.outer_cv = StratifiedKFold(
            n_splits=outer_cv_folds,
            shuffle=True,
            random_state=random_state,
        )

        # Setup logging
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    def run(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Run complete nested CV for all models.

        Returns
        -------
        dict
            Results dictionary with per-fold metrics for each model
        """
        logger.info(
            f"Starting nested {self.outer_cv_folds}×{self.inner_cv_folds} CV "
            f"with {len(self.model_types)} models"
        )
        logger.info(f"Optimization metric: {self.optimization_metric}")
        logger.info(f"Models: {[get_model_name(m) for m in self.model_types]}")
        
        # Save config file if artifacts directory provided
        if self.artifacts_dir is not None:
            self.artifacts_dir.mkdir(parents=True, exist_ok=True)
            config_path = self.artifacts_dir / "config.json"
            with open(config_path, "w") as f:
                json.dump(_to_serializable(self.config), f, indent=2, default=str)
            logger.info(f"Saved configuration to {config_path}")

        # Iterate over outer folds
        fold_iterator = enumerate(self.outer_cv.split(self.X, self.y), start=1)

        for fold_idx, (train_idx, test_idx) in tqdm(
            fold_iterator,
            total=self.outer_cv_folds,
            desc="Outer CV folds",
        ):
            logger.info(f"\n{'='*80}")
            logger.info(f"Outer Fold {fold_idx}/{self.outer_cv_folds}")
            logger.info(f"{'='*80}")

            # Split data
            X_train, X_test = self.X.iloc[train_idx], self.X.iloc[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]

            logger.info(
                f"Train: {len(y_train)} samples, "
                f"Test: {len(y_test)} samples"
            )
            logger.info(
                f"Train prevalence: {y_train.mean()*100:.1f}%, "
                f"Test prevalence: {y_test.mean()*100:.1f}%"
            )

            # Preprocess training data
            preprocessor = create_preprocessing_pipeline(
                knn_neighbors=self.config.get("knn_neighbors", 10),
                smote_random_state=self.random_state,
                smote_k_neighbors=self.smote_k_neighbors,
            )

            X_train_prep, y_train_processed = preprocessor.fit_transform(
                X_train, y_train, apply_smote=False
            )

            # Apply dimensionality reduction
            dim_reducer = create_dimensionality_reduction(
                pca_variance_threshold=0.999,
                rfecv_cv=self.inner_cv_folds,
                random_state=self.random_state,
            )

            X_train_reduced = dim_reducer.fit_transform(X_train_prep, y_train_processed)

            logger.info(
                f"Dimensionality reduction: {X_train_prep.shape[1]} -> "
                f"{X_train_reduced.shape[1]} features"
            )
            logger.info(
                f"PCA components: {dim_reducer.n_components_}, "
                f"RFECV selected: {dim_reducer.n_features_selected_}"
            )

            # Preprocess test data (no SMOTE)
            X_test_prep = preprocessor.transform(X_test)
            X_test_reduced = dim_reducer.transform(X_test_prep)

            # Train baseline
            self._train_baseline(fold_idx, y_train, y_test)

            # Train each model
            for model_type in self.model_types:
                logger.info(f"\n--- Training {get_model_name(model_type)} ---")

                # Hyperparameter optimization (inner CV)
                best_params = self._optimize_hyperparameters(
                    model_type=model_type,
                    X_train=X_train_reduced,
                    y_train=y_train_processed,
                    fold_idx=fold_idx,
                )

                # Refit on full outer training fold with best params using SMOTE-inside pipeline
                model_pipeline = self._create_model_pipeline(
                    model_type=model_type,
                    params=best_params,
                    y=y_train_processed,
                )
                model_pipeline.fit(X_train_reduced, y_train_processed)

                # Predict on outer test fold
                y_pred_proba = model_pipeline.predict_proba(X_test_reduced)[:, 1]
                y_pred = (y_pred_proba >= CLASSIFICATION_THRESHOLD).astype(int)

                # Calculate metrics
                metrics = calculate_all_metrics(
                    y_true=y_test,
                    y_pred=y_pred,
                    y_proba=y_pred_proba,
                    threshold=CLASSIFICATION_THRESHOLD,
                )

                # Store results (including original data and indices for misclassification analysis)
                fold_result = {
                    "fold": fold_idx,
                    "model_type": model_type,
                    "best_params": best_params,
                    "y_true": y_test,
                    "y_pred": y_pred,
                    "y_proba": y_pred_proba,
                    "metrics": metrics,
                    "test_indices": test_idx,  # Original indices in full dataset
                    "X_test_original": X_test.values,  # Original data before preprocessing
                }

                self.results_[model_type].append(fold_result)
                self.best_params_[model_type].append(best_params)

                logger.info(
                    f"Fold {fold_idx} {get_model_name(model_type)} results: "
                    f"Acc={metrics['accuracy']:.3f}, "
                    f"Sens={metrics['sensitivity']:.3f}, "
                    f"Spec={metrics['specificity']:.3f}, "
                    f"F1={metrics['f1']:.3f}, "
                    f"ROC-AUC={metrics['roc_auc']:.3f}"
                )

                # Save artifacts if requested
                if self.artifacts_dir is not None:
                    self._save_fold_artifacts(
                        fold_idx=fold_idx,
                        model_type=model_type,
                        fold_result=fold_result,
                        train_idx=train_idx,
                        test_idx=test_idx,
                        X_train=X_train,
                        X_test=X_test,
                        y_train=y_train,
                        y_test=y_test,
                        preprocessor=preprocessor,
                        dim_reducer=dim_reducer,
                        model_pipeline=model_pipeline,
                        X_train_processed=X_train_prep,
                        X_test_processed=X_test_prep,
                        X_train_reduced=X_train_reduced,
                        X_test_reduced=X_test_reduced,
                        y_train_processed=y_train_processed,
                    )

        logger.info(f"\n{'='*80}")
        logger.info("Nested CV complete!")
        logger.info(f"{'='*80}\n")

        return self.results_

    def _optimize_hyperparameters(
        self,
        model_type: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        fold_idx: int,
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Optuna with inner CV.

        Parameters
        ----------
        model_type : str
            Model type code
        X_train : np.ndarray
            Training features (already preprocessed)
        y_train : np.ndarray
            Training labels
        fold_idx : int
            Outer fold index (for logging)

        Returns
        -------
        dict
            Best hyperparameters
        """

        def objective(trial: optuna.Trial) -> float:
            # Get hyperparameters from search space
            params = get_search_space(model_type, trial, self.config)

            # Create model pipeline with optional SMOTE step
            model = self._create_model_pipeline(
                model_type=model_type,
                params=params,
                y=y_train,
            )

            # Inner CV evaluation
            inner_cv = StratifiedKFold(
                n_splits=self.inner_cv_folds,
                shuffle=True,
                random_state=self.random_state,
            )

            # Validate optimization metric
            valid_metrics = ["accuracy", "roc_auc", "balanced_accuracy", "f1"]
            if self.optimization_metric not in valid_metrics:
                raise ValueError(
                    f"Invalid optimization_metric: {self.optimization_metric}. "
                    f"Must be one of: {valid_metrics}"
                )
            
            scores = cross_val_score(
                model,
                X_train,
                y_train,
                cv=inner_cv,
                scoring=self.optimization_metric,
                n_jobs=1,  # Parallelism handled by Optuna
            )

            return scores.mean()

        # Track best trial for logging and early stopping
        best_trial_number = [None]
        trials_since_best = [0]
        
        # Early stopping configuration
        enable_early_stopping = self.config.get("enable_early_stopping", True)
        early_stopping_patience = self.config.get("early_stopping_patience", 20)
        enable_pruning = self.config.get("enable_pruning", True)
        
        if enable_early_stopping:
            logger.debug(f"  Early stopping enabled: patience={early_stopping_patience} trials")
        
        # Setup pruning if enabled
        # Note: Pruning works best with iterative training. For CV-based optimization,
        # the callback-based early stopping is more effective.
        if enable_pruning:
            pruning_n_startup = self.config.get("pruning_n_startup_trials", 5)
            pruning_n_warmup = self.config.get("pruning_n_warmup_steps", 1)
            pruner = optuna.pruners.MedianPruner(
                n_startup_trials=pruning_n_startup,
                n_warmup_steps=pruning_n_warmup,
            )
            logger.debug(f"  Pruning enabled: MedianPruner (startup={pruning_n_startup}, warmup={pruning_n_warmup})")
        else:
            pruner = optuna.pruners.NopPruner()
            logger.debug("  Pruning disabled")

        def callback(study: optuna.Study, trial: optuna.Trial) -> None:
            """Callback to log when a new best trial is found and check for early stopping."""
            # Check if a new best trial was found
            if study.best_trial is None:
                return
            
            current_best_number = study.best_trial.number
            current_trial_number = trial.number
            
            if current_best_number != best_trial_number[0]:
                # New best trial found
                best_trial_number[0] = current_best_number
                trials_since_best[0] = current_trial_number - current_best_number
                logger.info(
                    f"  Trial {current_best_number}: New best {self.optimization_metric} = "
                    f"{study.best_value:.4f}"
                )
            else:
                # No new best - calculate trials since best
                trials_since_best[0] = current_trial_number - best_trial_number[0]
            
            # Early stopping check: stop if no improvement for N consecutive trials
            if enable_early_stopping and trials_since_best[0] >= early_stopping_patience:
                logger.info(
                    f"  Early stopping triggered: No improvement for {trials_since_best[0]} trials "
                    f"(patience={early_stopping_patience}). "
                    f"Best {self.optimization_metric} = {study.best_value:.4f} "
                    f"(found at trial {best_trial_number[0]}, stopped at trial {current_trial_number})"
                )
                study.stop()

        # Create Optuna study with pruner
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
            pruner=pruner,
        )

        # Optimize
        study.optimize(
            objective,
            n_trials=self.n_trials,
            n_jobs=self.n_jobs,
            show_progress_bar=False,
            callbacks=[callback],
        )

        best_params = study.best_params
        best_score = study.best_value
        n_completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        was_stopped_early = n_completed_trials < self.n_trials

        if was_stopped_early:
            logger.info(
                f"  HPO complete (early stopped): best {self.optimization_metric} = {best_score:.4f} "
                f"(found at trial {study.best_trial.number}, completed {n_completed_trials}/{self.n_trials} trials)"
            )
        else:
            logger.info(
                f"  HPO complete: best {self.optimization_metric} = {best_score:.4f} "
                f"(found at trial {study.best_trial.number}, completed all {n_completed_trials} trials)"
            )
        logger.debug(f"  Best params: {best_params}")

        # Save study if artifacts directory provided
        if self.artifacts_dir is not None:
            study_path = (
                self.artifacts_dir
                / f"optuna_study_fold{fold_idx}_{model_type}.json"
            )
            study_path.parent.mkdir(parents=True, exist_ok=True)

            study_data = {
                "best_params": best_params,
                "best_value": best_score,
                "best_trial_number": study.best_trial.number,
                "n_trials_completed": n_completed_trials,
                "n_trials_requested": self.n_trials,
                "early_stopped": was_stopped_early,
                "early_stopping_patience": early_stopping_patience if enable_early_stopping else None,
            }

            with open(study_path, "w") as f:
                json.dump(_to_serializable(study_data), f, indent=2)

        return best_params

    def _create_model_pipeline(
        self,
        model_type: str,
        params: Dict[str, Any],
        y: np.ndarray,
    ) -> ImbPipeline:
        """
        Create an imbalanced-learn pipeline that applies SMOTE inside CV folds
        before fitting the estimator.
        """
        steps: List[Tuple[str, Any]] = []

        smote_step = self._make_smote_step(y)
        if smote_step is not None:
            steps.append(("smote", smote_step))

        estimator = create_model(
            model_type=model_type,
            params=params,
            random_state=self.random_state,
        )
        steps.append(("model", estimator))

        return ImbPipeline(steps)

    def _make_smote_step(self, y: np.ndarray) -> Optional[SMOTE]:
        """
        Build a SMOTE instance with k_neighbors adapted to the minority class size.
        Returns None if SMOTE should be skipped.
        """
        if not self.apply_smote:
            return None

        classes, counts = np.unique(y, return_counts=True)
        if len(classes) < 2:
            logger.warning(
                "Skipping SMOTE: only one class present in training data."
            )
            return None

        min_class_count = counts.min()
        if min_class_count <= 1:
            logger.warning(
                "Skipping SMOTE: minority class has <=1 sample in training data."
            )
            return None

        desired_k = self.smote_k_neighbors
        k_neighbors = min(desired_k, max(1, min_class_count - 1))

        if k_neighbors < 1:
            logger.warning(
                "Skipping SMOTE: could not determine a valid k_neighbors value."
            )
            return None

        if k_neighbors < desired_k:
            logger.debug(
                "Adjusted SMOTE k_neighbors from %s to %s based on minority "
                "class size.",
                desired_k,
                k_neighbors,
            )

        return SMOTE(
            random_state=self.random_state,
            k_neighbors=k_neighbors,
        )

    def _train_baseline(
        self,
        fold_idx: int,
        y_train: np.ndarray,
        y_test: np.ndarray,
    ):
        """Train and evaluate majority-class baseline."""
        baseline = MajorityClassBaseline()
        baseline.fit(y_train)

        y_pred = baseline.predict(len(y_test))
        y_proba = baseline.predict_proba(len(y_test))

        metrics = calculate_all_metrics(
            y_true=y_test,
            y_pred=y_pred,
            y_proba=y_proba,
            threshold=CLASSIFICATION_THRESHOLD,
        )

        self.baseline_results_.append(
            {
                "fold": fold_idx,
                "metrics": metrics,
            }
        )

        logger.info(
            f"Baseline (majority class): "
            f"Acc={metrics['accuracy']:.3f}, "
            f"ROC-AUC={metrics['roc_auc']:.3f}"
        )

    def _save_fold_artifacts(
        self,
        fold_idx: int,
        model_type: str,
        fold_result: Dict[str, Any],
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: np.ndarray,
        y_test: np.ndarray,
        preprocessor: Any,
        dim_reducer: Any,
        model_pipeline: Any,
        X_train_processed: np.ndarray,
        X_test_processed: np.ndarray,
        X_train_reduced: np.ndarray,
        X_test_reduced: np.ndarray,
        y_train_processed: np.ndarray,
    ):
        """
        Save per-fold artifacts to disk for complete reproducibility.
        
        Saves:
        - Predictions, metrics, best hyperparameters
        - Train/test data splits (raw, processed, reduced)
        - Fitted preprocessor (full object + components)
        - Fitted dimensionality reducer (full object + components)
        - Fitted model pipeline (with SMOTE) and SMOTE metadata
        """
        fold_dir = self.artifacts_dir / f"fold_{fold_idx}" / model_type
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
            json.dump(_to_serializable(fold_result["metrics"]), f, indent=2)

        # Save best params
        with open(fold_dir / "best_params.json", "w") as f:
            json.dump(_to_serializable(fold_result["best_params"]), f, indent=2)

        # Save train/test data splits
        train_data = X_train.copy()
        train_data["y"] = y_train
        train_data.to_csv(fold_dir / "train_data.csv", index=False)
        
        test_data = X_test.copy()
        test_data["y"] = y_test
        test_data.to_csv(fold_dir / "test_data.csv", index=False)

        # Save processed (imputed + scaled) representations
        processed_feature_names = getattr(preprocessor, "feature_names_", None)
        if processed_feature_names is None:
            processed_feature_names = [
                f"feature_{i}" for i in range(X_train_processed.shape[1])
            ]
        train_processed_df = pd.DataFrame(
            X_train_processed, columns=processed_feature_names
        )
        train_processed_df["y"] = y_train_processed
        train_processed_df.to_csv(fold_dir / "train_data_processed.csv", index=False)

        test_processed_df = pd.DataFrame(X_test_processed, columns=processed_feature_names)
        test_processed_df["y"] = y_test
        test_processed_df.to_csv(fold_dir / "test_data_processed.csv", index=False)

        # Save reduced (PCA/RFECV) representations
        reduced_feature_names = [f"component_{i+1}" for i in range(X_train_reduced.shape[1])]
        train_reduced_df = pd.DataFrame(
            X_train_reduced, columns=reduced_feature_names
        )
        train_reduced_df["y"] = y_train_processed
        train_reduced_df.to_csv(fold_dir / "train_data_reduced.csv", index=False)

        test_reduced_df = pd.DataFrame(X_test_reduced, columns=reduced_feature_names)
        test_reduced_df["y"] = y_test
        test_reduced_df.to_csv(fold_dir / "test_data_reduced.csv", index=False)
        
        # Save indices for reproducibility
        pd.DataFrame({"index": train_idx}).to_csv(
            fold_dir / "train_indices.csv", index=False
        )
        pd.DataFrame({"index": test_idx}).to_csv(
            fold_dir / "test_indices.csv", index=False
        )

        # Save fitted preprocessor (entire object, imputers, scaler)
        preprocessor_dir = fold_dir / "preprocessor"
        preprocessor_dir.mkdir(exist_ok=True)
        joblib.dump(preprocessor, preprocessor_dir / "preprocessor.joblib")
        
        # Save imputers
        joblib.dump(preprocessor.numeric_imputer, preprocessor_dir / "numeric_imputer.joblib")
        joblib.dump(preprocessor.categorical_imputer, preprocessor_dir / "categorical_imputer.joblib")
        
        # Save scaler (with means and stds)
        joblib.dump(preprocessor.scaler, preprocessor_dir / "scaler.joblib")
        
        # Save scaler statistics separately for easy inspection
        scaler_stats = {
            "mean": preprocessor.scaler.mean_.tolist(),
            "std": preprocessor.scaler.scale_.tolist(),
            "feature_names": preprocessor.feature_names_,
            "numeric_features": preprocessor.numeric_features_,
            "categorical_features": preprocessor.categorical_features_,
        }
        with open(preprocessor_dir / "scaler_stats.json", "w") as f:
            json.dump(_to_serializable(scaler_stats), f, indent=2)

        # Save fitted dimensionality reducer (full object + components)
        dim_reducer_dir = fold_dir / "dimensionality_reducer"
        dim_reducer_dir.mkdir(exist_ok=True)
        joblib.dump(dim_reducer, dim_reducer_dir / "dimensionality_reducer.joblib")
        joblib.dump(dim_reducer.pca, dim_reducer_dir / "pca.joblib")
        joblib.dump(dim_reducer.rfecv, dim_reducer_dir / "rfecv.joblib")
        
        # Save dimensionality reduction metadata
        dim_metadata = {
            "n_components": int(dim_reducer.n_components_),
            "n_features_selected": int(dim_reducer.n_features_selected_),
            "pca_variance_threshold": dim_reducer.pca_variance_threshold,
            "pca_explained_variance_ratio": dim_reducer.pca.explained_variance_ratio_.tolist(),
            "pca_cumulative_variance": np.cumsum(dim_reducer.pca.explained_variance_ratio_).tolist(),
            "rfecv_support": dim_reducer.rfecv.support_.tolist(),
            "rfecv_ranking": dim_reducer.rfecv.ranking_.tolist(),
        }
        with open(dim_reducer_dir / "metadata.json", "w") as f:
            json.dump(_to_serializable(dim_metadata), f, indent=2)

        # Save fitted model pipeline (includes SMOTE if applied)
        joblib.dump(model_pipeline, fold_dir / "model_pipeline.joblib")
        
        # Extract and save SMOTE separately if present
        if hasattr(model_pipeline, "named_steps") and "smote" in model_pipeline.named_steps:
            smote_step = model_pipeline.named_steps["smote"]
            joblib.dump(smote_step, fold_dir / "smote.joblib")
            
            # Save SMOTE metadata
            smote_metadata = {
                "k_neighbors": smote_step.k_neighbors,
                "random_state": smote_step.random_state,
            }
            with open(fold_dir / "smote_metadata.json", "w") as f:
                json.dump(_to_serializable(smote_metadata), f, indent=2)
        
        # Save metadata for quick inspection
        metadata = {
            "fold": fold_idx,
            "model_type": model_type,
            "train_size": int(len(train_idx)),
            "test_size": int(len(test_idx)),
            "random_state": self.random_state,
            "optimization_metric": self.optimization_metric,
        }
        with open(fold_dir / "metadata.json", "w") as f:
            json.dump(_to_serializable(metadata), f, indent=2)

        logger.debug(
            "Saved reproducibility artifacts for fold %s, model %s at %s",
            fold_idx,
            model_type,
            fold_dir,
        )


def run_nested_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    model_types: List[str],
    config: Dict[str, Any],
    artifacts_dir: Optional[Path] = None,
    original_df: Optional[pd.DataFrame] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Convenience function to run nested CV.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : np.ndarray
        Target labels
    model_types : list
        Model types to train
    config : dict
        Configuration dictionary
    artifacts_dir : Path, optional
        Directory to save artifacts

    Returns
    -------
    dict
        Results dictionary
    """
    orchestrator = NestedCVOrchestrator(
        X=X,
        y=y,
        model_types=model_types,
        config=config,
        outer_cv_folds=config.get("outer_cv_folds", 5),
        inner_cv_folds=config.get("inner_cv_folds", 5),
        n_trials=config.get("n_trials", 50),
        optimization_metric=config.get("optimization_metric", "accuracy"),
        random_state=config.get("random_state", 42),
        n_jobs=config.get("n_jobs", -1),
        artifacts_dir=artifacts_dir,
        original_df=original_df,
    )

    results = orchestrator.run()

    # Return both model results and baseline
    return {
        "models": results,
        "baseline": orchestrator.baseline_results_,
    }

