"""
Preprocessing module for sepsis prediction pipeline.

All transformers are fitted ONLY on training data and applied to test data.

Includes:
- KNN imputation for numeric features (k=10)
- Mode imputation for categorical features
- StandardScaler for numeric features
- SMOTE for class balancing (training only)
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """
    Complete preprocessing pipeline for sepsis prediction.

    Steps (all fitted on training data only):
    1. Impute missing values (KNN for numeric, mode for categorical)
    2. Standardize numeric features
    3. Apply SMOTE (training only)

    Parameters
    ----------
    knn_neighbors : int
        Number of neighbors for KNN imputation (default: 10)
    smote_random_state : int
        Random state for SMOTE
    smote_k_neighbors : int
        Number of neighbors for SMOTE (default: 5)
    """

    def __init__(
        self,
        knn_neighbors: int = 10,
        smote_random_state: int = 42,
        smote_k_neighbors: int = 5,
    ):
        self.knn_neighbors = knn_neighbors
        self.smote_random_state = smote_random_state
        self.smote_k_neighbors = smote_k_neighbors

        # Imputers
        self.numeric_imputer = KNNImputer(n_neighbors=knn_neighbors)
        self.categorical_imputer = SimpleImputer(strategy="most_frequent")

        # Scaler
        self.scaler = StandardScaler()

        # Feature tracking
        self.numeric_features_ = None
        self.categorical_features_ = None
        self.feature_names_ = None

    def _detect_feature_types(self, X: pd.DataFrame) -> Tuple[list, list]:
        """Detect numeric and categorical features."""
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()
        return numeric_features, categorical_features

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        apply_smote: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit preprocessing pipeline on training data and transform.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (training data)
        y : np.ndarray
            Target labels (training data)
        apply_smote : bool
            Whether to apply SMOTE (default: True)

        Returns
        -------
        X_transformed : np.ndarray
            Transformed features
        y_transformed : np.ndarray
            Transformed labels (resampled if SMOTE applied)
        """
        logger.debug(f"Fitting preprocessing on {X.shape[0]} samples")

        self.feature_names_ = X.columns.tolist()
        self.numeric_features_, self.categorical_features_ = self._detect_feature_types(X)

        logger.debug(
            f"Detected {len(self.numeric_features_)} numeric, "
            f"{len(self.categorical_features_)} categorical features"
        )

        # Step 1: Imputation
        X_imputed = self._impute(X, fit=True)

        # Step 2: Standardization
        X_scaled = self.scaler.fit_transform(X_imputed)

        # Step 3: SMOTE (if requested)
        if apply_smote:
            X_transformed, y_transformed = self._apply_smote(X_scaled, y)
        else:
            X_transformed = X_scaled
            y_transformed = y

        logger.debug(
            f"Preprocessing complete: {X_transformed.shape[0]} samples "
            f"after transformations"
        )

        return X_transformed, y_transformed

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform new data using fitted preprocessing pipeline.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (test data)

        Returns
        -------
        X_transformed : np.ndarray
            Transformed features
        """
        if self.feature_names_ is None:
            raise ValueError("Pipeline not fitted yet. Call fit_transform first.")

        # Ensure same feature order
        X = X[self.feature_names_]

        # Step 1: Imputation
        X_imputed = self._impute(X, fit=False)

        # Step 2: Standardization
        X_scaled = self.scaler.transform(X_imputed)

        return X_scaled

    def _impute(self, X: pd.DataFrame, fit: bool) -> np.ndarray:
        """Impute missing values."""
        X_imputed = X.copy()

        if self.numeric_features_:
            if fit:
                X_imputed[self.numeric_features_] = self.numeric_imputer.fit_transform(
                    X[self.numeric_features_]
                )
            else:
                X_imputed[self.numeric_features_] = self.numeric_imputer.transform(
                    X[self.numeric_features_]
                )

        if self.categorical_features_:
            if fit:
                X_imputed[self.categorical_features_] = (
                    self.categorical_imputer.fit_transform(
                        X[self.categorical_features_]
                    )
                )
            else:
                X_imputed[self.categorical_features_] = (
                    self.categorical_imputer.transform(X[self.categorical_features_])
                )

        return X_imputed.values

    def _apply_smote(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply SMOTE to balance classes."""
        # Check if we have more than one class
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            logger.error(
                f"Cannot apply SMOTE: Only {len(unique_classes)} class found in target variable!"
            )
            logger.error(f"Class value(s): {unique_classes}")
            logger.error(
                "This likely means your data has no positive sepsis cases. "
                "Check your qSOFA column or sepsis outcome definition."
            )
            raise ValueError(
                f"Target variable 'y' contains only {len(unique_classes)} class. "
                f"Need at least 2 classes for classification. "
                f"Found class(es): {unique_classes}. "
                "Check your data's outcome variable (qSOFA >= 2 for sepsis)."
            )
        
        original_counts = np.bincount(y)
        logger.debug(f"Class distribution before SMOTE: {original_counts}")

        # Adjust k_neighbors if minority class is too small
        min_class_size = min(np.bincount(y))
        k_neighbors = min(self.smote_k_neighbors, min_class_size - 1)

        if k_neighbors < 1:
            logger.warning(
                f"Minority class size ({min_class_size}) too small for SMOTE. "
                "Skipping SMOTE."
            )
            return X, y

        smote = SMOTE(
            random_state=self.smote_random_state,
            k_neighbors=k_neighbors,
        )
        X_resampled, y_resampled = smote.fit_resample(X, y)

        resampled_counts = np.bincount(y_resampled)
        logger.debug(f"Class distribution after SMOTE: {resampled_counts}")

        return X_resampled, y_resampled


def create_preprocessing_pipeline(
    knn_neighbors: int = 10,
    smote_random_state: int = 42,
    smote_k_neighbors: int = 5,
) -> PreprocessingPipeline:
    """
    Factory function to create preprocessing pipeline.

    Parameters
    ----------
    knn_neighbors : int
        Number of neighbors for KNN imputation
    smote_random_state : int
        Random state for SMOTE
    smote_k_neighbors : int
        Number of neighbors for SMOTE

    Returns
    -------
    PreprocessingPipeline
        Configured preprocessing pipeline
    """
    return PreprocessingPipeline(
        knn_neighbors=knn_neighbors,
        smote_random_state=smote_random_state,
        smote_k_neighbors=smote_k_neighbors,
    )

