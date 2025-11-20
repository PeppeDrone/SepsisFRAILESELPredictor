"""
Dimensionality reduction and feature selection module.

Includes:
- PCA with whitening, retaining ≥0.999 explained variance
- RFECV with DecisionTreeClassifier base estimator
"""

import logging
from typing import Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

logger = logging.getLogger(__name__)


class DimensionalityReduction:
    """
    Combined PCA and RFECV pipeline.

    Steps:
    1. PCA with whitening to retain ≥0.999 explained variance
    2. RFECV with DecisionTree base estimator

    Parameters
    ----------
    pca_variance_threshold : float
        Minimum explained variance to retain (default: 0.999)
    rfecv_cv : int
        Number of CV folds for RFECV (default: 5)
    rfecv_step : int or float
        Step size for RFECV (default: 1)
    random_state : int
        Random state for reproducibility
    """

    def __init__(
        self,
        pca_variance_threshold: float = 0.999,
        rfecv_cv: int = 5,
        rfecv_step: int = 1,
        random_state: int = 42,
    ):
        self.pca_variance_threshold = pca_variance_threshold
        self.rfecv_cv = rfecv_cv
        self.rfecv_step = rfecv_step
        self.random_state = random_state

        self.pca = None
        self.rfecv = None
        self.n_components_ = None
        self.n_features_selected_ = None

    def fit_transform(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """
        Fit PCA and RFECV on training data and transform.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (training data)
        y : np.ndarray
            Target labels (training data)

        Returns
        -------
        X_transformed : np.ndarray
            Transformed features after PCA and RFECV
        """
        logger.debug(f"Fitting dimensionality reduction on {X.shape}")

        # Step 1: PCA with whitening
        X_pca = self._fit_transform_pca(X)

        # Step 2: RFECV
        X_selected = self._fit_transform_rfecv(X_pca, y)

        logger.info(
            f"Dimensionality reduction: {X.shape[1]} -> "
            f"{X_pca.shape[1]} (PCA) -> {X_selected.shape[1]} (RFECV)"
        )

        return X_selected

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform new data using fitted PCA and RFECV.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (test data)

        Returns
        -------
        X_transformed : np.ndarray
            Transformed features
        """
        if self.pca is None or self.rfecv is None:
            raise ValueError("Pipeline not fitted yet. Call fit_transform first.")

        # Apply PCA
        X_pca = self.pca.transform(X)

        # Apply RFECV
        X_selected = self.rfecv.transform(X_pca)

        return X_selected

    def _fit_transform_pca(self, X: np.ndarray) -> np.ndarray:
        """Fit PCA to retain ≥ variance threshold."""
        # Determine number of components needed
        n_samples, n_features = X.shape
        max_components = min(n_samples, n_features)

        # Fit PCA with all components to get explained variance
        pca_full = PCA(whiten=True, random_state=self.random_state)
        pca_full.fit(X)

        # Find minimum components to exceed variance threshold
        cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
        n_components = np.argmax(cumulative_variance >= self.pca_variance_threshold) + 1

        # Ensure at least 1 component
        n_components = max(1, min(n_components, max_components))

        logger.debug(
            f"PCA: {n_components}/{max_components} components to retain "
            f"{cumulative_variance[n_components-1]:.6f} variance "
            f"(threshold: {self.pca_variance_threshold})"
        )

        # Refit with selected components
        self.pca = PCA(
            n_components=n_components,
            whiten=True,
            random_state=self.random_state,
        )
        X_pca = self.pca.fit_transform(X)
        self.n_components_ = n_components

        return X_pca

    def _fit_transform_rfecv(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit RFECV with DecisionTree base estimator."""
        base_estimator = DecisionTreeClassifier(random_state=self.random_state)

        cv = StratifiedKFold(n_splits=self.rfecv_cv, shuffle=True, random_state=self.random_state)

        self.rfecv = RFECV(
            estimator=base_estimator,
            step=self.rfecv_step,
            cv=cv,
            scoring="accuracy",
            n_jobs=-1,
        )

        X_selected = self.rfecv.fit_transform(X, y)
        self.n_features_selected_ = self.rfecv.n_features_

        logger.debug(
            f"RFECV: selected {self.n_features_selected_}/{X.shape[1]} features"
        )

        return X_selected


def create_dimensionality_reduction(
    pca_variance_threshold: float = 0.999,
    rfecv_cv: int = 5,
    rfecv_step: int = 1,
    random_state: int = 42,
) -> DimensionalityReduction:
    """
    Factory function to create dimensionality reduction pipeline.

    Parameters
    ----------
    pca_variance_threshold : float
        Minimum explained variance for PCA
    rfecv_cv : int
        Number of CV folds for RFECV
    rfecv_step : int
        Step size for RFECV
    random_state : int
        Random state for reproducibility

    Returns
    -------
    DimensionalityReduction
        Configured pipeline
    """
    return DimensionalityReduction(
        pca_variance_threshold=pca_variance_threshold,
        rfecv_cv=rfecv_cv,
        rfecv_step=rfecv_step,
        random_state=random_state,
    )

