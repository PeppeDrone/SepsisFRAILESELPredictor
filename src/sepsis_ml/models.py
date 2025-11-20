"""
Model definitions for sepsis prediction.

Includes:
- Elastic-Net Logistic Regression
- K-Nearest Neighbors
- Decision Tree
- Random Forest (expanded ranges)
- XGBoost (new)
"""

import logging
from typing import Any, Dict

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


def create_elastic_net_logistic_regression(
    params: Dict[str, Any],
    random_state: int = 42,
) -> LogisticRegression:
    """
    Create Elastic-Net Logistic Regression model.

    Parameters
    ----------
    params : dict
        Hyperparameters (C, l1_ratio, class_weight)
    random_state : int
        Random state

    Returns
    -------
    LogisticRegression
        Configured model
    """
    return LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        C=params.get("C", 1.0),
        l1_ratio=params.get("l1_ratio", 0.5),
        class_weight=params.get("class_weight", None),
        max_iter=1000,
        random_state=random_state,
        n_jobs=-1,
    )


def create_knn_classifier(
    params: Dict[str, Any],
) -> KNeighborsClassifier:
    """
    Create K-Nearest Neighbors classifier.

    Parameters
    ----------
    params : dict
        Hyperparameters (n_neighbors, weights, metric)

    Returns
    -------
    KNeighborsClassifier
        Configured model
    """
    return KNeighborsClassifier(
        n_neighbors=params.get("n_neighbors", 5),
        weights=params.get("weights", "uniform"),
        metric=params.get("metric", "euclidean"),
        n_jobs=-1,
    )


def create_decision_tree_classifier(
    params: Dict[str, Any],
    random_state: int = 42,
) -> DecisionTreeClassifier:
    """
    Create Decision Tree classifier.

    Parameters
    ----------
    params : dict
        Hyperparameters (max_depth, min_samples_split, min_samples_leaf, class_weight)
    random_state : int
        Random state

    Returns
    -------
    DecisionTreeClassifier
        Configured model
    """
    return DecisionTreeClassifier(
        max_depth=params.get("max_depth", None),
        min_samples_split=params.get("min_samples_split", 2),
        min_samples_leaf=params.get("min_samples_leaf", 1),
        class_weight=params.get("class_weight", None),
        random_state=random_state,
    )


def create_random_forest_classifier(
    params: Dict[str, Any],
    random_state: int = 42,
) -> RandomForestClassifier:
    """
    Create Random Forest classifier with expanded hyperparameter ranges.

    Parameters
    ----------
    params : dict
        Hyperparameters (n_estimators, max_depth, max_features,
                         min_samples_split, min_samples_leaf,
                         bootstrap, class_weight)
    random_state : int
        Random state

    Returns
    -------
    RandomForestClassifier
        Configured model
    """
    return RandomForestClassifier(
        n_estimators=params.get("n_estimators", 100),
        max_depth=params.get("max_depth", None),
        max_features=params.get("max_features", "sqrt"),
        min_samples_split=params.get("min_samples_split", 2),
        min_samples_leaf=params.get("min_samples_leaf", 1),
        bootstrap=params.get("bootstrap", True),
        class_weight=params.get("class_weight", None),
        random_state=random_state,
        n_jobs=-1,
    )


def create_xgboost_classifier(
    params: Dict[str, Any],
    random_state: int = 42,
) -> XGBClassifier:
    """
    Create XGBoost classifier.

    Parameters
    ----------
    params : dict
        Hyperparameters (n_estimators, max_depth, learning_rate,
                         subsample, colsample_bytree, reg_alpha,
                         reg_lambda, min_child_weight, gamma)
    random_state : int
        Random state

    Returns
    -------
    XGBClassifier
        Configured model
    """
    return XGBClassifier(
        n_estimators=params.get("n_estimators", 100),
        max_depth=params.get("max_depth", 6),
        learning_rate=params.get("learning_rate", 0.1),
        subsample=params.get("subsample", 1.0),
        colsample_bytree=params.get("colsample_bytree", 1.0),
        reg_alpha=params.get("reg_alpha", 0.0),
        reg_lambda=params.get("reg_lambda", 1.0),
        min_child_weight=params.get("min_child_weight", 1),
        gamma=params.get("gamma", 0.0),
        random_state=random_state,
        eval_metric="logloss",
        # use_label_encoder=False,
        n_jobs=-1,
    )


MODEL_FACTORY = {
    "en": create_elastic_net_logistic_regression,
    "knn": create_knn_classifier,
    "dtc": create_decision_tree_classifier,
    "rf": create_random_forest_classifier,
    "xgb": create_xgboost_classifier,
}

MODEL_NAMES = {
    "en": "Elastic-Net Logistic Regression",
    "knn": "K-Nearest Neighbors",
    "dtc": "Decision Tree",
    "rf": "Random Forest",
    "xgb": "XGBoost",
}


def create_model(
    model_type: str,
    params: Dict[str, Any],
    random_state: int = 42,
):
    """
    Factory function to create any model by type.

    Parameters
    ----------
    model_type : str
        Model type code ('en', 'knn', 'dtc', 'rf', 'xgb')
    params : dict
        Hyperparameters
    random_state : int
        Random state

    Returns
    -------
    estimator
        Configured sklearn/xgboost estimator
    """
    if model_type not in MODEL_FACTORY:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Available: {list(MODEL_FACTORY.keys())}"
        )

    factory_func = MODEL_FACTORY[model_type]

    # Pass random_state to models that need it
    if model_type in ["en", "dtc", "rf", "xgb"]:
        return factory_func(params, random_state=random_state)
    else:
        return factory_func(params)


def get_model_name(model_type: str) -> str:
    """Get human-readable model name."""
    return MODEL_NAMES.get(model_type, model_type)

