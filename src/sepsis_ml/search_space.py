"""
Optuna hyperparameter search spaces for all models.

All ranges are configurable via the config dictionary.
"""

import optuna
from typing import Dict, Any


def get_elastic_net_search_space(trial: optuna.Trial, config: Dict[str, Any] = None) -> dict:
    """
    Elastic-Net Logistic Regression search space.

    Parameters
    ----------
    trial : optuna.Trial
        Optuna trial object
    config : dict, optional
        Configuration dictionary with search ranges

    Returns
    -------
    dict
        Hyperparameter dictionary
    """
    if config is None:
        config = {}
    
    cfg = config.get("elastic_net", {})
    
    return {
        "C": trial.suggest_float(
            "C", 
            cfg.get("C_min", 1e-4), 
            cfg.get("C_max", 1e2), 
            log=True
        ),
        "l1_ratio": trial.suggest_float(
            "l1_ratio", 
            cfg.get("l1_ratio_min", 0.0), 
            cfg.get("l1_ratio_max", 1.0)
        ),
        "class_weight": trial.suggest_categorical(
            "class_weight", 
            cfg.get("class_weight", [None, "balanced"])
        ),
    }


def get_knn_search_space(trial: optuna.Trial, config: Dict[str, Any] = None) -> dict:
    """
    K-Nearest Neighbors search space.

    Parameters
    ----------
    trial : optuna.Trial
        Optuna trial object
    config : dict, optional
        Configuration dictionary with search ranges

    Returns
    -------
    dict
        Hyperparameter dictionary
    """
    if config is None:
        config = {}
    
    cfg = config.get("knn", {})
    
    return {
        "n_neighbors": trial.suggest_int(
            "n_neighbors", 
            cfg.get("n_neighbors_min", 3), 
            cfg.get("n_neighbors_max", 30)
        ),
        "weights": trial.suggest_categorical(
            "weights", 
            cfg.get("weights", ["uniform", "distance"])
        ),
        "metric": trial.suggest_categorical(
            "metric", 
            cfg.get("metric", ["euclidean", "manhattan", "minkowski"])
        ),
    }


def get_decision_tree_search_space(trial: optuna.Trial, config: Dict[str, Any] = None) -> dict:
    """
    Decision Tree search space.

    Parameters
    ----------
    trial : optuna.Trial
        Optuna trial object
    config : dict, optional
        Configuration dictionary with search ranges

    Returns
    -------
    dict
        Hyperparameter dictionary
    """
    if config is None:
        config = {}
    
    cfg = config.get("decision_tree", {})
    
    return {
        "max_depth": trial.suggest_int(
            "max_depth", 
            cfg.get("max_depth_min", 1), 
            cfg.get("max_depth_max", 20)
        ),
        "min_samples_split": trial.suggest_int(
            "min_samples_split", 
            cfg.get("min_samples_split_min", 2), 
            cfg.get("min_samples_split_max", 20)
        ),
        "min_samples_leaf": trial.suggest_int(
            "min_samples_leaf", 
            cfg.get("min_samples_leaf_min", 1), 
            cfg.get("min_samples_leaf_max", 10)
        ),
        "class_weight": trial.suggest_categorical(
            "class_weight", 
            cfg.get("class_weight", [None, "balanced"])
        ),
    }


def get_random_forest_search_space(trial: optuna.Trial, config: Dict[str, Any] = None) -> dict:
    """
    Random Forest search space (expanded per reviewer feedback).

    All ranges configurable via config dictionary.

    Parameters
    ----------
    trial : optuna.Trial
        Optuna trial object
    config : dict, optional
        Configuration dictionary with search ranges

    Returns
    -------
    dict
        Hyperparameter dictionary
    """
    if config is None:
        config = {}
    
    cfg = config.get("random_forest", {})
    
    params = {
        "n_estimators": trial.suggest_int(
            "n_estimators", 
            cfg.get("n_estimators_min", 100), 
            cfg.get("n_estimators_max", 500)
        ),
        "min_samples_split": trial.suggest_int(
            "min_samples_split", 
            cfg.get("min_samples_split_min", 2), 
            cfg.get("min_samples_split_max", 20)
        ),
        "min_samples_leaf": trial.suggest_int(
            "min_samples_leaf", 
            cfg.get("min_samples_leaf_min", 1), 
            cfg.get("min_samples_leaf_max", 10)
        ),
        "bootstrap": trial.suggest_categorical(
            "bootstrap", 
            cfg.get("bootstrap", [True, False])
        ),
        "class_weight": trial.suggest_categorical(
            "class_weight", 
            cfg.get("class_weight", [None, "balanced"])
        ),
    }

    # max_depth: None or int
    if cfg.get("max_depth_none", True) and trial.suggest_categorical("max_depth_none", [True, False]):
        params["max_depth"] = None
    else:
        params["max_depth"] = trial.suggest_int(
            "max_depth_int", 
            cfg.get("max_depth_min", 5), 
            cfg.get("max_depth_max", 20)
        )

    # max_features: categorical or float
    max_features_options = cfg.get("max_features_options", ["sqrt", "log2", "float"])
    max_features_type = trial.suggest_categorical("max_features_type", max_features_options)
    
    if max_features_type in ["sqrt", "log2"]:
        params["max_features"] = max_features_type
    else:
        params["max_features"] = trial.suggest_float(
            "max_features_float", 
            cfg.get("max_features_float_min", 0.5), 
            cfg.get("max_features_float_max", 1.0)
        )

    return params


def get_xgboost_search_space(trial: optuna.Trial, config: Dict[str, Any] = None) -> dict:
    """
    XGBoost search space.

    All ranges configurable via config dictionary.

    Parameters
    ----------
    trial : optuna.Trial
        Optuna trial object
    config : dict, optional
        Configuration dictionary with search ranges

    Returns
    -------
    dict
        Hyperparameter dictionary
    """
    if config is None:
        config = {}
    
    cfg = config.get("xgboost", {})
    
    return {
        "n_estimators": trial.suggest_int(
            "n_estimators", 
            cfg.get("n_estimators_min", 100), 
            cfg.get("n_estimators_max", 500)
        ),
        "max_depth": trial.suggest_int(
            "max_depth", 
            cfg.get("max_depth_min", 3), 
            cfg.get("max_depth_max", 20)
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate", 
            cfg.get("learning_rate_min", 1e-3), 
            cfg.get("learning_rate_max", 0.3), 
            log=True
        ),
        "subsample": trial.suggest_float(
            "subsample", 
            cfg.get("subsample_min", 0.5), 
            cfg.get("subsample_max", 1.0)
        ),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree", 
            cfg.get("colsample_bytree_min", 0.5), 
            cfg.get("colsample_bytree_max", 1.0)
        ),
        "reg_alpha": trial.suggest_float(
            "reg_alpha", 
            cfg.get("reg_alpha_min", 0.0), 
            cfg.get("reg_alpha_max", 1.0)
        ),
        "reg_lambda": trial.suggest_float(
            "reg_lambda", 
            cfg.get("reg_lambda_min", 0.0), 
            cfg.get("reg_lambda_max", 10.0)
        ),
        "min_child_weight": trial.suggest_int(
            "min_child_weight", 
            cfg.get("min_child_weight_min", 1), 
            cfg.get("min_child_weight_max", 20)
        ),
        "gamma": trial.suggest_float(
            "gamma", 
            cfg.get("gamma_min", 0.0), 
            cfg.get("gamma_max", 5.0)
        ),
    }


SEARCH_SPACE_FACTORY = {
    "en": get_elastic_net_search_space,
    "knn": get_knn_search_space,
    "dtc": get_decision_tree_search_space,
    "rf": get_random_forest_search_space,
    "xgb": get_xgboost_search_space,
}


def get_search_space(model_type: str, trial: optuna.Trial, config: Dict[str, Any] = None) -> dict:
    """
    Get search space for a given model type.

    Parameters
    ----------
    model_type : str
        Model type code ('en', 'knn', 'dtc', 'rf', 'xgb')
    trial : optuna.Trial
        Optuna trial object
    config : dict, optional
        Configuration dictionary with search ranges

    Returns
    -------
    dict
        Hyperparameter dictionary for the trial
    """
    if model_type not in SEARCH_SPACE_FACTORY:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Available: {list(SEARCH_SPACE_FACTORY.keys())}"
        )

    return SEARCH_SPACE_FACTORY[model_type](trial, config)

