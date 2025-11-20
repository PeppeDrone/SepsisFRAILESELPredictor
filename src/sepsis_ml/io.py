"""
Data I/O module for loading and filtering the sepsis dataset.

Responsibilities:
- Load data from Excel file
- Detect and parse data types
- Create binary sepsis outcome from qSOFA ≥ 2
- Select pre-operative variables
- Apply ≥90% completion filter
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_data(
    filepath: Path,
    sheet_name: Optional[str] = None,
    outcome_col: str = "qSOFA",
    outcome_threshold: int = 2,
) -> pd.DataFrame:
    """
    Load sepsis dataset from Excel file.

    Parameters
    ----------
    filepath : Path
        Path to Excel file
    sheet_name : str, optional
        Sheet name to load. If None, loads first sheet.
    outcome_col : str
        Column name containing qSOFA scores or binary sepsis outcome
    outcome_threshold : int
        Threshold for sepsis label (sepsis if >= threshold).
        If outcome_col is already binary (only 0 and 1), threshold is ignored.

    Returns
    -------
    pd.DataFrame
        Loaded dataframe with 'sepsis' outcome column
    """
    logger.info(f"Loading data from {filepath}")

    # Read Excel file
    if sheet_name is None:
        # Auto-detect first sheet
        df = pd.read_excel(filepath, sheet_name=0)
        logger.info(f"Loaded first sheet with shape {df.shape}")
    else:
        df = pd.read_excel(filepath, sheet_name=sheet_name)
        logger.info(f"Loaded sheet '{sheet_name}' with shape {df.shape}")

    # Create binary sepsis outcome if not already present
    if "sepsis" not in df.columns:
        if outcome_col in df.columns:
            # Check if outcome_col is already binary (0/1)
            unique_values = df[outcome_col].dropna().unique()
            is_binary = set(unique_values).issubset({0, 1, 0.0, 1.0})
            
            if is_binary:
                # Already binary, use directly
                df["sepsis"] = df[outcome_col].astype(int)
                logger.info(
                    f"Using '{outcome_col}' directly as binary outcome (already 0/1)"
                )
            else:
                # Apply threshold
                df["sepsis"] = (df[outcome_col] >= outcome_threshold).astype(int)
                logger.info(
                    f"Created 'sepsis' outcome from {outcome_col} >= {outcome_threshold}"
                )
            
            logger.info(
                f"Sepsis prevalence: {df['sepsis'].sum()}/{len(df)} "
                f"({df['sepsis'].mean()*100:.1f}%)"
            )
        else:
            raise ValueError(
                f"Outcome column '{outcome_col}' not found and 'sepsis' column "
                f"not present. Available columns: {list(df.columns)}"
            )

    return df


def detect_preop_variables(
    df: pd.DataFrame,
    exclude_keywords: Optional[list] = None,
) -> list:
    """
    Detect pre-operative variables based on column names.

    This function attempts to identify pre-operative variables by excluding
    columns that contain post-operative keywords or are metadata/identifiers.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    exclude_keywords : list, optional
        Keywords to exclude (e.g., 'post', 'intra', 'outcome')

    Returns
    -------
    list
        List of pre-operative variable names
    """
    if exclude_keywords is None:
        exclude_keywords = [
            "post",
            "intra",
            "during",
            "after",
            "qsofa",
            "sepsis",
            "outcome",
            "id",
            "patient",
            "date",
            "time",
        ]

    preop_vars = []
    for col in df.columns:
        col_lower = col.lower()
        # Exclude if any keyword matches
        if any(keyword in col_lower for keyword in exclude_keywords):
            continue
        preop_vars.append(col)

    logger.info(f"Detected {len(preop_vars)} pre-operative variables")
    logger.debug(f"Pre-operative variables: {preop_vars[:10]}...")

    return preop_vars


def apply_completion_filter(
    df: pd.DataFrame,
    feature_cols: list,
    threshold: float = 0.90,
) -> Tuple[pd.DataFrame, list, dict]:
    """
    Filter features to retain only those with ≥ threshold completion rate.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    feature_cols : list
        List of feature column names to filter
    threshold : float
        Minimum completion rate (default: 0.90 for ≥90% completion)

    Returns
    -------
    df_filtered : pd.DataFrame
        Dataframe with only retained features
    retained_features : list
        List of feature names that passed the filter
    completion_rates : dict
        Dictionary mapping feature names to their completion rates (0-1)
    """
    completion_rates = {}
    for col in feature_cols:
        completion_rate = 1 - df[col].isna().mean()
        completion_rates[col] = completion_rate

    retained_features = [
        col for col, rate in completion_rates.items() if rate >= threshold
    ]

    logger.info(
        f"Completion filter (≥{threshold*100:.0f}%): "
        f"{len(retained_features)}/{len(feature_cols)} features retained"
    )

    # Log dropped features at INFO level
    dropped_features = set(feature_cols) - set(retained_features)
    if dropped_features:
        logger.info(f"Dropped {len(dropped_features)} features due to low completion:")
        for col in sorted(dropped_features):
            missing_pct = (1 - completion_rates[col]) * 100
            logger.info(f"  {col}: {completion_rates[col]*100:.1f}% complete ({missing_pct:.1f}% missing)")

    # Return df with retained features + outcome
    df_filtered = df[retained_features + ["sepsis"]].copy()

    return df_filtered, retained_features, completion_rates


def prepare_dataset(
    filepath: Path,
    completion_threshold: float = 0.90,
    sheet_name: Optional[str] = None,
    outcome_col: str = "qSOFA",
    outcome_threshold: int = 2,
    preop_keywords_exclude: Optional[list] = None,
    config: Optional[dict] = None,
) -> Tuple[pd.DataFrame, list]:
    """
    Complete data preparation pipeline.

    Steps:
    1. Load data from Excel
    2. Create binary sepsis outcome
    3. Detect pre-operative variables
    4. Apply ≥90% completion filter

    Parameters
    ----------
    filepath : Path
        Path to Excel data file
    completion_threshold : float
        Minimum completion rate for features (default: 0.90)
    sheet_name : str, optional
        Excel sheet name to load
    outcome_col : str
        Column name for qSOFA scores
    outcome_threshold : int
        Threshold for sepsis label
    preop_keywords_exclude : list, optional
        Keywords to exclude when detecting pre-op variables
    config : dict, optional
        Configuration dictionary (overrides other parameters if provided)

    Returns
    -------
    df_clean : pd.DataFrame
        Cleaned dataframe with retained features and 'sepsis' outcome
    feature_names : list
        List of feature column names
    completion_rates : dict
        Dictionary mapping all pre-operative feature names to their completion rates (0-1)
    """
    # Override with config if provided
    if config is not None:
        completion_threshold = config.get("completion_threshold", completion_threshold)
        outcome_col = config.get("outcome_column", outcome_col)
        outcome_threshold = config.get("outcome_threshold", outcome_threshold)
    # Load data
    df = load_data(
        filepath=filepath,
        sheet_name=sheet_name,
        outcome_col=outcome_col,
        outcome_threshold=outcome_threshold,
    )

    # Detect pre-operative variables
    preop_vars = detect_preop_variables(
        df=df,
        exclude_keywords=preop_keywords_exclude,
    )

    if not preop_vars:
        raise ValueError("No pre-operative variables detected!")

    # Apply completion filter
    df_clean, feature_names, completion_rates = apply_completion_filter(
        df=df,
        feature_cols=preop_vars,
        threshold=completion_threshold,
    )

    logger.info(f"Final dataset: {df_clean.shape[0]} samples, {len(feature_names)} features")
    
    # Check sepsis prevalence
    sepsis_count = df_clean['sepsis'].sum()
    sepsis_pct = df_clean['sepsis'].mean() * 100
    logger.info(f"Sepsis prevalence: {sepsis_count}/{len(df_clean)} ({sepsis_pct:.1f}%)")
    
    # Warn if no positive cases
    if sepsis_count == 0:
        logger.error("=" * 80)
        logger.error("CRITICAL: NO SEPSIS CASES FOUND IN DATA!")
        logger.error("=" * 80)
        logger.error(f"All {len(df_clean)} samples have sepsis=0")
        logger.error("Possible reasons:")
        logger.error("  1. No patients in data have qSOFA >= 2")
        logger.error("  2. Wrong outcome column being used")
        logger.error("  3. Pre-operative variable filtering removed all positive cases")
        logger.error("")
        logger.error("Please check your data's qSOFA column or sepsis outcome definition.")
        logger.error("=" * 80)
        raise ValueError(
            f"No sepsis cases (sepsis=1) found in the data. "
            f"All {len(df_clean)} samples have sepsis=0. "
            "Cannot train a classifier without positive examples. "
            "Check your qSOFA column or outcome definition."
        )
    
    if sepsis_pct < 5.0:
        logger.warning(
            f"Very low sepsis prevalence ({sepsis_pct:.1f}%). "
            "This may cause issues with cross-validation splits."
        )

    # Check for remaining missing values
    missing_pct = df_clean[feature_names].isna().mean() * 100
    if missing_pct.sum() > 0:
        logger.info(
            f"Remaining missing values (will be imputed): "
            f"{missing_pct[missing_pct > 0].to_dict()}"
        )

    return df_clean, feature_names, completion_rates

