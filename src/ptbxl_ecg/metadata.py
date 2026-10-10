"""Metadata exploration, SCP-ECG diagnostic mapping, and stratified dataset partitioning."""

import ast
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd

from ptbxl_ecg.config import (
    SUPERCLASSES,
    TRAIN_FOLDS,
    VAL_FOLDS,
    TEST_FOLDS,
)


def parse_scp_statements(raw_val) -> dict:
    """Safely parse stringified Python dictionary from CSV column."""
    if isinstance(raw_val, dict):
        return raw_val
    if pd.isna(raw_val) or not str(raw_val).strip():
        return {}
    try:
        return ast.literal_eval(str(raw_val))
    except (ValueError, SyntaxError):
        return {}


def map_to_superclasses(
    scp_dict: dict,
    diag_ref: pd.DataFrame,
    superclasses: List[str] = SUPERCLASSES,
) -> List[str]:
    """Map granular SCP-ECG diagnostic statements to official superclasses.

    Parameters:
        scp_dict: Dictionary of {scp_code: likelihood_score}.
        diag_ref: DataFrame indexed by scp_code with 'diagnostic_class' column.
        superclasses: List of accepted diagnostic superclasses.

    Returns:
        Sorted list of unique diagnostic superclass labels.
    """
    assigned = set()
    for code in scp_dict.keys():
        if code in diag_ref.index:
            s_class = diag_ref.loc[code, "diagnostic_class"]
            if pd.notna(s_class) and str(s_class) in superclasses:
                assigned.add(str(s_class))
    return sorted(list(assigned))


def process_ptbxl_metadata(
    df_raw: pd.DataFrame,
    df_scp: pd.DataFrame,
    superclasses: List[str] = SUPERCLASSES,
) -> pd.DataFrame:
    """Enrich raw PTB-XL metadata with parsed superclass annotations and binary targets.

    Parameters:
        df_raw: Raw DataFrame loaded from ptbxl_database.csv.
        df_scp: Raw DataFrame loaded from scp_statements.csv.
        superclasses: List of superclasses to extract.

    Returns:
        Processed DataFrame with binary target columns and diagnostic superclasses.
    """
    df = df_raw.copy()

    # Ensure scp_codes column is converted to Python dictionaries
    if "scp_codes_dict" not in df.columns:
        df["scp_codes_dict"] = df["scp_codes"].apply(parse_scp_statements)

    # Filter scp reference dataframe to diagnostic entries only
    if "diagnostic" in df_scp.columns:
        diag_ref = df_scp[df_scp["diagnostic"] == 1.0]
    else:
        diag_ref = df_scp

    # Map statements to superclasses
    df["diagnostic_superclasses"] = df["scp_codes_dict"].apply(
        lambda x: map_to_superclasses(x, diag_ref, superclasses=superclasses)
    )

    # Encode multi-hot binary indicator columns for each superclass
    for sc in superclasses:
        df[sc] = df["diagnostic_superclasses"].apply(lambda tags: 1 if sc in tags else 0)

    # Filter out entries lacking any diagnostic superclass
    has_labels = df[superclasses].sum(axis=1) > 0
    df_labelled = df[has_labels].copy()

    return df_labelled


def create_stratified_splits(
    df: pd.DataFrame,
    train_folds: List[int] = TRAIN_FOLDS,
    val_folds: List[int] = VAL_FOLDS,
    test_folds: List[int] = TEST_FOLDS,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Partition dataset into Train, Validation, and Test subsets based on strat_fold.

    Parameters:
        df: Metadata DataFrame containing 'strat_fold' column (values 1 to 10).
        train_folds: Fold identifiers assigned to training split (typically 1 to 8).
        val_folds: Fold identifiers assigned to validation split (typically 9).
        test_folds: Fold identifiers assigned to testing split (typically 10).

    Returns:
        Tuple of (df_train, df_val, df_test).
    """
    if "strat_fold" not in df.columns:
        raise KeyError("Column 'strat_fold' not found in DataFrame.")

    df_train = df[df["strat_fold"].isin(train_folds)].copy()
    df_val = df[df["strat_fold"].isin(val_folds)].copy()
    df_test = df[df["strat_fold"].isin(test_folds)].copy()

    return df_train, df_val, df_test


def get_dataset_summary_statistics(
    df: pd.DataFrame,
    superclasses: List[str] = SUPERCLASSES,
) -> Dict[str, any]:
    """Compute demographic and clinical epidemiological summary statistics.

    Parameters:
        df: Processed PTB-XL DataFrame.
        superclasses: List of superclass column names.

    Returns:
        Dictionary containing counts, percentages, and demographic distributions.
    """
    total_records = len(df)
    prevalence = {sc: int(df[sc].sum()) for sc in superclasses if sc in df.columns}
    prevalence_pct = {
        sc: round(float(count) / total_records * 100, 2)
        for sc, count in prevalence.items()
    }

    stats = {
        "total_records": total_records,
        "age_mean": float(df["age"].mean()) if "age" in df.columns else None,
        "age_std": float(df["age"].std()) if "age" in df.columns else None,
        "age_min": float(df["age"].min()) if "age" in df.columns else None,
        "age_max": float(df["age"].max()) if "age" in df.columns else None,
        "male_count": int((df["sex"] == 0).sum()) if "sex" in df.columns else None,
        "female_count": int((df["sex"] == 1).sum()) if "sex" in df.columns else None,
        "superclass_counts": prevalence,
        "superclass_prevalence_pct": prevalence_pct,
    }
    return stats
