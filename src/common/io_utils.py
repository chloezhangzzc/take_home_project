"""
Tiny IO helpers for reading data, splitting, seeding, and saving artifacts.
"""

import os, json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def set_global_seed(seed: int = 42):
    """Set seeds for numpy (extend here if you add torch, random, etc.)."""
    np.random.seed(seed)


def read_data(path: str) -> pd.DataFrame:
    """Read CSV/Parquet by extension; add logic as needed."""
    if path.endswith(".csv"):
        return pd.read_csv(path)
    elif path.endswith(".parquet"):
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {path}")


def stratified_splits(df, y_col, test_size, val_size, seed):
    """
    Produce train/val/test splits with stratification on the label.
    Returns (X_train, y_train, X_val, y_val, X_test, y_test, w_train, w_val, w_test).
    """
    y = df[y_col]
    X = df.drop(columns=[y_col])
    # Optional weight col is handled by caller
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_size, stratify=y_trainval, random_state=seed
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


def ensure_dir(path: str):
    """Create directory if missing."""
    os.makedirs(path, exist_ok=True)