"""Utility helpers for loading the Iris dataset."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.datasets import load_iris

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _clean_column(name: str) -> str:
    return name.replace(" (cm)", "").replace(" ", "_")


def load_iris_df() -> Tuple[pd.DataFrame, pd.Series, list[str]]:
    """Return feature matrix, label vector, and target names."""
    iris = load_iris(as_frame=True)
    frame = iris.frame.copy()
    frame.columns = [_clean_column(col) for col in frame.columns]
    feature_cols = [col for col in frame.columns if col != "target"]
    X = frame[feature_cols]
    y = frame["target"].astype(int)
    target_names = list(iris.target_names)
    return X, y, target_names


__all__ = ["load_iris_df"]
