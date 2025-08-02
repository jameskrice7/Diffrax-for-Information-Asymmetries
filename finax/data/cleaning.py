"""Data cleaning utilities for Finax."""

from __future__ import annotations

import pandas as pd


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values with forward fill."""
    return df.ffill()


def detect_outliers(df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
    """Replace values with NaN when their z-score exceeds ``threshold``."""
    numeric = df.select_dtypes("number")
    z = (numeric - numeric.mean()) / numeric.std(ddof=0)
    return df.mask(abs(z) > threshold)
