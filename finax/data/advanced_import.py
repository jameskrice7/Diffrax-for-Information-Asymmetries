"""Advanced import utilities for high-dimensional and sparse datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def load_sparse_csv(path: str | Path, *, index_col: str | None = None) -> pd.DataFrame:
    """Load a CSV and convert numeric columns to sparse dtype.

    This is useful when the dataset contains many zeros and should be kept
    memory efficient while still behaving like a DataFrame.
    """

    df = pd.read_csv(path)
    if index_col is not None:
        df = df.set_index(index_col)

    for column in df.select_dtypes(include=["number"]).columns:
        df[column] = pd.arrays.SparseArray(df[column], fill_value=0)
    return df


def densify_sparse_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Convert sparse columns back to dense columns."""

    dense = df.copy()
    for column in dense.columns:
        series = dense[column]
        if isinstance(series.dtype, pd.SparseDtype):
            dense[column] = series.sparse.to_dense()
    return dense


def select_high_dimensional_block(
    df: pd.DataFrame,
    *,
    include_columns: Iterable[str] | None = None,
    min_numeric_columns: int = 10,
) -> pd.DataFrame:
    """Extract the high-dimensional numeric subset of a DataFrame.

    Parameters
    ----------
    include_columns:
        Optional explicit list of columns to keep.
    min_numeric_columns:
        Minimum number of numeric columns required.
    """

    if include_columns is not None:
        block = df.loc[:, list(include_columns)]
    else:
        block = df.select_dtypes(include=[np.number])

    if block.shape[1] < min_numeric_columns:
        raise ValueError(
            f"Expected at least {min_numeric_columns} numeric columns, got {block.shape[1]}."
        )
    return block
