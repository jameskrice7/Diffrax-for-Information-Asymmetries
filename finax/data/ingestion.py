"""Data ingestion utilities for Finax."""

from __future__ import annotations

from typing import Optional

import pandas as pd


def load_csv(path: str, *, parse_dates: Optional[list[str]] = None) -> pd.DataFrame:
    """Load CSV financial data into a DataFrame.

    Parameters
    ----------
    path:
        Local file path to a CSV file.
    parse_dates:
        Optional list of column names to parse as dates.
    """

    return pd.read_csv(path, parse_dates=parse_dates)


def load_parquet(path: str) -> pd.DataFrame:
    """Load Parquet financial data into a DataFrame."""
    return pd.read_parquet(path)


def load_json(path: str) -> pd.DataFrame:
    """Load JSON financial data into a DataFrame."""
    return pd.read_json(path)


def load_excel(path: str, *, sheet_name: str | int | None = 0) -> pd.DataFrame:
    """Load Excel financial data into a DataFrame.

    Parameters
    ----------
    path:
        Location of the Excel file.
    sheet_name:
        Sheet within the workbook to read. Defaults to the first sheet.
    """

    return pd.read_excel(path, sheet_name=sheet_name)


def load_hdf5(path: str, key: str = "data") -> pd.DataFrame:
    """Load HDF5 financial data into a DataFrame."""

    return pd.read_hdf(path, key=key)


def load_sqlite(path: str, query: str) -> pd.DataFrame:
    """Load data from a SQLite database using a SQL query."""
    import sqlite3

    with sqlite3.connect(path) as conn:
        return pd.read_sql_query(query, conn)



def load_remote_csv(url: str, *, parse_dates: Optional[list[str]] = None) -> pd.DataFrame:
    """Load a remote CSV file directly into a DataFrame using pandas."""

    return pd.read_csv(url, parse_dates=parse_dates)


def load_hf_dataset(name: str, *, split: str = "train", **kwargs) -> pd.DataFrame:
    """Load a dataset hosted on the Hugging Face Hub into a DataFrame.

    Parameters
    ----------
    name:
        Dataset identifier on the Hub.
    split:
        Which split to load (e.g. ``"train"`` or ``"test"``).
    **kwargs:
        Additional keyword arguments forwarded to ``datasets.load_dataset``.
    """

    from datasets import load_dataset  # type: ignore

    ds = load_dataset(name, split=split, **kwargs)
    return ds.to_pandas()

