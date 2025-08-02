"""Data ingestion utilities for Finax."""

from __future__ import annotations

from typing import Optional

import pandas as pd


def load_csv(path: str, *, parse_dates: Optional[list[str]] = None) -> pd.DataFrame:
    """Load CSV financial data into a DataFrame."""
    return pd.read_csv(path, parse_dates=parse_dates)


def load_parquet(path: str) -> pd.DataFrame:
    """Load Parquet financial data into a DataFrame."""
    return pd.read_parquet(path)


def load_json(path: str) -> pd.DataFrame:
    """Load JSON financial data into a DataFrame."""
    return pd.read_json(path)


def fetch_yahoo(symbol: str) -> pd.DataFrame:
    """Placeholder for Yahoo Finance API connector."""
    raise NotImplementedError("API connector not implemented.")
