
"""Data utilities for Finax."""

from .ingestion import (
    load_csv,
    load_parquet,
    load_json,
    load_excel,
    load_hdf5,
    load_sqlite,
    load_remote_csv,
    load_hf_dataset,
)
from .cleaning import fill_missing, detect_outliers
from .features import rolling_mean, technical_indicator


"""Data utilities for Finax.

This submodule collects helpers for loading, cleaning and engineering
financial time series.

Examples
--------
>>> import pandas as pd
>>> from finax.data import rsi, event_flags
>>> prices = pd.Series([1, 2, 3], index=pd.date_range("2024-01-01", periods=3))
>>> rsi(prices, window=2).round(0).tolist()
[nan, 100.0, 100.0]
>>> events = pd.DataFrame({"date": [pd.Timestamp("2024-01-02")], "event": ["earnings"]})
>>> event_flags(prices.to_frame("price"), events).loc["2024-01-02", "earnings"]
1
"""

from .ingestion import load_csv, load_parquet, load_json, fetch_yahoo
from .cleaning import fill_missing, detect_outliers
from .features import (
    rolling_mean,
    rsi,
    macd,
    bollinger_bands,
    rolling_volatility,
    event_flags,
)

from .eikon import fetch_eikon

__all__ = [
    "load_csv",
    "load_parquet",
    "load_json",
    "load_excel",
    "load_hdf5",
    "load_sqlite",
    "load_remote_csv",
    "load_hf_dataset",
    "fetch_eikon",
    "fill_missing",
    "detect_outliers",
    "rolling_mean",
    "technical_indicator",
    "rsi",
    "macd",
    "bollinger_bands",
    "rolling_volatility",
    "event_flags",
]
