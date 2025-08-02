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
from .ohlc import daily_ohlcv, monthly_ohlcv, compute_bid_ask_spread
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
    "daily_ohlcv",
    "monthly_ohlcv",
    "compute_bid_ask_spread",
]
