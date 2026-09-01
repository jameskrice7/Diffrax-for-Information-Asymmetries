"""Loading, cleaning and reshaping data.

Pandas-facing utilities, plus :mod:`finax.data.frames` which crosses the
boundary into JAX arrays.
"""

from .cleaning import clip_outliers, detect_outliers, fill_missing, winsorize
from .features import (
    bollinger_bands,
    event_flags,
    macd,
    realized_volatility,
    rolling_mean,
    rolling_volatility,
    rsi,
)
from .frames import align_frames, panel_to_batch, returns, to_arrays
from .ingestion import (
    fetch_url_csv,
    load_csv,
    load_excel,
    load_hdf5,
    load_hf_dataset,
    load_json,
    load_parquet,
    load_remote_csv,
    load_sqlite,
    stream_quotes,
)
from .ohlc import compute_bid_ask_spread, resample_ohlcv

__all__ = [
    # Ingestion
    "load_csv",
    "load_parquet",
    "load_json",
    "load_excel",
    "load_hdf5",
    "load_sqlite",
    "load_remote_csv",
    "load_hf_dataset",
    "fetch_url_csv",
    "stream_quotes",
    # Pandas <-> JAX
    "to_arrays",
    "panel_to_batch",
    "returns",
    "align_frames",
    # Cleaning
    "fill_missing",
    "detect_outliers",
    "clip_outliers",
    "winsorize",
    # Features
    "rolling_mean",
    "rolling_volatility",
    "realized_volatility",
    "rsi",
    "macd",
    "bollinger_bands",
    "event_flags",
    # OHLC
    "resample_ohlcv",
    "compute_bid_ask_spread",
]
