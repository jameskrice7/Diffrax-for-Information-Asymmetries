"""Utilities for working with OHLCV market data."""

from __future__ import annotations

import pandas as pd


def _resample_ohlcv(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Resample intraday data to a desired frequency.

    Parameters
    ----------
    df:
        DataFrame containing ``open``, ``high``, ``low``, ``close`` and
        ``volume`` columns. Optionally, ``bid`` and ``ask`` columns will be
        used to compute bid-ask spreads.
    freq:
        Resample frequency such as ``'D'`` for daily or ``'M'`` for monthly.
    """

    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    if "bid" in df.columns:
        agg["bid"] = "first"
    if "ask" in df.columns:
        agg["ask"] = "last"

    out = df.resample(freq).agg(agg)
    if {"bid", "ask"}.issubset(out.columns):
        out["spread"] = out["ask"] - out["bid"]
    return out


def daily_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate intraday data to daily OHLCV records."""

    return _resample_ohlcv(df, "D")


def monthly_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate intraday data to monthly OHLCV records."""

    return _resample_ohlcv(df, "M")


def compute_bid_ask_spread(df: pd.DataFrame) -> pd.Series:
    """Return the bid-ask spread from ``bid`` and ``ask`` columns."""

    if "bid" not in df.columns or "ask" not in df.columns:
        raise KeyError("DataFrame must contain 'bid' and 'ask' columns")
    return df["ask"] - df["bid"]
