"""OHLCV bar construction from intraday data."""

from __future__ import annotations

from typing import Any

from ..errors import DataValidationError, require

__all__ = ["resample_ohlcv", "compute_bid_ask_spread"]


def resample_ohlcv(df: Any, freq: str = "D", *, price_column: str | None = None):
    """Resample trade or bar data to a coarser frequency.

    Parameters
    ----------
    df:
        Frame with a ``DatetimeIndex``. Either supply ``open``/``high``/``low``/
        ``close``/``volume`` columns, or a single ``price_column`` from which
        OHLC is derived.
    freq:
        Any pandas offset alias: ``"D"``, ``"W"``, ``"ME"``, ``"5min"``, ...
    price_column:
        Build OHLC from this trade-price column instead of existing OHLC
        columns.

    Examples
    --------
    From tick prices:

    >>> import pandas as pd, numpy as np
    >>> idx = pd.date_range("2024-01-01", periods=48, freq="h")
    >>> ticks = pd.DataFrame({"price": np.arange(48.0), "volume": 1.0}, index=idx)
    >>> bars = resample_ohlcv(ticks, "D", price_column="price")
    >>> bars[["open", "high", "low", "close"]].iloc[0].tolist()
    [0.0, 23.0, 0.0, 23.0]
    >>> bars["volume"].tolist()
    [24.0, 24.0]
    """
    pd = require("pandas", purpose="OHLCV resampling")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise DataValidationError("df must have a DatetimeIndex.")

    if price_column is not None:
        if price_column not in df.columns:
            raise DataValidationError(
                f"price_column {price_column!r} not in {list(df.columns)}."
            )
        out = df[price_column].resample(freq).ohlc()
        if "volume" in df.columns:
            out["volume"] = df["volume"].resample(freq).sum()
    else:
        required = {"open", "high", "low", "close"}
        missing = required - set(df.columns)
        if missing:
            raise DataValidationError(
                f"Missing OHLC columns {sorted(missing)}; pass price_column to "
                "build them from trade prices instead."
            )
        agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
        if "volume" in df.columns:
            agg["volume"] = "sum"
        out = df.resample(freq).agg(agg)

    if "bid" in df.columns and "ask" in df.columns:
        out["bid"] = df["bid"].resample(freq).last()
        out["ask"] = df["ask"].resample(freq).last()
        out["spread"] = out["ask"] - out["bid"]

    return out.dropna(how="all")


def compute_bid_ask_spread(df: Any, *, relative: bool = False):
    """Return the bid-ask spread from ``bid`` and ``ask`` columns.

    Parameters
    ----------
    relative:
        Divide by the midpoint, giving a proportional spread. This is what you
        want for comparisons across stocks, since an absolute spread of one cent
        means very different things at $5 and at $500.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"bid": [99.0], "ask": [101.0]})
    >>> float(compute_bid_ask_spread(df).iloc[0])
    2.0
    >>> float(compute_bid_ask_spread(df, relative=True).iloc[0])
    0.02
    """
    require("pandas", purpose="spread computation")
    missing = {"bid", "ask"} - set(df.columns)
    if missing:
        raise DataValidationError(f"DataFrame is missing columns {sorted(missing)}.")
    spread = df["ask"] - df["bid"]
    if relative:
        return spread / (0.5 * (df["ask"] + df["bid"]))
    return spread
