"""Feature engineering utilities for Finax."""

from __future__ import annotations

import pandas as pd


def rolling_mean(series: pd.Series, window: int) -> pd.Series:
    """Compute a rolling mean over ``window`` observations."""
    return series.rolling(window).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Compute the Relative Strength Index (RSI).

    Parameters
    ----------
    series:
        Price series.
    window:
        Number of periods to use for averaging gains and losses.
    """
    diff = series.diff()
    gain = diff.clip(lower=0)
    loss = -diff.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    rsi_val = 100 - 100 / (1 + rs)
    rsi_val = rsi_val.where(avg_loss != 0, 100)
    rsi_val = rsi_val.where(avg_gain != 0, 0)
    return rsi_val


def macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    """Compute the Moving Average Convergence Divergence (MACD).

    Returns a ``DataFrame`` with ``macd``, ``signal`` and ``hist`` columns.
    """
    fast_ema = series.ewm(span=fast, adjust=False).mean()
    slow_ema = series.ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "signal": signal_line, "hist": hist})


def bollinger_bands(
    series: pd.Series, window: int = 20, num_std: float = 2.0
) -> pd.DataFrame:
    """Compute Bollinger Bands.

    Returns a ``DataFrame`` with ``middle``, ``upper`` and ``lower`` bands.
    """
    middle = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    return pd.DataFrame({"middle": middle, "upper": upper, "lower": lower})


def rolling_volatility(series: pd.Series, window: int) -> pd.Series:
    """Compute the rolling volatility from percentage returns."""
    returns = series.pct_change()
    return returns.rolling(window).std()


def event_flags(df: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Annotate price ``df`` with binary flags for specified ``events``.

    Parameters
    ----------
    df:
        DataFrame with a ``DatetimeIndex``.
    events:
        DataFrame with ``date`` and ``event`` columns. ``date`` should be
        convertible to ``datetime``.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df must have a DatetimeIndex")

    flagged = df.copy()
    events = events.copy()
    events["date"] = pd.to_datetime(events["date"])

    for name, group in events.groupby("event"):
        flagged[name] = (
            flagged.index.normalize().isin(group["date"].dt.normalize()).astype(int)
        )

    return flagged
