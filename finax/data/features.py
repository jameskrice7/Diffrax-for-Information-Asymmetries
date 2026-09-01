"""Feature engineering for price series."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..errors import DataValidationError, require

__all__ = [
    "rolling_mean",
    "rolling_volatility",
    "realized_volatility",
    "rsi",
    "macd",
    "bollinger_bands",
    "event_flags",
]


def rolling_mean(series: Any, window: int, *, min_periods: int | None = None):
    """Rolling mean over ``window`` observations."""
    require("pandas", purpose="feature engineering")
    return series.rolling(window, min_periods=min_periods).mean()


def rolling_volatility(
    series: Any, window: int, *, annualize: int | None = None, log: bool = True
):
    """Rolling standard deviation of returns.

    Parameters
    ----------
    annualize:
        Periods per year (252 for daily data). Scales by ``sqrt(periods)``.
    log:
        Use log returns rather than simple returns.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> s = pd.Series(100 * np.exp(np.cumsum(np.full(300, 0.001))))
    >>> vol = rolling_volatility(s, 20)
    >>> bool(vol.iloc[-1] < 1e-6)  # constant growth has no volatility
    True
    """
    require("pandas", purpose="feature engineering")
    rets = np.log(series).diff() if log else series.pct_change()
    vol = rets.rolling(window).std()
    if annualize is not None:
        vol = vol * np.sqrt(annualize)
    return vol


def realized_volatility(series: Any, window: int, *, annualize: int | None = None):
    """Realized volatility: the square root of summed squared log returns.

    Distinct from :func:`rolling_volatility`, which uses a *centred* standard
    deviation. Realized volatility does not subtract a mean, matching the
    quadratic-variation definition that high-frequency econometrics uses -- over
    short horizons the drift is negligible and estimating it only adds noise.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> rng = np.random.default_rng(0)
    >>> s = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 2000))))
    >>> rv = realized_volatility(s, 250, annualize=252)
    >>> bool(abs(rv.iloc[-1] - 0.01 * np.sqrt(252)) < 0.02)
    True
    """
    require("pandas", purpose="feature engineering")
    log_returns = np.log(series).diff()
    rv = np.sqrt((log_returns**2).rolling(window).sum() / window)
    if annualize is not None:
        rv = rv * np.sqrt(annualize)
    return rv


def rsi(series: Any, window: int = 14):
    """Relative Strength Index, using Wilder's smoothing.

    Wilder's original formulation uses an exponentially weighted average with
    ``alpha = 1/window``, not a simple rolling mean. The two give visibly
    different values, and charting packages universally implement Wilder's.

    Returns values in ``[0, 100]``.

    Examples
    --------
    >>> import pandas as pd
    >>> up = pd.Series(range(1, 60), dtype=float)
    >>> float(rsi(up).iloc[-1])  # a series that only ever rises
    100.0
    >>> down = pd.Series(range(60, 1, -1), dtype=float)
    >>> float(rsi(down).iloc[-1])
    0.0
    """
    require("pandas", purpose="feature engineering")
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    # avg_loss == 0 means an unbroken run of gains: RSI is 100 by definition.
    return out.where(avg_loss != 0.0, 100.0).where(avg_gain != 0.0, 0.0)


def macd(series: Any, *, fast: int = 12, slow: int = 26, signal: int = 9):
    """Moving Average Convergence Divergence.

    Returns a DataFrame with ``macd``, ``signal`` and ``hist`` columns.

    Examples
    --------
    >>> import pandas as pd
    >>> out = macd(pd.Series(range(100), dtype=float))
    >>> list(out.columns)
    ['macd', 'signal', 'hist']
    """
    pd = require("pandas", purpose="feature engineering")
    if not fast < slow:
        raise DataValidationError(f"Need fast < slow, got fast={fast}, slow={slow}.")
    fast_ema = series.ewm(span=fast, adjust=False).mean()
    slow_ema = series.ewm(span=slow, adjust=False).mean()
    line = fast_ema - slow_ema
    signal_line = line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({"macd": line, "signal": signal_line, "hist": line - signal_line})


def bollinger_bands(series: Any, *, window: int = 20, num_std: float = 2.0):
    """Bollinger Bands.

    Returns a DataFrame with ``middle``, ``upper``, ``lower`` and ``bandwidth``.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> out = bollinger_bands(pd.Series(np.arange(50, dtype=float)))
    >>> list(out.columns)
    ['middle', 'upper', 'lower', 'bandwidth']
    >>> bool((out["upper"].dropna() >= out["lower"].dropna()).all())
    True
    """
    pd = require("pandas", purpose="feature engineering")
    middle = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    return pd.DataFrame(
        {
            "middle": middle,
            "upper": upper,
            "lower": lower,
            "bandwidth": (upper - lower) / middle,
        }
    )


def event_flags(
    df: Any,
    events: Any,
    *,
    date_column: str = "date",
    event_column: str = "event",
):
    """Annotate a date-indexed frame with a binary column per event type.

    Useful for information-asymmetry work, where the object of study is often
    behaviour around scheduled disclosures -- earnings dates, guidance, filings.

    Parameters
    ----------
    df:
        Frame with a ``DatetimeIndex``.
    events:
        Frame with date and event-name columns.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"p": [1.0, 2.0, 3.0]},
    ...                   index=pd.to_datetime(["2024-01-01", "2024-01-02",
    ...                                         "2024-01-03"]))
    >>> ev = pd.DataFrame({"date": pd.to_datetime(["2024-01-02"]),
    ...                    "event": ["earnings"]})
    >>> event_flags(df, ev)["earnings"].tolist()
    [0, 1, 0]
    """
    pd = require("pandas", purpose="feature engineering")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise DataValidationError("df must have a DatetimeIndex.")
    for column in (date_column, event_column):
        if column not in events.columns:
            raise DataValidationError(
                f"events is missing column {column!r}; has {list(events.columns)}."
            )

    flagged = df.copy()
    dates = pd.to_datetime(events[date_column]).dt.normalize()
    index_dates = flagged.index.normalize()
    for name, group in events.assign(**{date_column: dates}).groupby(event_column):
        flagged[name] = index_dates.isin(group[date_column]).astype(int)
    return flagged
