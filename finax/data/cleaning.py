"""Cleaning utilities for financial DataFrames."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..errors import DataValidationError, require

__all__ = ["fill_missing", "detect_outliers", "clip_outliers", "winsorize"]


def fill_missing(df: Any, *, method: str = "ffill", limit: int | None = None):
    """Fill missing values.

    Parameters
    ----------
    method:
        ``"ffill"`` forward fill, ``"bfill"`` backward fill, ``"zero"``, or
        ``"interpolate"`` for time-weighted linear interpolation.
    limit:
        Maximum number of consecutive NaNs to fill. Leaving this unset on a
        series with long gaps propagates a stale price across the whole gap,
        which shows up later as spurious zero-volatility periods -- set it.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> df = pd.DataFrame({"x": [1.0, np.nan, np.nan, 4.0]})
    >>> fill_missing(df)["x"].tolist()
    [1.0, 1.0, 1.0, 4.0]
    >>> fill_missing(df, limit=1)["x"].tolist()
    [1.0, 1.0, nan, 4.0]
    """
    require("pandas", purpose="data cleaning")
    if method == "ffill":
        return df.ffill(limit=limit)
    if method == "bfill":
        return df.bfill(limit=limit)
    if method == "zero":
        return df.fillna(0.0)
    if method == "interpolate":
        return df.interpolate(limit=limit)
    raise DataValidationError(
        f"method must be 'ffill', 'bfill', 'zero' or 'interpolate', got {method!r}."
    )


def detect_outliers(df: Any, *, threshold: float = 3.0, robust: bool = True):
    """Return a boolean mask of outlying numeric entries.

    Parameters
    ----------
    threshold:
        Number of (robust) standard deviations beyond which a point is flagged.
    robust:
        Use the median and the median absolute deviation rather than the mean
        and standard deviation. Strongly recommended: the classical z-score is
        computed *from* the data it is testing, so a single extreme outlier
        inflates the standard deviation enough to mask itself. The MAD is scaled
        by 1.4826 to be consistent with the standard deviation under normality.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"x": [1.0, 1.1, 0.9, 1.05, 100.0]})
    >>> detect_outliers(df)["x"].tolist()
    [False, False, False, False, True]

    The non-robust z-score misses it, because the outlier inflates its own
    denominator:

    >>> detect_outliers(df, robust=False)["x"].tolist()
    [False, False, False, False, False]
    """
    require("pandas", purpose="data cleaning")
    numeric = df.select_dtypes(include=[np.number])
    if robust:
        centre = numeric.median()
        scale = (numeric - centre).abs().median() * 1.4826
    else:
        centre = numeric.mean()
        scale = numeric.std(ddof=0)
    scale = scale.replace(0.0, np.nan)
    return ((numeric - centre).abs() / scale) > threshold


def clip_outliers(df: Any, *, threshold: float = 3.0, robust: bool = True):
    """Replace outliers with NaN, leaving the rest of the frame untouched.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"x": [1.0, 1.1, 0.9, 1.05, 100.0]})
    >>> clip_outliers(df)["x"].tolist()
    [1.0, 1.1, 0.9, 1.05, nan]
    """
    mask = detect_outliers(df, threshold=threshold, robust=robust)
    return df.mask(mask.reindex(columns=df.columns, fill_value=False))


def winsorize(df: Any, *, lower: float = 0.01, upper: float = 0.99):
    """Clip numeric columns to the given quantiles.

    Preferred over dropping outliers in asset pricing, where extreme returns are
    real events rather than errors: winsorizing limits their leverage on an
    estimate without discarding the observation.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"x": list(range(101))})
    >>> out = winsorize(df, lower=0.05, upper=0.95)
    >>> float(out["x"].min()), float(out["x"].max())
    (5.0, 95.0)
    """
    require("pandas", purpose="data cleaning")
    if not 0.0 <= lower < upper <= 1.0:
        raise DataValidationError(
            f"Need 0 <= lower < upper <= 1, got lower={lower}, upper={upper}."
        )
    result = df.copy()
    numeric = result.select_dtypes(include=[np.number]).columns
    for column in numeric:
        low, high = result[column].quantile([lower, upper])
        result[column] = result[column].clip(low, high)
    return result
