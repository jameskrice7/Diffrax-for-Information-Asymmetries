"""Feature engineering utilities for Finax."""

from __future__ import annotations

import pandas as pd


def rolling_mean(series: pd.Series, window: int) -> pd.Series:
    """Compute a rolling mean over ``window`` observations."""
    return series.rolling(window).mean()


def technical_indicator(series: pd.Series) -> pd.Series:
    """Placeholder for a technical indicator such as RSI."""
    raise NotImplementedError("Indicator not implemented.")
