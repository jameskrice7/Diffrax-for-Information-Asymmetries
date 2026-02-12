import pandas as pd
import numpy as np
import pytest
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from finax.data import (
    fill_missing,
    detect_outliers,
    rolling_mean,
    daily_ohlcv,
    monthly_ohlcv,
    compute_bid_ask_spread,
)


def test_fill_missing():
    df = pd.DataFrame({"price": [1.0, np.nan, 3.0]})
    filled = fill_missing(df)
    assert filled.loc[1, "price"] == 1.0


def test_detect_outliers():
    df = pd.DataFrame({"price": [1.0, 2.0, 100.0]})
    cleaned = detect_outliers(df, threshold=1.0)
    assert pd.isna(cleaned.loc[2, "price"])


def test_rolling_mean():
    series = pd.Series([1.0, 2.0, 3.0])
    result = rolling_mean(series, window=2)
    assert pd.isna(result.iloc[0])
    assert result.iloc[1] == 1.5
    assert result.iloc[2] == 2.5


def intraday_df():
    idx = pd.date_range("2020-01-01", periods=4, freq="h")
    return pd.DataFrame(
        {
            "open": [1, 2, 3, 4],
            "high": [2, 3, 4, 5],
            "low": [0, 1, 2, 3],
            "close": [1.5, 2.5, 3.5, 4.5],
            "volume": [10, 20, 30, 40],
            "bid": [0.9, 1.9, 2.9, 3.9],
            "ask": [1.1, 2.1, 3.1, 4.1],
        },
        index=idx,
    )


def test_daily_ohlcv():
    df = intraday_df()
    daily = daily_ohlcv(df)
    assert len(daily) == 1
    assert "spread" in daily.columns


def test_monthly_ohlcv():
    df = intraday_df()
    monthly = monthly_ohlcv(df)
    assert len(monthly) == 1


def test_compute_bid_ask_spread():
    df = pd.DataFrame({"bid": [1.0, 1.5], "ask": [1.1, 1.7]})
    spread = compute_bid_ask_spread(df)
    assert spread.tolist() == pytest.approx([0.1, 0.2])
