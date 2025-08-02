import sys
from pathlib import Path

import pandas as pd
import pandas.testing as tm

# Ensure package root is on the import path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from finax.data import (
    rsi,
    macd,
    bollinger_bands,
    rolling_volatility,
    event_flags,
)


def test_rsi_basic():
    s = pd.Series(range(1, 11), dtype=float)
    result = rsi(s, window=2)
    assert pd.isna(result.iloc[0])
    assert result.iloc[-1] == 100


def test_macd_matches_manual():
    s = pd.Series(range(1, 11), dtype=float)
    out = macd(s)
    fast_ema = s.ewm(span=12, adjust=False).mean()
    slow_ema = s.ewm(span=26, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    hist = macd_line - signal_line
    tm.assert_series_equal(out["macd"], macd_line, check_names=False)
    tm.assert_series_equal(out["signal"], signal_line, check_names=False)
    tm.assert_series_equal(out["hist"], hist, check_names=False)


def test_bollinger_bands():
    s = pd.Series(range(1, 21), dtype=float)
    bands = bollinger_bands(s, window=5, num_std=2)
    middle = s.rolling(5).mean()
    std = s.rolling(5).std()
    upper = middle + 2 * std
    lower = middle - 2 * std
    tm.assert_series_equal(bands["middle"], middle, check_names=False)
    tm.assert_series_equal(bands["upper"], upper, check_names=False)
    tm.assert_series_equal(bands["lower"], lower, check_names=False)


def test_rolling_volatility():
    s = pd.Series([1, 2, 4, 8, 16], dtype=float)
    vol = rolling_volatility(s, window=2)
    expected = s.pct_change().rolling(2).std()
    tm.assert_series_equal(vol, expected)


def test_event_flags():
    idx = pd.date_range("2024-01-01", periods=3)
    df = pd.DataFrame({"price": [1, 2, 3]}, index=idx)
    events = pd.DataFrame({"date": [pd.Timestamp("2024-01-02")], "event": ["earnings"]})
    flagged = event_flags(df, events)
    assert "earnings" in flagged.columns
    assert flagged.loc[pd.Timestamp("2024-01-02"), "earnings"] == 1
    assert flagged["earnings"].sum() == 1
