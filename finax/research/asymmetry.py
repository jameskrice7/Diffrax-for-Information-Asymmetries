"""Information asymmetry metrics for financial research."""

from __future__ import annotations

import pandas as pd


def probability_of_informed_trading(buys: pd.Series, sells: pd.Series) -> float:
    """Estimate the probability of informed trading (PIN).

    This simplified estimator follows Easley et al. (1996) and computes PIN as
    the average order imbalance normalized by total order flow.

    Parameters
    ----------
    buys, sells:
        Series of buy and sell order counts for a given period.
    """

    total = buys + sells
    imbalance = (buys - sells).abs()
    return (imbalance / total).mean()


def vpin(volume: pd.Series, price: pd.Series, window: int = 50) -> pd.Series:
    """Compute the VPIN (Volume-synchronized PIN) metric.

    The algorithm follows Easley et al. (2012) using a rolling window of volume
    buckets where order flow is classified by price changes.

    Parameters
    ----------
    volume:
        Trade volume series.
    price:
        Trade price series aligned with ``volume``.
    window:
        Number of buckets to use for the rolling computation.
    """

    price_diff = price.diff().fillna(0)
    buy_volume = volume.where(price_diff > 0, 0.0)
    sell_volume = volume.where(price_diff <= 0, 0.0)
    vol_imbalance = (buy_volume - sell_volume).abs()
    rolling_imbalance = vol_imbalance.rolling(window).sum()
    rolling_volume = volume.rolling(window).sum()
    return rolling_imbalance / rolling_volume


def information_asymmetry_index(spread: pd.Series, volume: pd.Series) -> float:
    """Naive information asymmetry index based on spreads and volume.

    The index averages the bid-ask spread scaled by traded volume, providing a
    rough proxy for market microstructure frictions.
    """

    return (spread / volume).mean()


def pin_from_daily_prices(data: pd.DataFrame) -> float:
    """Estimate PIN using daily OHLCV data.

    This heuristic classifies each day's volume as buyer- or seller-initiated
    based on the sign of the daily return (close minus open) and computes the
    average absolute imbalance.

    Parameters
    ----------
    data:
        DataFrame containing ``open``, ``close``, and ``volume`` columns.
    """

    price_change = data["close"] - data["open"]
    buy_volume = data["volume"].where(price_change > 0, 0.0)
    sell_volume = data["volume"].where(price_change <= 0, 0.0)
    total = buy_volume + sell_volume
    imbalance = (buy_volume - sell_volume).abs()
    return (imbalance / total).mean()
