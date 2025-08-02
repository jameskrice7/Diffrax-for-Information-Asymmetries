"""Statistical tests for validating models and time-series residuals."""

from __future__ import annotations

from typing import Iterable, Dict

import numpy as np

try:  # pragma: no cover - optional dependency
    from statsmodels.tsa.stattools import adfuller, kpss
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.stats.stattools import jarque_bera
except Exception:  # pragma: no cover
    adfuller = None  # type: ignore
    kpss = None  # type: ignore
    acorr_ljungbox = None  # type: ignore
    jarque_bera = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from scipy import stats
except Exception:  # pragma: no cover
    stats = None  # type: ignore


def _require_statsmodels() -> None:
    if any(fn is None for fn in (adfuller, kpss, acorr_ljungbox, jarque_bera)):
        raise ImportError("statsmodels is required for statistical tests")


def _require_scipy() -> None:
    if stats is None:
        raise ImportError("scipy is required for the KS test")


def adf_test(series: Iterable[float]) -> Dict[str, float]:
    """Augmented Dickey–Fuller test for stationarity."""
    _require_statsmodels()
    stat, pvalue, *_ = adfuller(np.asarray(series))
    return {"statistic": float(stat), "pvalue": float(pvalue)}


def kpss_test(series: Iterable[float]) -> Dict[str, float]:
    """KPSS stationarity test."""
    _require_statsmodels()
    stat, pvalue, *_ = kpss(np.asarray(series), nlags="auto")
    return {"statistic": float(stat), "pvalue": float(pvalue)}


def ljung_box(residuals: Iterable[float], lags: int = 20) -> Dict[str, float]:
    """Ljung–Box test for autocorrelation."""
    _require_statsmodels()
    stat, pvalue = acorr_ljungbox(np.asarray(residuals), lags=[lags], return_df=False)
    return {"statistic": float(stat[0]), "pvalue": float(pvalue[0])}


def jarque_bera_test(residuals: Iterable[float]) -> Dict[str, float]:
    """Jarque–Bera normality test."""
    _require_statsmodels()
    stat, pvalue, _, _ = jarque_bera(np.asarray(residuals))
    return {"statistic": float(stat), "pvalue": float(pvalue)}


def ks_test(residuals: Iterable[float], dist: str = "norm") -> Dict[str, float]:
    """Kolmogorov–Smirnov test against a reference distribution."""
    _require_scipy()
    stat, pvalue = stats.kstest(np.asarray(residuals), dist)
    return {"statistic": float(stat), "pvalue": float(pvalue)}
