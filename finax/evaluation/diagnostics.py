"""Statistical tests for model residuals.

Wrappers over statsmodels and scipy with a uniform return shape:
``{"statistic": float, "pvalue": float, ...}``.

Note the differing null hypotheses -- ADF's null is a unit root (non-stationary)
while KPSS's null is stationarity, so they are complementary rather than
redundant, and a series both tests reject is usually fractionally integrated.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from ..errors import DataValidationError, require

__all__ = [
    "adf_test",
    "kpss_test",
    "ljung_box",
    "jarque_bera_test",
    "ks_test",
    "arch_lm_test",
    "residual_diagnostics",
]


def _as_array(x: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(x) if not isinstance(x, np.ndarray) else x, dtype=float)
    array = array[np.isfinite(array)]
    if array.size < 8:
        raise DataValidationError(
            f"Need at least 8 finite observations, got {array.size}."
        )
    return array


def adf_test(series: Iterable[float], **kwargs) -> dict[str, float]:
    """Augmented Dickey--Fuller test. Null: a unit root is present.

    A small p-value is evidence *for* stationarity.
    """
    stattools = require("statsmodels.tsa.stattools", purpose="the ADF test")
    stat, pvalue, used_lag, nobs, *_ = stattools.adfuller(_as_array(series), **kwargs)
    return {
        "statistic": float(stat),
        "pvalue": float(pvalue),
        "used_lag": int(used_lag),
        "nobs": int(nobs),
    }


def kpss_test(series: Iterable[float], **kwargs) -> dict[str, float]:
    """KPSS test. Null: the series is stationary.

    A small p-value is evidence *against* stationarity -- the opposite of
    :func:`adf_test`.
    """
    stattools = require("statsmodels.tsa.stattools", purpose="the KPSS test")
    kwargs.setdefault("nlags", "auto")
    import warnings

    with warnings.catch_warnings():
        # statsmodels warns when the p-value is clipped to the edge of its
        # lookup table; that is expected, not an error.
        warnings.simplefilter("ignore")
        stat, pvalue, lags, _ = stattools.kpss(_as_array(series), **kwargs)
    return {"statistic": float(stat), "pvalue": float(pvalue), "lags": int(lags)}


def ljung_box(residuals: Iterable[float], *, lags: int = 20) -> dict[str, float]:
    """Ljung--Box test for autocorrelation. Null: no autocorrelation up to ``lags``.

    Modern statsmodels always returns a DataFrame from ``acorr_ljungbox``; the
    ``return_df=False`` tuple form was removed. This reads the DataFrame.
    """
    diagnostic = require("statsmodels.stats.diagnostic", purpose="the Ljung-Box test")
    array = _as_array(residuals)
    lags = min(lags, max(1, array.size // 5))
    result = diagnostic.acorr_ljungbox(array, lags=[lags])
    return {
        "statistic": float(result["lb_stat"].iloc[0]),
        "pvalue": float(result["lb_pvalue"].iloc[0]),
        "lags": int(lags),
    }


def jarque_bera_test(residuals: Iterable[float]) -> dict[str, float]:
    """Jarque--Bera normality test, reporting skewness and kurtosis too."""
    stattools = require("statsmodels.stats.stattools", purpose="the Jarque-Bera test")
    stat, pvalue, skew, kurtosis = stattools.jarque_bera(_as_array(residuals))
    return {
        "statistic": float(stat),
        "pvalue": float(pvalue),
        "skew": float(skew),
        "kurtosis": float(kurtosis),
    }


def ks_test(residuals: Iterable[float], dist: str = "norm") -> dict[str, float]:
    """Kolmogorov--Smirnov goodness-of-fit test.

    Residuals are standardised before comparison, since ``scipy``'s ``"norm"``
    reference is the *standard* normal -- testing raw residuals against it would
    reject purely because of their scale.
    """
    scipy_stats = require("scipy.stats", purpose="the KS test")
    array = _as_array(residuals)
    if dist == "norm":
        array = (array - array.mean()) / max(array.std(ddof=1), 1e-12)
    stat, pvalue = scipy_stats.kstest(array, dist)
    return {"statistic": float(stat), "pvalue": float(pvalue)}


def arch_lm_test(residuals: Iterable[float], *, lags: int = 12) -> dict[str, float]:
    """Engle's ARCH LM test. Null: no conditional heteroskedasticity.

    Rejecting means volatility clusters, which is the standard justification for
    a GARCH or stochastic-volatility model.
    """
    diagnostic = require("statsmodels.stats.diagnostic", purpose="the ARCH LM test")
    array = _as_array(residuals)
    lags = min(lags, max(1, array.size // 5))
    stat, pvalue, f_stat, f_pvalue = diagnostic.het_arch(array, nlags=lags)
    return {
        "statistic": float(stat),
        "pvalue": float(pvalue),
        "f_statistic": float(f_stat),
        "f_pvalue": float(f_pvalue),
        "lags": int(lags),
    }


def residual_diagnostics(
    residuals: Iterable[float], *, lags: int = 20
) -> dict[str, dict[str, float]]:
    """Run the full battery of residual tests.

    Any test whose optional dependency is missing is reported as an ``error``
    entry rather than aborting the whole report.

    Returns
    -------
    Dict keyed by test name.
    """
    tests = {
        "adf": lambda r: adf_test(r),
        "kpss": lambda r: kpss_test(r),
        "ljung_box": lambda r: ljung_box(r, lags=lags),
        "jarque_bera": lambda r: jarque_bera_test(r),
        "ks": lambda r: ks_test(r),
        "arch_lm": lambda r: arch_lm_test(r),
    }
    report: dict[str, dict[str, float]] = {}
    for name, fn in tests.items():
        try:
            report[name] = fn(residuals)
        except Exception as exc:
            report[name] = {"error": str(exc)}
    return report
