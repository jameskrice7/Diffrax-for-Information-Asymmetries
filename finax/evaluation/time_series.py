"""Classical time-series models for post-simulation analysis."""

from __future__ import annotations

import pandas as pd

try:  # pragma: no cover - optional dependency
    from statsmodels.tsa.ar_model import AutoReg
    from statsmodels.tsa.arima.model import ARIMA
except Exception:  # pragma: no cover
    AutoReg = None  # type: ignore
    ARIMA = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from arch import arch_model
except Exception:  # pragma: no cover
    arch_model = None  # type: ignore


def _check_statsmodels() -> None:
    if AutoReg is None or ARIMA is None:
        raise ImportError("statsmodels is required for time-series models")


def _check_arch() -> None:
    if arch_model is None:
        raise ImportError("arch is required for GARCH models")


def fit_ar(series: pd.Series, lags: int):
    """Fit an autoregressive (AR) model."""

    _check_statsmodels()
    model = AutoReg(series, lags=lags, old_names=False)
    return model.fit()


def fit_ma(series: pd.Series, q: int):
    """Fit a moving-average (MA) model."""

    _check_statsmodels()
    model = ARIMA(series, order=(0, 0, q))
    return model.fit()


def fit_arma(series: pd.Series, p: int, q: int):
    """Fit an ARMA model."""

    _check_statsmodels()
    model = ARIMA(series, order=(p, 0, q))
    return model.fit()


def fit_arima(series: pd.Series, p: int, d: int, q: int):
    """Fit an ARIMA model."""

    _check_statsmodels()
    model = ARIMA(series, order=(p, d, q))
    return model.fit()


def fit_garch(series: pd.Series, p: int = 1, q: int = 1):
    """Fit a GARCH model using the `arch` package."""

    _check_arch()
    model = arch_model(series, vol="GARCH", p=p, q=q)
    return model.fit(disp="off")
