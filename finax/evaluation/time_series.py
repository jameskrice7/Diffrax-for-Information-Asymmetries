"""Classical time-series baselines.

A neural SDE is worth using only if it beats ARIMA and GARCH. These wrappers
make that comparison cheap to run, so it stops being skipped.
"""

from __future__ import annotations

from typing import Any

from ..errors import require

__all__ = ["fit_ar", "fit_ma", "fit_arima", "fit_garch"]


def fit_ar(series: Any, lags: int, **kwargs: Any):
    """Fit an autoregressive model of order ``lags``."""
    ar_model = require("statsmodels.tsa.ar_model", purpose="AR models")
    return ar_model.AutoReg(series, lags=lags, old_names=False, **kwargs).fit()


def fit_ma(series: Any, q: int, **kwargs: Any):
    """Fit a moving-average model of order ``q``."""
    return fit_arima(series, p=0, d=0, q=q, **kwargs)


def fit_arima(series: Any, *, p: int, d: int, q: int, **kwargs: Any):
    """Fit an ARIMA(p, d, q) model.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> s = pd.Series(rng.normal(size=200).cumsum())
    >>> res = fit_arima(s, p=1, d=1, q=0)
    >>> hasattr(res, "aic")
    True
    """
    arima = require("statsmodels.tsa.arima.model", purpose="ARIMA models")
    return arima.ARIMA(series, order=(p, d, q), **kwargs).fit()


def fit_garch(
    series: Any, *, p: int = 1, q: int = 1, mean: str = "Constant", **kwargs: Any
):
    """Fit a GARCH(p, q) model using the ``arch`` package.

    Note that ``arch`` expects returns scaled to percentage points; feeding it
    raw decimal returns produces a poorly-scaled optimisation and a loud warning.
    """
    arch = require("arch", purpose="GARCH models")
    model = arch.arch_model(series, vol="GARCH", p=p, q=q, mean=mean, **kwargs)
    return model.fit(disp="off")
