"""Metrics and statistical diagnostics."""

from .diagnostics import (
    adf_test,
    arch_lm_test,
    jarque_bera_test,
    kpss_test,
    ks_test,
    ljung_box,
    residual_diagnostics,
)
from .metrics import (
    calmar_ratio,
    continuous_ranked_probability_score,
    hit_rate,
    mae,
    mape,
    max_drawdown,
    r_squared,
    rmse,
    sharpe_ratio,
    sortino_ratio,
)
from .time_series import fit_ar, fit_arima, fit_garch, fit_ma

__all__ = [
    # Metrics
    "rmse",
    "mae",
    "mape",
    "r_squared",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "calmar_ratio",
    "hit_rate",
    "continuous_ranked_probability_score",
    # Diagnostics
    "adf_test",
    "kpss_test",
    "ljung_box",
    "jarque_bera_test",
    "ks_test",
    "arch_lm_test",
    "residual_diagnostics",
    # Classical models
    "fit_ar",
    "fit_ma",
    "fit_arima",
    "fit_garch",
]
