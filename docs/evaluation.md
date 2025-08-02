# Evaluation

Finax supplies metrics and classical time-series models for analyzing model outputs.

## Metrics
- `finax.evaluation.metrics.rmse` computes root-mean-square error.
- `finax.evaluation.metrics.sharpe_ratio` measures risk-adjusted return.

## Statistical Tests
Finax offers common diagnostics used in modern time-series research:
- `finax.evaluation.tests.adf_test` for Augmented Dickey–Fuller stationarity checks.
- `finax.evaluation.tests.kpss_test` for KPSS stationarity testing.
- `finax.evaluation.tests.ljung_box` for autocorrelation.
- `finax.evaluation.tests.jarque_bera_test` for residual normality.
- `finax.evaluation.tests.ks_test` for Kolmogorov–Smirnov goodness-of-fit.
- `finax.evaluation.time_series.residual_diagnostics` runs all tests on a residual series.

## Time-Series Models
- `finax.evaluation.time_series.fit_ar` fits an autoregressive model.
- `finax.evaluation.time_series.fit_ma` fits a moving-average model.
- `finax.evaluation.time_series.fit_arma` fits an ARMA model.
- `finax.evaluation.time_series.fit_arima` fits an ARIMA model.
- `finax.evaluation.time_series.fit_garch` fits a GARCH volatility model.
