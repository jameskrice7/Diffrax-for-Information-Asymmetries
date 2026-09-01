# Evaluation

## Forecast metrics

`rmse`, `mae`, `mape`, `r_squared`, and `hit_rate` — the fraction of predictions
with the correct sign. For financial forecasting `hit_rate` is often more
informative than RMSE: getting direction right is what a position depends on.

## Probabilistic forecasts

```python
from finax.evaluation import continuous_ranked_probability_score as crps

paths = model.sample(y0, 0., 1., key=key, n_paths=1000)
score = crps(paths[:, -1], observed)
```

CRPS is the natural scoring rule for the Monte Carlo ensembles that
`NeuralSDE.sample` produces. It rewards a forecast for being both accurate *and*
appropriately confident, and reduces to absolute error for a deterministic
forecast. Lower is better.

## Portfolio metrics

```python
from finax.evaluation import sharpe_ratio, sortino_ratio, max_drawdown, calmar_ratio

sharpe_ratio(returns, periods_per_year=252)
```

Pass `periods_per_year` — published Sharpe ratios are essentially always
annualised, and omitting it produces a number that is not comparable to them.

`sortino_ratio` penalises only downside deviation, which is the more honest
statistic for strategies with deliberately asymmetric payoffs.

## Residual diagnostics

```python
from finax.evaluation import residual_diagnostics

report = residual_diagnostics(residuals)
```

Runs ADF, KPSS, Ljung–Box, Jarque–Bera, KS and Engle's ARCH LM. Any test whose
optional dependency is missing is reported as an `error` entry rather than
aborting the whole report.

Note the opposing nulls: **ADF's null is a unit root** (non-stationary) while
**KPSS's null is stationarity**. They are complementary, not redundant, and a
series both tests reject is usually fractionally integrated.

Rejecting the ARCH LM test means volatility clusters, which is the standard
justification for reaching for a GARCH or stochastic-volatility model.

The KS test standardises residuals first, since scipy's `"norm"` reference is the
*standard* normal — testing raw residuals against it would reject purely on scale.

## Classical baselines

```python
from finax.evaluation import fit_arima, fit_garch

fit_arima(series, p=1, d=1, q=1)
fit_garch(returns * 100, p=1, q=1)   # arch expects percentage points
```

A neural SDE is worth using only if it beats ARIMA and GARCH. These wrappers make
that comparison cheap enough that it stops being skipped.
