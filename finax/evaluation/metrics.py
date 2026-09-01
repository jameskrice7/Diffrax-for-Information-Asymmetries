"""Forecast and portfolio evaluation metrics, in pure JAX."""

from __future__ import annotations

import jax.numpy as jnp

from .._typing import Array, Float
from ..errors import DataValidationError, ShapeError

__all__ = [
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
]


def _check(a: Array, b: Array) -> tuple[Array, Array]:
    a, b = jnp.asarray(a), jnp.asarray(b)
    if a.shape != b.shape:
        raise ShapeError(f"Shapes must match, got {a.shape} and {b.shape}.")
    return a, b


def rmse(y_true: Float[Array, "..."], y_pred: Float[Array, "..."]) -> Float[Array, ""]:
    """Root mean squared error.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> float(rmse(jnp.array([1.0, 2.0]), jnp.array([1.0, 4.0])))
    1.4142...
    """
    y_true, y_pred = _check(y_true, y_pred)
    return jnp.sqrt(jnp.mean((y_true - y_pred) ** 2))


def mae(y_true: Float[Array, "..."], y_pred: Float[Array, "..."]) -> Float[Array, ""]:
    """Mean absolute error."""
    y_true, y_pred = _check(y_true, y_pred)
    return jnp.mean(jnp.abs(y_true - y_pred))


def mape(y_true: Float[Array, "..."], y_pred: Float[Array, "..."]) -> Float[Array, ""]:
    """Mean absolute percentage error, as a proportion rather than a percent."""
    y_true, y_pred = _check(y_true, y_pred)
    return jnp.mean(jnp.abs((y_true - y_pred) / jnp.maximum(jnp.abs(y_true), 1e-12)))


def r_squared(
    y_true: Float[Array, "..."], y_pred: Float[Array, "..."]
) -> Float[Array, ""]:
    """Coefficient of determination.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> x = jnp.array([1.0, 2.0, 3.0])
    >>> float(r_squared(x, x))
    1.0
    """
    y_true, y_pred = _check(y_true, y_pred)
    ss_res = jnp.sum((y_true - y_pred) ** 2)
    ss_tot = jnp.sum((y_true - jnp.mean(y_true)) ** 2)
    return 1.0 - ss_res / jnp.maximum(ss_tot, 1e-12)


def sharpe_ratio(
    returns: Float[Array, " time"],
    *,
    risk_free: float = 0.0,
    periods_per_year: int | None = None,
) -> Float[Array, ""]:
    """Sharpe ratio of a return series.

    Parameters
    ----------
    risk_free:
        Per-period risk-free rate, in the same units as ``returns``.
    periods_per_year:
        If given, annualise by ``sqrt(periods_per_year)``. Published Sharpe
        ratios are almost always annualised, so leaving this unset produces a
        number that is not comparable with them.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> r = jnp.full((252,), 0.001)  # constant returns: infinite Sharpe
    >>> bool(jnp.isinf(sharpe_ratio(r)) | (sharpe_ratio(r) > 1e6))
    True
    """
    returns = jnp.asarray(returns)
    excess = returns - risk_free
    ratio = jnp.mean(excess) / jnp.maximum(jnp.std(excess, ddof=1), 1e-12)
    if periods_per_year is not None:
        ratio = ratio * jnp.sqrt(periods_per_year)
    return ratio


def sortino_ratio(
    returns: Float[Array, " time"],
    *,
    target: float = 0.0,
    periods_per_year: int | None = None,
) -> Float[Array, ""]:
    """Sortino ratio: excess return per unit of *downside* deviation.

    Unlike Sharpe, upside volatility is not penalised. For strategies with
    deliberately asymmetric payoffs this is the more honest statistic.

    Examples
    --------
    >>> import jax.numpy as jnp, jax.random as jr
    >>> r = jr.normal(jr.PRNGKey(0), (1000,)) * 0.01 + 0.001
    >>> bool(sortino_ratio(r) > sharpe_ratio(r))  # upside is not penalised
    True
    """
    returns = jnp.asarray(returns)
    excess = returns - target
    downside = jnp.sqrt(jnp.mean(jnp.minimum(excess, 0.0) ** 2))
    ratio = jnp.mean(excess) / jnp.maximum(downside, 1e-12)
    if periods_per_year is not None:
        ratio = ratio * jnp.sqrt(periods_per_year)
    return ratio


def max_drawdown(returns: Float[Array, " time"]) -> Float[Array, ""]:
    """Largest peak-to-trough decline of the cumulative return, as a positive proportion.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> r = jnp.array([0.1, -0.5, 0.1])  # a 50% loss from the peak
    >>> bool(abs(float(max_drawdown(r)) - 0.5) < 1e-5)
    True
    """
    returns = jnp.asarray(returns)
    wealth = jnp.cumprod(1.0 + returns)
    running_peak = jnp.maximum.accumulate(wealth)
    return jnp.max(1.0 - wealth / running_peak)


def calmar_ratio(
    returns: Float[Array, " time"], *, periods_per_year: int = 252
) -> Float[Array, ""]:
    """Annualised return divided by maximum drawdown."""
    returns = jnp.asarray(returns)
    annual = jnp.mean(returns) * periods_per_year
    return annual / jnp.maximum(max_drawdown(returns), 1e-12)


def hit_rate(
    y_true: Float[Array, "..."], y_pred: Float[Array, "..."]
) -> Float[Array, ""]:
    """Fraction of predictions with the correct sign.

    For financial forecasting this is often more informative than RMSE: getting
    direction right is what a position depends on.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> float(hit_rate(jnp.array([1.0, -1.0, 1.0]), jnp.array([2.0, -3.0, -1.0])))
    0.6666...
    """
    y_true, y_pred = _check(y_true, y_pred)
    return jnp.mean((jnp.sign(y_true) == jnp.sign(y_pred)).astype(jnp.float32))


def continuous_ranked_probability_score(
    samples: Float[Array, "sample ..."], observation: Float[Array, "..."]
) -> Float[Array, ""]:
    """CRPS of an ensemble forecast, by the energy-score identity.

    .. math:: \\mathrm{CRPS} = \\mathbb{E}|X - y|
              - \\tfrac{1}{2}\\mathbb{E}|X - X'|

    The natural scoring rule for the Monte Carlo ensembles that
    :meth:`~finax.models.NeuralSDE.sample` produces: it rewards a forecast for
    being both accurate and appropriately confident, and it reduces to absolute
    error for a deterministic forecast. Lower is better.

    Parameters
    ----------
    samples:
        Ensemble members along the leading axis.
    observation:
        The realised value.

    Examples
    --------
    A point forecast's CRPS is just its absolute error:

    >>> import jax.numpy as jnp
    >>> s = jnp.full((100,), 2.0)
    >>> float(continuous_ranked_probability_score(s, jnp.array(3.0)))
    1.0

    A well-calibrated ensemble beats an overconfident wrong one:

    >>> import jax.random as jr
    >>> good = jr.normal(jr.PRNGKey(0), (2000,)) + 3.0
    >>> bad = jnp.full((2000,), 5.0)
    >>> y = jnp.array(3.0)
    >>> bool(continuous_ranked_probability_score(good, y)
    ...      < continuous_ranked_probability_score(bad, y))
    True
    """
    samples = jnp.asarray(samples)
    observation = jnp.asarray(observation)
    if samples.ndim < 1:
        raise DataValidationError("samples must have a leading ensemble axis.")

    n = samples.shape[0]
    accuracy = jnp.mean(jnp.abs(samples - observation))
    # Mean over all n^2 ordered pairs. The n diagonal terms are identically
    # zero, so rescaling by n/(n-1) recovers the unbiased estimate of E|X - X'|
    # over distinct pairs; without it CRPS is biased downwards for small n.
    pairwise = jnp.mean(jnp.abs(samples[:, None] - samples[None, :]))
    spread = pairwise * n / max(n - 1, 1)
    return accuracy - 0.5 * spread
