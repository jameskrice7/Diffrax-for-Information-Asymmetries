"""Calibrating classical processes to observed data by maximum likelihood.

Works with any process in :mod:`finax.processes` that exposes an exact
``log_likelihood``. Parameters are optimised in an unconstrained space and
mapped back through a bijector, so positivity and interval constraints hold at
every iterate without projection.
"""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from .._typing import Array, Float, PyTree

__all__ = ["CalibrationResult", "fit_mle", "fit_gbm", "fit_ou"]


class CalibrationResult(eqx.Module):
    """Outcome of a maximum-likelihood calibration.

    Attributes
    ----------
    process:
        The fitted process, ready to ``sample`` from.
    log_likelihood:
        Maximised log-likelihood.
    history:
        Log-likelihood at each iteration, for checking convergence.
    """

    process: PyTree
    log_likelihood: Float[Array, ""]
    history: Float[Array, " step"]


def fit_mle(
    build: Callable[[Float[Array, " param"]], PyTree],
    initial_raw: Float[Array, " param"],
    log_likelihood: Callable[[PyTree], Float[Array, ""]],
    *,
    steps: int = 1000,
    learning_rate: float = 0.05,
) -> CalibrationResult:
    """Maximise a log-likelihood over an unconstrained parameter vector.

    Parameters
    ----------
    build:
        Maps an unconstrained vector to a process object, applying whatever
        transforms the constraints require.
    initial_raw:
        Starting point in unconstrained space.
    log_likelihood:
        Maps a process to its scalar log-likelihood.
    steps, learning_rate:
        Adam settings.

    Examples
    --------
    >>> import jax, jax.numpy as jnp, jax.random as jr
    >>> from finax.processes import GeometricBrownianMotion
    >>> ts = jnp.linspace(0, 4, 2001)
    >>> path = GeometricBrownianMotion(mu=0.08, sigma=0.25).sample(
    ...     jnp.array(100.0), ts=ts, key=jr.PRNGKey(0), n_paths=1)[0]
    >>> build = lambda r: GeometricBrownianMotion(mu=r[0],
    ...                                           sigma=jax.nn.softplus(r[1]))
    >>> res = fit_mle(build, jnp.array([0.0, 0.0]),
    ...               lambda p: p.log_likelihood(path, ts))
    >>> bool(abs(float(res.process.sigma) - 0.25) < 0.02)
    True
    """
    optimiser = optax.adam(learning_rate)

    @eqx.filter_jit
    def run(raw0):
        opt_state = optimiser.init(raw0)

        def body(carry, _):
            raw, state = carry
            value, grads = jax.value_and_grad(lambda r: -log_likelihood(build(r)))(raw)
            grads = jnp.where(jnp.isfinite(grads), grads, 0.0)
            updates, state = optimiser.update(grads, state, raw)
            return (optax.apply_updates(raw, updates), state), -value

        (raw, _), history = jax.lax.scan(body, (raw0, opt_state), None, length=steps)
        return raw, history

    raw, history = run(jnp.asarray(initial_raw, jnp.float32))
    process = build(raw)
    return CalibrationResult(
        process=process,
        log_likelihood=log_likelihood(process),
        history=history,
    )


def fit_gbm(
    path: Float[Array, " time"],
    ts: Float[Array, " time"],
) -> CalibrationResult:
    """Calibrate a :class:`~finax.processes.GeometricBrownianMotion` in closed form.

    GBM has an analytic MLE -- log-returns are iid Gaussian, so the estimates are
    just their sample mean and variance rescaled by the time step. No iteration
    is needed, and iterating would only add error.

    Examples
    --------
    >>> import jax.numpy as jnp, jax.random as jr
    >>> from finax.processes import GeometricBrownianMotion
    >>> ts = jnp.linspace(0, 10, 5001)
    >>> path = GeometricBrownianMotion(mu=0.07, sigma=0.3).sample(
    ...     jnp.array(50.0), ts=ts, key=jr.PRNGKey(0), n_paths=1)[0]
    >>> res = fit_gbm(path, ts)
    >>> bool(abs(float(res.process.sigma) - 0.3) < 0.01)
    True
    """
    from ..processes import GeometricBrownianMotion

    path = jnp.asarray(path)
    ts = jnp.asarray(ts)
    dt = jnp.mean(jnp.diff(ts))
    log_returns = jnp.diff(jnp.log(path))

    sigma = jnp.sqrt(jnp.maximum(jnp.var(log_returns, ddof=1) / dt, 1e-12))
    mu = jnp.mean(log_returns) / dt + 0.5 * sigma**2

    process = GeometricBrownianMotion(mu=mu, sigma=sigma)
    return CalibrationResult(
        process=process,
        log_likelihood=process.log_likelihood(path, ts),
        history=jnp.asarray([]),
    )


def fit_ou(
    path: Float[Array, " time"],
    ts: Float[Array, " time"],
) -> CalibrationResult:
    """Calibrate an :class:`~finax.processes.OrnsteinUhlenbeck` in closed form.

    The exact transition law is Gaussian AR(1), so the MLE follows from an OLS
    regression of ``X_{t+1}`` on ``X_t`` -- again no iteration required.

    Examples
    --------
    >>> import jax.numpy as jnp, jax.random as jr
    >>> from finax.processes import OrnsteinUhlenbeck
    >>> ts = jnp.linspace(0, 200, 20001)
    >>> path = OrnsteinUhlenbeck(kappa=1.5, theta=0.03, sigma=0.2).sample(
    ...     jnp.array(0.03), ts=ts, key=jr.PRNGKey(0), n_paths=1)[0]
    >>> res = fit_ou(path, ts)
    >>> bool(abs(float(res.process.kappa) - 1.5) < 0.2)
    True
    >>> bool(abs(float(res.process.theta) - 0.03) < 0.02)
    True
    """
    from ..processes import OrnsteinUhlenbeck

    path = jnp.asarray(path)
    ts = jnp.asarray(ts)
    dt = jnp.mean(jnp.diff(ts))

    x, y = path[:-1], path[1:]
    x_mean, y_mean = jnp.mean(x), jnp.mean(y)
    cov = jnp.mean((x - x_mean) * (y - y_mean))
    var = jnp.maximum(jnp.mean((x - x_mean) ** 2), 1e-12)

    # y = a + b x + e, with b = exp(-kappa dt).
    b = jnp.clip(cov / var, 1e-6, 1 - 1e-6)
    a = y_mean - b * x_mean

    kappa = -jnp.log(b) / dt
    theta = a / (1.0 - b)
    residual_var = jnp.mean((y - a - b * x) ** 2)
    sigma = jnp.sqrt(jnp.maximum(residual_var * 2.0 * kappa / (1.0 - b**2), 1e-12))

    process = OrnsteinUhlenbeck(kappa=kappa, theta=theta, sigma=sigma)
    return CalibrationResult(
        process=process,
        log_likelihood=process.log_likelihood(path, ts),
        history=jnp.asarray([]),
    )
