"""Verifying that a numerical SDE solve is actually correct.

Getting a plausible-looking path out of an SDE solver tells you almost nothing.
The step size might be too coarse, the solver's assumptions might be violated
(applying a Stratonovich solver to Ito coefficients, or an additive-noise solver
to multiplicative noise), and in every one of those cases you still get smooth,
believable-looking output -- just wrong.

The tools here turn that into something measurable.

:func:`strong_order`
    Does the solver converge pathwise at its advertised rate? Compares against a
    fine reference solve **driven by the same Brownian path**, which is what
    makes it a pathwise rather than distributional comparison.
:func:`weak_order`
    Does the solver get *expectations* right? This is what matters for pricing
    and for moment matching, and it converges faster than strong order.
:func:`martingale_test`
    Is a quantity that theory says should be a martingale actually one in the
    simulation? A failure here usually means a missing Ito correction.
:func:`moment_report`
    Compares simulated moments against analytic targets.

None of this exists as a packaged tool elsewhere in the JAX ecosystem, and it is
the difference between "the code ran" and "the numbers are right".
"""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ._typing import Array, Float, PRNGKeyArray
from .errors import DataValidationError

__all__ = [
    "ConvergenceReport",
    "strong_order",
    "weak_order",
    "martingale_test",
    "moment_report",
]


class ConvergenceReport(eqx.Module):
    """Result of a convergence study.

    Attributes
    ----------
    step_sizes:
        The step sizes tested, descending.
    errors:
        Measured error at each step size.
    estimated_order:
        Slope of ``log(error)`` against ``log(dt)``, fitted by least squares.
        This is the empirical convergence order.
    r_squared:
        Fit quality. A low value means the errors do not lie on a clean power
        law, so ``estimated_order`` should not be trusted -- typically because
        the step sizes are too coarse to be in the asymptotic regime, or because
        Monte Carlo noise dominates the discretisation error.
    """

    step_sizes: Float[Array, " level"]
    errors: Float[Array, " level"]
    estimated_order: float
    r_squared: float

    def __repr__(self) -> str:
        rows = "\n".join(
            f"    dt={float(d):<12.6g} error={float(e):.6g}"
            for d, e in zip(self.step_sizes, self.errors, strict=True)
        )
        return (
            f"ConvergenceReport(order={self.estimated_order:.3f}, "
            f"r_squared={self.r_squared:.4f})\n{rows}"
        )


def _fit_log_slope(
    dts: Float[Array, " level"], errors: Float[Array, " level"]
) -> tuple[float, float]:
    """Least-squares slope of log(error) on log(dt), plus R^2."""
    x = np.log(np.asarray(dts, dtype=np.float64))
    y = np.log(np.maximum(np.asarray(errors, dtype=np.float64), 1e-300))
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    ss_res = float(np.sum((y - predicted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return float(slope), r_squared


def strong_order(
    simulate: Callable[[float, PRNGKeyArray], Array],
    *,
    step_sizes: tuple[float, ...] = (0.1, 0.05, 0.025, 0.0125),
    reference_step: float | None = None,
    n_paths: int = 256,
    key: PRNGKeyArray,
) -> ConvergenceReport:
    """Estimate the strong (pathwise) convergence order of a solver.

    Strong order ``p`` means ``E|Y_dt(T) - Y(T)| = O(dt^p)``. Euler--Maruyama has
    ``p = 0.5`` for general noise and ``p = 1.0`` for additive noise; Milstein
    has ``p = 1.0``.

    Parameters
    ----------
    simulate:
        ``simulate(dt, key) -> terminal_value``. It **must** use ``key`` to drive
        the Brownian path in a way that is consistent across step sizes -- a
        ``diffrax.VirtualBrownianTree`` constructed from ``key`` does exactly
        this, which is why it is the right tool here. Without that consistency
        you are comparing different random paths and measuring nothing.
    step_sizes:
        Step sizes to test.
    reference_step:
        Step size for the "truth". Defaults to a quarter of the smallest tested
        step. Ignored if ``simulate`` is exact.
    n_paths:
        Number of paths averaged over.
    key:
        PRNG key.

    Returns
    -------
    A :class:`ConvergenceReport`.

    Notes
    -----
    **Always read** :attr:`ConvergenceReport.r_squared`. The reference solve is
    itself numerical, so it sets an error floor. For a high-order solver
    (``ShARK``, ``SRA1``) the tested step sizes may already be at that floor, in
    which case the measured "error" is Brownian-tree tolerance and float32 round-
    off rather than discretisation error, and the fitted order is meaningless.
    A low ``r_squared`` is the signature. When it happens, either use much
    coarser ``step_sizes``, tighten the ``tol`` of the ``VirtualBrownianTree``,
    enable float64, or compare against an exact sampler from
    :mod:`finax.processes` instead of a fine numerical solve.

    Examples
    --------
    Euler--Maruyama on geometric Brownian motion, which has multiplicative
    noise, should show strong order near 0.5:

    >>> import diffrax, jax.numpy as jnp, jax.random as jr
    >>> def simulate(dt, key):
    ...     bm = diffrax.VirtualBrownianTree(0.0, 1.0, tol=1e-4, shape=(),
    ...                                      key=key)
    ...     terms = diffrax.MultiTerm(
    ...         diffrax.ODETerm(lambda t, y, a: 0.05 * y),
    ...         diffrax.ControlTerm(lambda t, y, a: 0.3 * y, bm))
    ...     sol = diffrax.diffeqsolve(terms, diffrax.Euler(), 0.0, 1.0, dt,
    ...                               jnp.array(1.0), max_steps=None)
    ...     return sol.ys[-1]
    >>> report = strong_order(simulate, key=jr.PRNGKey(0), n_paths=128)
    >>> bool(0.3 < report.estimated_order < 0.75)
    True
    """
    step_sizes = tuple(sorted(step_sizes, reverse=True))
    if len(step_sizes) < 2:
        raise DataValidationError("Need at least 2 step sizes to fit a slope.")
    if reference_step is None:
        reference_step = min(step_sizes) / 4.0

    keys = jax.random.split(key, n_paths)
    batched = jax.jit(jax.vmap(simulate, in_axes=(None, 0)), static_argnums=())

    reference = batched(reference_step, keys)

    errors = []
    for dt in step_sizes:
        approx = batched(dt, keys)
        # Pathwise absolute error, averaged over paths: the strong error.
        errors.append(jnp.mean(jnp.abs(approx - reference)))
    errors_arr = jnp.stack(errors)

    order, r_squared = _fit_log_slope(jnp.asarray(step_sizes), errors_arr)
    return ConvergenceReport(
        step_sizes=jnp.asarray(step_sizes),
        errors=errors_arr,
        estimated_order=order,
        r_squared=r_squared,
    )


def weak_order(
    simulate: Callable[[float, PRNGKeyArray], Array],
    *,
    functional: Callable[[Array], Array] = lambda y: y,
    exact_expectation: float,
    step_sizes: tuple[float, ...] = (0.1, 0.05, 0.025, 0.0125),
    n_paths: int = 8192,
    key: PRNGKeyArray,
) -> ConvergenceReport:
    """Estimate the weak convergence order against a known expectation.

    Weak order ``q`` means ``|E[f(Y_dt(T))] - E[f(Y(T))]| = O(dt^q)``. Both Euler
    and Milstein have ``q = 1``. Weak order is the relevant notion whenever you
    care about an average -- a derivative price, a moment, a risk measure --
    rather than an individual path.

    Parameters
    ----------
    simulate:
        ``simulate(dt, key) -> terminal_value``.
    functional:
        The ``f`` whose expectation is compared. Defaults to the identity.
    exact_expectation:
        The analytic value of ``E[f(Y(T))]``. This is why the process library in
        :mod:`finax.processes` documents closed-form moments.
    n_paths:
        Monte Carlo sample size. Must be large: the Monte Carlo standard error
        scales as ``1/sqrt(n_paths)`` and will swamp the discretisation bias if
        it is too small, which shows up as a poor ``r_squared``.

    Returns
    -------
    A :class:`ConvergenceReport`.

    Examples
    --------
    >>> import diffrax, jax.numpy as jnp, jax.random as jr
    >>> mu, sigma = 0.05, 0.3
    >>> def simulate(dt, key):
    ...     bm = diffrax.VirtualBrownianTree(0.0, 1.0, tol=1e-4, shape=(),
    ...                                      key=key)
    ...     terms = diffrax.MultiTerm(
    ...         diffrax.ODETerm(lambda t, y, a: mu * y),
    ...         diffrax.ControlTerm(lambda t, y, a: sigma * y, bm))
    ...     sol = diffrax.diffeqsolve(terms, diffrax.Euler(), 0.0, 1.0, dt,
    ...                               jnp.array(1.0), max_steps=None)
    ...     return sol.ys[-1]
    >>> report = weak_order(simulate, exact_expectation=float(jnp.exp(mu)),
    ...                     key=jr.PRNGKey(0), n_paths=4096)
    >>> report.errors.shape
    (4,)
    """
    step_sizes = tuple(sorted(step_sizes, reverse=True))
    if len(step_sizes) < 2:
        raise DataValidationError("Need at least 2 step sizes to fit a slope.")

    keys = jax.random.split(key, n_paths)
    batched = jax.jit(jax.vmap(simulate, in_axes=(None, 0)))

    errors = []
    for dt in step_sizes:
        values = jax.vmap(functional)(batched(dt, keys))
        errors.append(jnp.abs(jnp.mean(values) - exact_expectation))
    errors_arr = jnp.stack(errors)

    order, r_squared = _fit_log_slope(jnp.asarray(step_sizes), errors_arr)
    return ConvergenceReport(
        step_sizes=jnp.asarray(step_sizes),
        errors=errors_arr,
        estimated_order=order,
        r_squared=r_squared,
    )


def martingale_test(
    paths: Float[Array, "path time"],
    *,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Test whether simulated paths have the martingale property ``E[X_t] = X_0``.

    Many quantities are martingales by construction: a discounted price under
    the risk-neutral measure, a compensated jump process, ``exp(sigma W_t -
    sigma^2 t / 2)``. If the simulation breaks that, something is wrong --
    typically a missing Ito correction or a solver applied under the wrong
    stochastic calculus.

    Returns a per-time-point z-statistic for ``E[X_t] - X_0`` and reports the
    worst violation.

    Parameters
    ----------
    paths:
        ``(path, time)`` array of simulated trajectories.
    alpha:
        Significance level for the two-sided test.

    Returns
    -------
    Dict with ``max_abs_z``, ``critical_value``, ``passed``, and
    ``worst_time_index``.

    Examples
    --------
    Standard Brownian motion is a martingale:

    >>> import jax.numpy as jnp, jax.random as jr
    >>> steps = jr.normal(jr.PRNGKey(0), (20000, 50)) * jnp.sqrt(1 / 50)
    >>> bm = jnp.concatenate([jnp.zeros((20000, 1)), jnp.cumsum(steps, axis=1)], 1)
    >>> martingale_test(bm)["passed"]
    True

    Adding a drift breaks it, and the test detects that:

    >>> drifted = bm + 0.5 * jnp.linspace(0, 1, 51)
    >>> martingale_test(drifted)["passed"]
    False
    """
    paths = jnp.asarray(paths)
    if paths.ndim != 2:
        raise DataValidationError(f"paths must be (path, time), got shape {paths.shape}.")

    n_paths = paths.shape[0]
    x0 = paths[:, 0]
    deviations = paths - x0[:, None]

    means = jnp.mean(deviations, axis=0)
    stds = jnp.std(deviations, axis=0, ddof=1)
    standard_errors = jnp.maximum(stds / jnp.sqrt(n_paths), 1e-12)
    z = means / standard_errors
    # t=0 is identically zero by construction, so exclude it.
    z = z[1:]

    # Two-sided test with a Bonferroni correction across time points, since we
    # are taking a maximum over many correlated tests.
    n_tests = z.shape[0]
    from scipy.stats import norm  # local import: scipy is an optional extra

    critical = float(norm.ppf(1 - alpha / (2 * n_tests)))
    max_abs_z = float(jnp.max(jnp.abs(z)))
    return {
        "max_abs_z": max_abs_z,
        "critical_value": critical,
        "passed": bool(max_abs_z <= critical),
        "worst_time_index": int(jnp.argmax(jnp.abs(z))) + 1,
    }


def moment_report(
    samples: Float[Array, " path"],
    *,
    expected_mean: float | None = None,
    expected_variance: float | None = None,
    expected_skewness: float | None = None,
    expected_kurtosis: float | None = None,
) -> dict[str, dict[str, float]]:
    """Compare sample moments against analytic targets, with Monte Carlo error bars.

    Each moment is reported alongside its standard error, so a discrepancy can be
    judged against sampling noise rather than eyeballed. A ``z`` above roughly 3
    indicates a genuine mismatch rather than an unlucky draw.

    Returns
    -------
    Nested dict keyed by moment name, each with ``sample``, ``expected``,
    ``std_error`` and ``z``.

    Examples
    --------
    >>> import jax.numpy as jnp, jax.random as jr
    >>> x = jr.normal(jr.PRNGKey(0), (100000,))
    >>> rep = moment_report(x, expected_mean=0.0, expected_variance=1.0)
    >>> bool(abs(rep["mean"]["z"]) < 3.0 and abs(rep["variance"]["z"]) < 3.0)
    True
    """
    x = jnp.asarray(samples)
    if x.ndim != 1:
        raise DataValidationError(f"samples must be 1-D, got shape {x.shape}.")
    n = x.shape[0]

    mean = float(jnp.mean(x))
    var = float(jnp.var(x, ddof=1))
    std = float(jnp.sqrt(var))
    centred = (x - mean) / jnp.maximum(std, 1e-12)
    skew = float(jnp.mean(centred**3))
    kurt = float(jnp.mean(centred**4))

    # Standard errors under approximate normality of the sampling distribution.
    errors = {
        "mean": std / np.sqrt(n),
        "variance": var * np.sqrt(2.0 / (n - 1)),
        "skewness": np.sqrt(6.0 / n),
        "kurtosis": np.sqrt(24.0 / n),
    }
    values = {"mean": mean, "variance": var, "skewness": skew, "kurtosis": kurt}
    targets = {
        "mean": expected_mean,
        "variance": expected_variance,
        "skewness": expected_skewness,
        "kurtosis": expected_kurtosis,
    }

    report: dict[str, dict[str, float]] = {}
    for name, value in values.items():
        target = targets[name]
        entry = {"sample": value, "std_error": float(errors[name])}
        if target is not None:
            entry["expected"] = float(target)
            entry["z"] = (value - float(target)) / max(float(errors[name]), 1e-12)
        report[name] = entry
    return report
