"""Probability of Informed Trading (PIN), estimated by differentiable MLE.

The PIN model of Easley, Kiefer, O'Hara & Paperman (1996) treats each trading
day as a draw from a mixture:

* with probability ``1 - alpha`` nothing happens, and buys and sells arrive as
  independent Poisson processes with rates ``eps_b`` and ``eps_s``;
* with probability ``alpha * delta`` there is **bad** news, and informed traders
  add sell volume at rate ``mu``;
* with probability ``alpha * (1 - delta)`` there is **good** news, and informed
  traders add buy volume at rate ``mu``.

The headline statistic is the share of order flow that is informed:

.. math:: \\mathrm{PIN} = \\frac{\\alpha \\mu}{\\alpha \\mu + \\epsilon_b + \\epsilon_s}.

Numerical stability
-------------------
The naive likelihood contains ``eps_b ** B`` and ``exp(-eps_b)`` separately. For
an actively traded stock ``B`` runs to five or six figures, both factors
overflow or underflow to ``inf``/``0``, and the estimate silently collapses.
This is the single best-documented failure mode in the PIN literature.

This module uses the **Lin & Ke (2011) factorization**, which pulls out the
common factor ``exp(-eps_b - eps_s) (eps_b + mu)^B (eps_s + mu)^S / (B! S!)``
and evaluates the remaining three-component mixture through ``logsumexp``.
Nothing is ever exponentiated before it is safe to do so. Lin & Ke's form is
the one shown by Ersan & Alici (2016) to be accurate where the earlier EHO
factorization is downward-biased.

What is new here
----------------
Every function is pure JAX, so the estimator is:

* **jit-compiled** -- the optimisation loop runs entirely on the accelerator;
* **vmapped** over the cross-section -- :func:`estimate_pin_panel` fits
  thousands of stock-quarters in parallel rather than looping in Python;
* **differentiable** -- ``jax.grad`` of PIN with respect to the trade counts is
  well-defined, so an estimated PIN can sit *inside* a larger model (e.g. as an
  input to a neural SDE) and still be trained end-to-end.

The last point is what no existing PIN package offers.

References
----------
Easley, Kiefer, O'Hara & Paperman (1996), *Liquidity, Information, and
Infrequently Traded Stocks*, Journal of Finance 51(4).
Lin & Ke (2011), *A computing bias in estimating the probability of informed
trading*, Journal of Financial Markets 14(4).
Yan & Zhang (2012), *An improved estimation method and empirical properties of
the probability of informed trading*, Journal of Banking & Finance 36(2).
Ersan & Alici (2016), *An unbiased computation methodology for estimating the
probability of informed trading*, Journal of International Financial Markets,
Institutions & Money 43.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jax.scipy.special import gammaln, logsumexp

from .._typing import Array, Float
from ..errors import ShapeError

__all__ = [
    "PINParams",
    "PINResult",
    "pin_log_likelihood",
    "initial_parameter_grid",
    "estimate_pin",
    "estimate_pin_panel",
]

_EPS = 1e-10


class PINParams(eqx.Module):
    """The five structural parameters of the EKOP model.

    Attributes
    ----------
    alpha:
        Probability that an information event occurs on a given day.
    delta:
        Probability that an event is bad news, conditional on one occurring.
    mu:
        Arrival rate of informed traders on event days.
    eps_b, eps_s:
        Arrival rates of uninformed buyers and sellers.

    Examples
    --------
    >>> p = PINParams(alpha=0.4, delta=0.5, mu=100.0, eps_b=50.0, eps_s=50.0)
    >>> round(float(p.pin), 6)  # 0.4*100 / (0.4*100 + 50 + 50)
    0.285714
    """

    alpha: Float[Array, ""]
    delta: Float[Array, ""]
    mu: Float[Array, ""]
    eps_b: Float[Array, ""]
    eps_s: Float[Array, ""]

    @property
    def pin(self) -> Float[Array, ""]:
        """The probability of informed trading."""
        informed = self.alpha * self.mu
        return informed / (informed + self.eps_b + self.eps_s + _EPS)

    def as_array(self) -> Float[Array, " 5"]:
        """Stack the parameters into a ``(5,)`` array."""
        return jnp.stack(
            [
                jnp.asarray(self.alpha),
                jnp.asarray(self.delta),
                jnp.asarray(self.mu),
                jnp.asarray(self.eps_b),
                jnp.asarray(self.eps_s),
            ]
        )


class PINResult(eqx.Module):
    """Outcome of a PIN estimation.

    Attributes
    ----------
    params:
        The fitted :class:`PINParams`.
    pin:
        The PIN statistic implied by ``params``.
    log_likelihood:
        Maximised log-likelihood, summed over days.
    at_boundary:
        ``True`` when ``alpha`` or ``delta`` has been driven to within
        ``boundary_tol`` of 0 or 1. Boundary solutions are a known pathology of
        the Lin--Ke factorization and the resulting PIN should be treated as
        unreliable rather than taken at face value -- so it is reported rather
        than hidden.
    n_starts:
        How many initial values were tried.
    """

    params: PINParams
    pin: Float[Array, ""]
    log_likelihood: Float[Array, ""]
    at_boundary: Array
    n_starts: int = eqx.field(static=True)


# -- Unconstrained reparameterisation ---------------------------------------
#
# Optimisers work in R^5; the model needs alpha, delta in (0, 1) and mu, eps_b,
# eps_s > 0. Transforming is more robust than projecting or clipping, which
# would put zero gradient on any active constraint.


def _constrain(raw: Float[Array, " 5"]) -> PINParams:
    return PINParams(
        alpha=jax.nn.sigmoid(raw[0]),
        delta=jax.nn.sigmoid(raw[1]),
        mu=jax.nn.softplus(raw[2]) + _EPS,
        eps_b=jax.nn.softplus(raw[3]) + _EPS,
        eps_s=jax.nn.softplus(raw[4]) + _EPS,
    )


def _logit(p: Array) -> Array:
    p = jnp.clip(p, 1e-6, 1 - 1e-6)
    return jnp.log(p) - jnp.log1p(-p)


def _softplus_inv(x: Array) -> Array:
    x = jnp.maximum(x, 1e-6)
    # log(exp(x) - 1), computed stably for large x.
    return x + jnp.log(-jnp.expm1(-x))


def _unconstrain(params: PINParams) -> Float[Array, " 5"]:
    return jnp.stack(
        [
            _logit(jnp.asarray(params.alpha)),
            _logit(jnp.asarray(params.delta)),
            _softplus_inv(jnp.asarray(params.mu)),
            _softplus_inv(jnp.asarray(params.eps_b)),
            _softplus_inv(jnp.asarray(params.eps_s)),
        ]
    )


# -- Likelihood --------------------------------------------------------------


def pin_log_likelihood(
    params: PINParams,
    buys: Float[Array, " day"],
    sells: Float[Array, " day"],
) -> Float[Array, ""]:
    """Total log-likelihood of ``(buys, sells)`` under the EKOP model.

    Uses the Lin & Ke (2011) factorization, so it is finite for trade counts far
    beyond where the textbook formula overflows.

    Parameters
    ----------
    params:
        Structural parameters.
    buys, sells:
        Daily counts of buyer- and seller-initiated trades, each ``(day,)``.

    Returns
    -------
    Scalar log-likelihood summed over days.

    Examples
    --------
    The likelihood stays finite at trade counts that overflow the naive formula
    (``eps_b ** B`` with ``B = 50000`` is ``inf`` in float64, let alone float32):

    >>> import jax.numpy as jnp
    >>> p = PINParams(alpha=0.3, delta=0.5, mu=2000.0, eps_b=40000.0, eps_s=40000.0)
    >>> ll = pin_log_likelihood(p, jnp.array([50000.0]), jnp.array([48000.0]))
    >>> bool(jnp.isfinite(ll))
    True

    It behaves like a log-likelihood: the truth beats a badly wrong alternative.

    >>> import jax, jax.random as jr
    >>> key = jr.PRNGKey(0)
    >>> buys = jr.poisson(key, 100.0, (500,)).astype(float)
    >>> sells = jr.poisson(jr.PRNGKey(1), 100.0, (500,)).astype(float)
    >>> good = PINParams(alpha=0.01, delta=0.5, mu=1.0, eps_b=100.0, eps_s=100.0)
    >>> bad = PINParams(alpha=0.9, delta=0.5, mu=500.0, eps_b=10.0, eps_s=10.0)
    >>> bool(pin_log_likelihood(good, buys, sells)
    ...      > pin_log_likelihood(bad, buys, sells))
    True
    """
    buys = jnp.asarray(buys, jnp.float64 if jax.config.jax_enable_x64 else jnp.float32)
    sells = jnp.asarray(sells, buys.dtype)
    if buys.shape != sells.shape:
        raise ShapeError(
            f"buys has shape {buys.shape} but sells has shape {sells.shape}."
        )

    alpha = jnp.clip(params.alpha, _EPS, 1 - _EPS)
    delta = jnp.clip(params.delta, _EPS, 1 - _EPS)
    mu = jnp.maximum(params.mu, _EPS)
    eps_b = jnp.maximum(params.eps_b, _EPS)
    eps_s = jnp.maximum(params.eps_s, _EPS)

    # log(1 + mu/eps), the per-trade log-odds of the "no informed flow" branch.
    x_b = jnp.log1p(mu / eps_b)
    x_s = jnp.log1p(mu / eps_s)

    # Exponents of the three mixture components after factoring out
    #   exp(-eps_b - eps_s) (eps_b + mu)^B (eps_s + mu)^S / (B! S!).
    e_none = -buys * x_b - sells * x_s
    e_bad = -mu - buys * x_b
    e_good = -mu - sells * x_s

    weights = jnp.stack(
        [
            jnp.log1p(-alpha) + e_none,
            jnp.log(alpha) + jnp.log(delta) + e_bad,
            jnp.log(alpha) + jnp.log1p(-delta) + e_good,
        ]
    )
    mixture = logsumexp(weights, axis=0)

    constant = (
        -eps_b
        - eps_s
        + buys * jnp.log(eps_b + mu)
        + sells * jnp.log(eps_s + mu)
        - gammaln(buys + 1.0)
        - gammaln(sells + 1.0)
    )
    return jnp.sum(constant + mixture)


# -- Initial values ----------------------------------------------------------


def initial_parameter_grid(
    buys: Float[Array, " day"],
    sells: Float[Array, " day"],
    *,
    n_alpha: int = 5,
    n_delta: int = 5,
    n_gamma: int = 5,
) -> Float[Array, "start 5"]:
    """Build the Yan & Zhang (2012) grid of initial values.

    The PIN likelihood is multimodal, and a single arbitrary start lands in a
    local optimum often enough to matter empirically. Yan & Zhang's algorithm
    sweeps ``(alpha, delta, gamma)`` over a grid and back-solves the arrival
    rates from the sample means so that every start is *feasible* -- it
    reproduces the observed average order flow:

    .. code-block:: text

        eps_b = gamma * mean(buys)
        mu    = (mean(buys) - eps_b) / (alpha * (1 - delta))
        eps_s = mean(sells) - alpha * delta * mu

    Returns
    -------
    Unconstrained ``(n_alpha * n_delta * n_gamma, 5)`` starting points.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> grid = initial_parameter_grid(jnp.full((10,), 100.0), jnp.full((10,), 90.0))
    >>> grid.shape
    (125, 5)
    >>> bool(jnp.all(jnp.isfinite(grid)))
    True
    """
    buys = jnp.asarray(buys, jnp.float32)
    sells = jnp.asarray(sells, jnp.float32)
    mean_b = jnp.mean(buys)
    mean_s = jnp.mean(sells)

    # Interior grid points; the endpoints 0 and 1 are excluded because they make
    # the back-solved rates singular.
    alphas = jnp.linspace(0.1, 0.9, n_alpha)
    deltas = jnp.linspace(0.1, 0.9, n_delta)
    gammas = jnp.linspace(0.1, 0.9, n_gamma)

    a, d, g = jnp.meshgrid(alphas, deltas, gammas, indexing="ij")
    a, d, g = a.reshape(-1), d.reshape(-1), g.reshape(-1)

    eps_b = g * mean_b
    mu = jnp.maximum((mean_b - eps_b) / (a * (1.0 - d)), 1.0)
    eps_s = jnp.maximum(mean_s - a * d * mu, 1.0)

    return jax.vmap(
        lambda a_, d_, m_, b_, s_: _unconstrain(
            PINParams(alpha=a_, delta=d_, mu=m_, eps_b=b_, eps_s=s_)
        )
    )(a, d, mu, eps_b, eps_s)


# -- Estimation --------------------------------------------------------------


@eqx.filter_jit
def _fit_from_starts(
    starts: Float[Array, "start 5"],
    buys: Float[Array, " day"],
    sells: Float[Array, " day"],
    *,
    steps: int,
    learning_rate: float,
) -> tuple[Float[Array, " 5"], Float[Array, ""]]:
    """Run Adam from every start in parallel and return the best parameters."""
    optimiser = optax.adam(learning_rate)

    def negative_ll(raw):
        return -pin_log_likelihood(_constrain(raw), buys, sells)

    def run_one(raw0):
        opt_state = optimiser.init(raw0)

        def body(carry, _):
            raw, state = carry
            loss, grads = jax.value_and_grad(negative_ll)(raw)
            # A non-finite gradient anywhere would poison the parameters for
            # every remaining step; freeze instead so one bad start cannot take
            # down the whole vmapped batch.
            grads = jnp.where(jnp.isfinite(grads), grads, 0.0)
            updates, state = optimiser.update(grads, state, raw)
            return (optax.apply_updates(raw, updates), state), loss

        (raw, _), _ = jax.lax.scan(body, (raw0, opt_state), None, length=steps)
        return raw, -negative_ll(raw)

    raws, lls = jax.vmap(run_one)(starts)
    lls = jnp.where(jnp.isfinite(lls), lls, -jnp.inf)
    best = jnp.argmax(lls)
    return raws[best], lls[best]


def estimate_pin(
    buys: Float[Array, " day"],
    sells: Float[Array, " day"],
    *,
    steps: int = 500,
    learning_rate: float = 0.05,
    n_alpha: int = 5,
    n_delta: int = 5,
    n_gamma: int = 5,
    boundary_tol: float = 1e-3,
    starts: Float[Array, "start 5"] | None = None,
) -> PINResult:
    """Estimate PIN by maximum likelihood from daily buy and sell counts.

    Parameters
    ----------
    buys, sells:
        Daily counts of buyer- and seller-initiated trades, each ``(day,)``.
        Classify raw trades first with
        :func:`~finax.microstructure.classification.classify_trades`.
    steps:
        Adam iterations per start.
    learning_rate:
        Adam step size, in the unconstrained parameterisation.
    n_alpha, n_delta, n_gamma:
        Shape of the Yan--Zhang initial-value grid. The default 5x5x5 = 125
        starts all run in parallel under ``vmap``, so the cost is a single
        batched solve rather than 125 sequential ones.
    boundary_tol:
        How close ``alpha`` or ``delta`` may get to 0 or 1 before the result is
        flagged via :attr:`PINResult.at_boundary`.
    starts:
        Explicit unconstrained starting points, overriding the grid.

    Returns
    -------
    A :class:`PINResult`.

    Examples
    --------
    Recover a known PIN from data simulated under the model itself:

    >>> import jax, jax.numpy as jnp, jax.random as jr
    >>> true = PINParams(alpha=0.4, delta=0.5, mu=80.0, eps_b=60.0, eps_s=60.0)
    >>> float(true.pin)
    0.2105...
    >>> k1, k2, k3 = jr.split(jr.PRNGKey(0), 3)
    >>> n = 400
    >>> event = jr.bernoulli(k1, 0.4, (n,))
    >>> bad = jr.bernoulli(k2, 0.5, (n,))
    >>> rate_b = 60.0 + 80.0 * (event & ~bad)
    >>> rate_s = 60.0 + 80.0 * (event & bad)
    >>> kb, ks = jr.split(k3)
    >>> buys = jr.poisson(kb, rate_b).astype(float)
    >>> sells = jr.poisson(ks, rate_s).astype(float)
    >>> res = estimate_pin(buys, sells)
    >>> bool(abs(float(res.pin) - float(true.pin)) < 0.05)
    True
    >>> bool(res.at_boundary)
    False
    """
    buys = jnp.asarray(buys)
    sells = jnp.asarray(sells)
    if buys.ndim != 1:
        raise ShapeError(f"buys must be 1-D (day,), got shape {buys.shape}.")
    if buys.shape != sells.shape:
        raise ShapeError(
            f"buys has shape {buys.shape} but sells has shape {sells.shape}."
        )

    if starts is None:
        starts = initial_parameter_grid(
            buys, sells, n_alpha=n_alpha, n_delta=n_delta, n_gamma=n_gamma
        )

    raw, ll = _fit_from_starts(
        starts, buys, sells, steps=steps, learning_rate=learning_rate
    )
    params = _constrain(raw)
    at_boundary = (
        (params.alpha < boundary_tol)
        | (params.alpha > 1 - boundary_tol)
        | (params.delta < boundary_tol)
        | (params.delta > 1 - boundary_tol)
    )
    return PINResult(
        params=params,
        pin=params.pin,
        log_likelihood=ll,
        at_boundary=at_boundary,
        n_starts=int(starts.shape[0]),
    )


def estimate_pin_panel(
    buys: Float[Array, "series day"],
    sells: Float[Array, "series day"],
    *,
    steps: int = 500,
    learning_rate: float = 0.05,
    n_alpha: int = 5,
    n_delta: int = 5,
    n_gamma: int = 5,
    boundary_tol: float = 1e-3,
) -> PINResult:
    """Estimate PIN for a whole cross-section at once.

    Fits every row of ``buys``/``sells`` independently, but as one compiled,
    vectorised computation rather than a Python loop.

    How much time that saves depends on the hardware. On CPU the gain is modest
    (roughly 1.5x for 64 series in our measurements) because the inner
    125-start ``vmap`` already saturates the available cores; the batching pays
    off on GPU/TPU, where parallelism is left over, and it always avoids
    per-series Python dispatch overhead.

    Returns
    -------
    A :class:`PINResult` whose fields carry a leading ``series`` axis.

    Examples
    --------
    >>> import jax.numpy as jnp, jax.random as jr
    >>> keys = jr.split(jr.PRNGKey(0), 4)
    >>> buys = jnp.stack([jr.poisson(k, 100.0, (200,)).astype(float) for k in keys])
    >>> sells = jnp.stack([jr.poisson(k, 95.0, (200,)).astype(float) for k in keys])
    >>> res = estimate_pin_panel(buys, sells, steps=200)
    >>> res.pin.shape
    (4,)
    >>> bool(jnp.all((res.pin >= 0.0) & (res.pin <= 1.0)))
    True
    """
    buys = jnp.asarray(buys)
    sells = jnp.asarray(sells)
    if buys.ndim != 2:
        raise ShapeError(f"buys must be 2-D (series, day), got shape {buys.shape}.")
    if buys.shape != sells.shape:
        raise ShapeError(
            f"buys has shape {buys.shape} but sells has shape {sells.shape}."
        )

    def one(b, s):
        starts = initial_parameter_grid(
            b, s, n_alpha=n_alpha, n_delta=n_delta, n_gamma=n_gamma
        )
        raw, ll = _fit_from_starts(starts, b, s, steps=steps, learning_rate=learning_rate)
        return raw, ll

    raws, lls = eqx.filter_jit(jax.vmap(one))(buys, sells)
    params = jax.vmap(_constrain)(raws)
    at_boundary = (
        (params.alpha < boundary_tol)
        | (params.alpha > 1 - boundary_tol)
        | (params.delta < boundary_tol)
        | (params.delta > 1 - boundary_tol)
    )
    n_starts = n_alpha * n_delta * n_gamma
    return PINResult(
        params=params,
        pin=jax.vmap(lambda p: p.pin)(params),
        log_likelihood=lls,
        at_boundary=at_boundary,
        n_starts=n_starts,
    )
