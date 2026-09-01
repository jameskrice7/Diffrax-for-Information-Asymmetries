"""Trade-sign classification: turning raw trades into buys and sells.

Most microstructure estimators need to know whether each trade was initiated by
a buyer or a seller, but exchanges do not publish that. It has to be inferred,
and the inference is not innocuous: Boehmer, Grammig & Theissen (2007) show that
misclassification biases PIN estimates downward, so the choice of algorithm
changes the headline result.

Three algorithms are provided, in increasing order of data requirements:

:func:`tick_rule`
    Uses only the trade price sequence.
:func:`quote_rule` / :func:`lee_ready`
    Uses the prevailing bid and ask. Lee--Ready is the standard for trade-level
    data and what most published PIN estimates use.
:func:`bulk_volume_classification`
    Assigns a *fraction* of each bar's volume to buyers rather than labelling
    individual trades. Designed for the volume-bar setting of VPIN.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .._typing import Array, Float, Int
from ..errors import ShapeError

__all__ = [
    "tick_rule",
    "quote_rule",
    "lee_ready",
    "bulk_volume_classification",
    "aggregate_daily_counts",
]


def tick_rule(prices: Float[Array, " trade"]) -> Int[Array, " trade"]:
    """Classify trades by the sign of the price change.

    A trade above the previous *different* price is a buy (``+1``), below is a
    sell (``-1``); an unchanged price inherits the previous classification (a
    "zero tick"). Carrying the sign forward is what distinguishes the tick rule
    from a naive ``sign(diff)``, and it matters because zero ticks are the
    majority of observations in liquid names.

    The first trade has no predecessor and is classified ``+1`` by convention.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> tick_rule(jnp.array([10.0, 10.1, 10.1, 10.0, 10.0]))
    Array([ 1,  1,  1, -1, -1], dtype=int32)
    """
    prices = jnp.asarray(prices)
    if prices.ndim != 1:
        raise ShapeError(f"prices must be 1-D, got shape {prices.shape}.")

    raw = jnp.sign(jnp.diff(prices, prepend=prices[0]))

    def step(carry, s):
        sign = jnp.where(s == 0, carry, s)
        return sign, sign

    _, signs = jax.lax.scan(step, jnp.asarray(1.0, prices.dtype), raw)
    return signs.astype(jnp.int32)


def quote_rule(
    prices: Float[Array, " trade"],
    bids: Float[Array, " trade"],
    asks: Float[Array, " trade"],
) -> Int[Array, " trade"]:
    """Classify trades against the prevailing quote midpoint.

    Above the midpoint is a buy, below is a sell. Trades exactly *at* the
    midpoint are returned as ``0`` (unclassified) -- :func:`lee_ready` is what
    resolves those.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> quote_rule(jnp.array([10.2, 10.0, 10.1]),
    ...            jnp.array([10.0, 10.0, 10.0]),
    ...            jnp.array([10.2, 10.2, 10.2]))
    Array([ 1, -1,  0], dtype=int32)
    """
    prices, bids, asks = jnp.asarray(prices), jnp.asarray(bids), jnp.asarray(asks)
    if not (prices.shape == bids.shape == asks.shape):
        raise ShapeError(
            f"prices {prices.shape}, bids {bids.shape} and asks {asks.shape} "
            "must all have the same shape."
        )
    mid = 0.5 * (bids + asks)
    return jnp.sign(prices - mid).astype(jnp.int32)


def lee_ready(
    prices: Float[Array, " trade"],
    bids: Float[Array, " trade"],
    asks: Float[Array, " trade"],
) -> Int[Array, " trade"]:
    """Lee & Ready (1991) trade classification.

    The quote rule where the trade is away from the midpoint, and the tick rule
    where it is exactly at the midpoint. This hybrid is the field standard.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> prices = jnp.array([10.2, 10.0, 10.1, 10.1])
    >>> bids = jnp.full((4,), 10.0)
    >>> asks = jnp.full((4,), 10.2)
    >>> lee_ready(prices, bids, asks)  # midpoint trades fall back to the tick rule
    Array([ 1, -1,  1,  1], dtype=int32)

    References
    ----------
    Lee & Ready (1991), *Inferring Trade Direction from Intraday Data*,
    Journal of Finance 46(2).
    """
    quote = quote_rule(prices, bids, asks)
    tick = tick_rule(prices)
    return jnp.where(quote == 0, tick, quote).astype(jnp.int32)


def bulk_volume_classification(
    prices: Float[Array, " bar"],
    volumes: Float[Array, " bar"],
    *,
    sigma: Float[Array, ""] | float | None = None,
    df: float | None = None,
) -> tuple[Float[Array, " bar"], Float[Array, " bar"]]:
    """Split each bar's volume into buy and sell components (Easley et al., 2012).

    Rather than labelling individual trades, BVC assigns a *fraction* of the
    bar's volume to buyers:

    .. math:: V^{buy}_\\tau = V_\\tau \\cdot Z\\!\\left(
              \\frac{P_\\tau - P_{\\tau-1}}{\\sigma_{\\Delta P}}\\right)

    where ``Z`` is a standard normal (or Student-t) CDF. This is what
    :func:`~finax.microstructure.vpin.vpin` uses, and it is deliberately *not*
    a hard classification: a bar with a small price move is treated as close to
    balanced rather than being forced entirely to one side.

    Being smooth, it is also differentiable -- unlike the tick and quote rules,
    whose ``sign`` has zero gradient almost everywhere.

    Parameters
    ----------
    prices:
        Closing price of each bar.
    volumes:
        Total volume of each bar.
    sigma:
        Standard deviation of price changes. Estimated from ``prices`` if not
        given.
    df:
        If given, use a Student-t CDF with ``df`` degrees of freedom instead of
        the normal. Easley et al. recommend this for fat-tailed price changes.

    Returns
    -------
    ``(buy_volume, sell_volume)``, each ``(bar,)`` and summing to ``volumes``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> p = jnp.array([100.0, 101.0, 100.5, 102.0])
    >>> v = jnp.full((4,), 1000.0)
    >>> buy, sell = bulk_volume_classification(p, v)
    >>> bool(jnp.allclose(buy + sell, v))
    True

    A rising bar is assigned more buy than sell volume:

    >>> bool(buy[1] > sell[1])
    True
    """
    prices = jnp.asarray(prices)
    volumes = jnp.asarray(volumes)
    if prices.shape != volumes.shape:
        raise ShapeError(
            f"prices {prices.shape} and volumes {volumes.shape} must have the same shape."
        )

    dp = jnp.diff(prices, prepend=prices[0])
    if sigma is None:
        sigma = jnp.std(dp)
    sigma = jnp.maximum(jnp.asarray(sigma), 1e-12)

    z = dp / sigma
    frac = jax.scipy.stats.norm.cdf(z) if df is None else jax.scipy.stats.t.cdf(z, df)

    buy = volumes * frac
    return buy, volumes - buy


def aggregate_daily_counts(
    day_index: Int[Array, " trade"],
    signs: Int[Array, " trade"],
    *,
    n_days: int,
) -> tuple[Float[Array, " day"], Float[Array, " day"]]:
    """Sum signed trades into per-day buy and sell counts for PIN estimation.

    Parameters
    ----------
    day_index:
        Zero-based day number for each trade. Must be in ``[0, n_days)``.
    signs:
        ``+1``/``-1`` classification from e.g. :func:`lee_ready`.
    n_days:
        Number of days. Static, because it sets the output shape.

    Returns
    -------
    ``(buys, sells)``, each ``(n_days,)``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> days = jnp.array([0, 0, 0, 1, 1])
    >>> signs = jnp.array([1, 1, -1, -1, -1])
    >>> buys, sells = aggregate_daily_counts(days, signs, n_days=2)
    >>> buys.tolist(), sells.tolist()
    ([2.0, 0.0], [1.0, 2.0])
    """
    day_index = jnp.asarray(day_index)
    signs = jnp.asarray(signs)
    if day_index.shape != signs.shape:
        raise ShapeError(
            f"day_index {day_index.shape} and signs {signs.shape} must match."
        )

    zeros = jnp.zeros((n_days,), jnp.float32)
    buys = zeros.at[day_index].add((signs > 0).astype(jnp.float32))
    sells = zeros.at[day_index].add((signs < 0).astype(jnp.float32))
    return buys, sells
