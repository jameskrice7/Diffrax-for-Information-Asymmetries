"""Liquidity and price-impact measures.

Complements the PIN family with the other standard proxies for adverse
selection. All are pure JAX, so they ``vmap`` across a cross-section and
differentiate with respect to their inputs.

Each estimator has a documented data requirement, and they are ordered here from
most to least demanding: :func:`kyle_lambda` needs signed order flow,
:func:`amihud_illiquidity` needs only daily returns and dollar volume, and
:func:`corwin_schultz_spread` needs only daily highs and lows.
"""

from __future__ import annotations

import jax.numpy as jnp

from .._typing import Array, Float
from ..errors import DataValidationError, ShapeError

__all__ = [
    "kyle_lambda",
    "amihud_illiquidity",
    "roll_spread",
    "corwin_schultz_spread",
    "effective_spread",
    "realized_spread",
    "price_impact",
]


def kyle_lambda(
    price_changes: Float[Array, " period"],
    signed_volume: Float[Array, " period"],
) -> Float[Array, ""]:
    """Kyle's lambda: the price impact of a unit of signed order flow.

    The slope of ``dP = lambda * signed_volume + noise``, fitted by OLS through
    the origin. In Kyle's (1985) model lambda is inversely proportional to
    market depth and increases in the amount of private information, so it reads
    directly as an adverse-selection cost.

    Parameters
    ----------
    price_changes:
        Price change over each period.
    signed_volume:
        Buy volume minus sell volume over the same period.

    Returns
    -------
    Scalar lambda.

    Examples
    --------
    Recovers a known impact coefficient exactly in the noiseless case:

    >>> import jax.numpy as jnp
    >>> flow = jnp.array([100.0, -50.0, 200.0, -300.0])
    >>> bool(jnp.allclose(kyle_lambda(0.002 * flow, flow), 0.002))
    True
    """
    price_changes = jnp.asarray(price_changes)
    signed_volume = jnp.asarray(signed_volume)
    if price_changes.shape != signed_volume.shape:
        raise ShapeError(
            f"price_changes {price_changes.shape} and signed_volume "
            f"{signed_volume.shape} must have the same shape."
        )
    denominator = jnp.sum(signed_volume**2)
    return jnp.sum(price_changes * signed_volume) / jnp.maximum(denominator, 1e-12)


def amihud_illiquidity(
    returns: Float[Array, " day"],
    dollar_volume: Float[Array, " day"],
) -> Float[Array, ""]:
    """Amihud (2002) illiquidity: average absolute return per dollar traded.

    .. math:: \\mathrm{ILLIQ} = \\frac{1}{T}\\sum_t \\frac{|r_t|}{\\mathrm{DVOL}_t}

    Crude but robust, and computable from daily data alone, which is why it is
    the most widely used illiquidity proxy in the asset-pricing literature.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> r = jnp.array([0.01, -0.02, 0.015])
    >>> dv = jnp.array([1e6, 1e6, 1e6])
    >>> bool(abs(float(amihud_illiquidity(r, dv)) * 1e6 - 0.015) < 1e-6)
    True

    Doubling volume halves illiquidity:

    >>> bool(jnp.allclose(amihud_illiquidity(r, 2 * dv),
    ...                   amihud_illiquidity(r, dv) / 2))
    True
    """
    returns = jnp.asarray(returns)
    dollar_volume = jnp.asarray(dollar_volume)
    if returns.shape != dollar_volume.shape:
        raise ShapeError(
            f"returns {returns.shape} and dollar_volume {dollar_volume.shape} "
            "must have the same shape."
        )
    return jnp.mean(jnp.abs(returns) / jnp.maximum(dollar_volume, 1e-12))


def roll_spread(prices: Float[Array, " period"]) -> Float[Array, ""]:
    """Roll (1984) implied effective spread from serial covariance of price changes.

    .. math:: s = 2\\sqrt{-\\mathrm{Cov}(\\Delta p_t, \\Delta p_{t-1})}

    Bid-ask bounce makes consecutive price changes negatively autocorrelated,
    and the strength of that effect identifies the spread. When the sample
    covariance comes out *positive* the model is rejected by the data; this
    returns ``0.0`` there rather than ``nan``, which is the convention in the
    empirical literature.

    Examples
    --------
    A pure bid-ask bounce of half-spread 0.05 around a constant efficient price
    implies a spread of 0.10. Note the trade signs must be *random*: Roll's
    identification relies on independent bounce, and a deterministic alternation
    would double the autocovariance and hence the estimate.

    >>> import jax.numpy as jnp, jax.random as jr
    >>> signs = 2.0 * jr.bernoulli(jr.PRNGKey(0), 0.5, (20000,)) - 1.0
    >>> p = 100.0 + 0.05 * signs
    >>> bool(abs(float(roll_spread(p)) - 0.10) < 0.005)
    True

    A trending price gives positive autocovariance, so the estimator is
    truncated at zero:

    >>> float(roll_spread(jnp.arange(100, dtype=jnp.float32)))
    0.0
    """
    prices = jnp.asarray(prices)
    if prices.ndim != 1:
        raise ShapeError(f"prices must be 1-D, got shape {prices.shape}.")
    if prices.shape[0] < 3:
        raise DataValidationError("roll_spread needs at least 3 prices.")

    dp = jnp.diff(prices)
    x, y = dp[:-1], dp[1:]
    cov = jnp.mean((x - jnp.mean(x)) * (y - jnp.mean(y)))
    return 2.0 * jnp.sqrt(jnp.maximum(-cov, 0.0))


def corwin_schultz_spread(
    highs: Float[Array, " day"],
    lows: Float[Array, " day"],
) -> Float[Array, " day-1"]:
    """Corwin & Schultz (2012) high--low spread estimator.

    Exploits the fact that the high-low range over two days reflects both
    volatility and the spread, but volatility scales with time while the spread
    does not. Separating the two identifies the spread from daily OHLC data --
    no intraday data required, which is why this estimator is so widely used for
    historical and emerging-market samples.

    Returns
    -------
    ``(day-1,)`` array of two-day spread estimates as a proportion of price.
    Negative estimates are set to zero, following the authors' recommendation.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> highs = jnp.array([101.0, 102.0, 101.5, 103.0])
    >>> lows = jnp.array([99.0, 100.0, 99.5, 101.0])
    >>> s = corwin_schultz_spread(highs, lows)
    >>> s.shape
    (3,)
    >>> bool(jnp.all(s >= 0.0))
    True
    """
    highs = jnp.asarray(highs)
    lows = jnp.asarray(lows)
    if highs.shape != lows.shape:
        raise ShapeError(
            f"highs {highs.shape} and lows {lows.shape} must have the same shape."
        )
    if highs.shape[0] < 2:
        raise DataValidationError("corwin_schultz_spread needs at least 2 days.")

    log_hl = jnp.log(highs / lows)
    # beta: sum of two consecutive single-day squared log ranges.
    beta = log_hl[:-1] ** 2 + log_hl[1:] ** 2

    # gamma: squared log range over the two-day window.
    high2 = jnp.maximum(highs[:-1], highs[1:])
    low2 = jnp.minimum(lows[:-1], lows[1:])
    gamma = jnp.log(high2 / low2) ** 2

    k = 3.0 - 2.0 * jnp.sqrt(2.0)
    alpha = (jnp.sqrt(2.0 * beta) - jnp.sqrt(beta)) / k - jnp.sqrt(gamma / k)
    spread = 2.0 * (jnp.exp(alpha) - 1.0) / (1.0 + jnp.exp(alpha))
    return jnp.maximum(spread, 0.0)


def effective_spread(
    trade_prices: Float[Array, " trade"],
    midpoints: Float[Array, " trade"],
    signs: Float[Array, " trade"],
) -> Float[Array, " trade"]:
    """Twice the signed distance from the trade price to the quote midpoint.

    .. math:: \\mathrm{ES}_t = 2 q_t (P_t - M_t) / M_t

    What the trade actually cost relative to the midpoint, as a proportion.
    Unlike the quoted spread it reflects where trades really executed, including
    price improvement inside the quote.

    .. warning::
       ``P_t - M_t`` is a difference of two nearly equal large numbers. In JAX's
       default float32 a price of 100 with a half-cent spread retains only about
       three significant digits of the difference. Enable float64 --
       ``jax.config.update("jax_enable_x64", True)`` -- for trade-level spread
       work.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> es = effective_spread(jnp.array([100.05]), jnp.array([100.0]),
    ...                       jnp.array([1.0]))
    >>> bool(jnp.allclose(es, 0.001, atol=1e-6))
    True
    """
    trade_prices = jnp.asarray(trade_prices)
    midpoints = jnp.asarray(midpoints)
    signs = jnp.asarray(signs)
    if not (trade_prices.shape == midpoints.shape == signs.shape):
        raise ShapeError(
            f"trade_prices {trade_prices.shape}, midpoints {midpoints.shape} and "
            f"signs {signs.shape} must all have the same shape."
        )
    return 2.0 * signs * (trade_prices - midpoints) / jnp.maximum(midpoints, 1e-12)


def realized_spread(
    trade_prices: Float[Array, " trade"],
    future_midpoints: Float[Array, " trade"],
    signs: Float[Array, " trade"],
) -> Float[Array, " trade"]:
    """The part of the effective spread the liquidity provider keeps.

    .. math:: \\mathrm{RS}_t = 2 q_t (P_t - M_{t+\\Delta}) / M_{t+\\Delta}

    Measured against the midpoint *after* the trade, so it nets out the
    permanent price move the trade caused. Effective spread minus realized
    spread is :func:`price_impact`, the adverse-selection component -- this
    decomposition is the standard way to separate what market makers earn from
    what informed traders take.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> rs = realized_spread(jnp.array([100.05]), jnp.array([100.02]),
    ...                      jnp.array([1.0]))
    >>> bool(jnp.allclose(rs, 2 * 0.03 / 100.02, atol=1e-6))
    True
    """
    trade_prices = jnp.asarray(trade_prices)
    future_midpoints = jnp.asarray(future_midpoints)
    signs = jnp.asarray(signs)
    if not (trade_prices.shape == future_midpoints.shape == signs.shape):
        raise ShapeError(
            f"trade_prices {trade_prices.shape}, future_midpoints "
            f"{future_midpoints.shape} and signs {signs.shape} must all have "
            "the same shape."
        )
    return (
        2.0
        * signs
        * (trade_prices - future_midpoints)
        / jnp.maximum(future_midpoints, 1e-12)
    )


def price_impact(
    trade_prices: Float[Array, " trade"],
    midpoints: Float[Array, " trade"],
    future_midpoints: Float[Array, " trade"],
    signs: Float[Array, " trade"],
) -> Float[Array, " trade"]:
    """Adverse-selection component: effective spread minus realized spread.

    This is the part of trading costs attributable to informed trading, and the
    most direct trade-level analogue of PIN.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> pi = price_impact(jnp.array([100.05]), jnp.array([100.0]),
    ...                   jnp.array([100.02]), jnp.array([1.0]))
    >>> bool(pi[0] > 0)  # the midpoint moved up after a buy
    True
    """
    return effective_spread(trade_prices, midpoints, signs) - realized_spread(
        trade_prices, future_midpoints, signs
    )
