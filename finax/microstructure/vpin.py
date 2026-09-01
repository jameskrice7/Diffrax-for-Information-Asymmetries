"""VPIN: the volume-synchronised probability of informed trading.

Easley, Lopez de Prado & O'Hara (2012) replace PIN's maximum likelihood with a
direct, model-free order-imbalance measure computed in **volume time**. The
algorithm has three steps, and skipping any of them gives something that is not
VPIN:

1. **Volume bars.** Group trades into consecutive buckets of *equal volume* ``V``
   rather than equal clock time. Information arrives with volume, not with the
   clock, so volume time is the natural clock for this measure.
2. **Bulk volume classification.** Within each bucket, split volume into buy and
   sell sides using the smooth normal-CDF rule of
   :func:`~finax.microstructure.classification.bulk_volume_classification`.
3. **Rolling imbalance.** Average the absolute imbalance over a window of ``n``
   buckets:

   .. math:: \\mathrm{VPIN} = \\frac{\\sum_{\\tau=1}^{n}
             |V^{buy}_\\tau - V^{sell}_\\tau|}{n V}.

Because the whole pipeline is built from smooth operations, VPIN here is
differentiable with respect to prices and volumes.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .._typing import Array, Float
from ..errors import DataValidationError, ShapeError
from .classification import bulk_volume_classification

__all__ = ["volume_bars", "vpin"]


def volume_bars(
    prices: Float[Array, " trade"],
    volumes: Float[Array, " trade"],
    *,
    bucket_volume: float,
    n_buckets: int,
) -> tuple[Float[Array, " bucket"], Float[Array, " bucket"]]:
    """Resample trades into equal-volume buckets.

    Each bucket accumulates trades until ``bucket_volume`` is reached; its price
    is the last trade price inside it. Trades are assigned whole to the bucket
    in which their *cumulative* volume falls, which keeps the operation a single
    vectorised scatter rather than a Python loop over trades.

    Parameters
    ----------
    prices, volumes:
        Trade-level price and size, each ``(trade,)``.
    bucket_volume:
        Target volume per bucket ``V``. A common choice is
        ``total_volume / (50 * n_days)``, i.e. roughly 50 buckets a day.
    n_buckets:
        Number of buckets to produce. Static, because it sets the output shape.
        Any bucket beyond the data is filled by forward-filling the last price
        and carries zero volume, so it contributes nothing.

    Returns
    -------
    ``(bar_prices, bar_volumes)``, each ``(n_buckets,)``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> p = jnp.array([10.0, 11.0, 12.0, 13.0])
    >>> v = jnp.array([50.0, 50.0, 50.0, 50.0])
    >>> bp, bv = volume_bars(p, v, bucket_volume=100.0, n_buckets=2)
    >>> bv.tolist()
    [100.0, 100.0]
    >>> bp.tolist()  # last price within each bucket
    [11.0, 13.0]
    """
    prices = jnp.asarray(prices)
    volumes = jnp.asarray(volumes)
    if prices.shape != volumes.shape:
        raise ShapeError(
            f"prices {prices.shape} and volumes {volumes.shape} must have the same shape."
        )
    if bucket_volume <= 0:
        raise DataValidationError(f"bucket_volume must be positive, got {bucket_volume}.")

    cumulative = jnp.cumsum(volumes)
    # Bucket index of each trade: which multiple of bucket_volume it lands in.
    # Subtracting a trade's own volume makes the boundary trade belong to the
    # bucket it fills rather than the next one.
    idx = jnp.floor((cumulative - volumes) / bucket_volume).astype(jnp.int32)
    idx = jnp.clip(idx, 0, n_buckets - 1)

    bar_volumes = jnp.zeros((n_buckets,), prices.dtype).at[idx].add(volumes)
    # Last price in each bucket: a max-scatter over trade position picks the
    # latest trade, then we gather its price.
    last_pos = (
        jnp.full((n_buckets,), -1, jnp.int32)
        .at[idx]
        .max(jnp.arange(prices.shape[0], dtype=jnp.int32))
    )

    # Empty buckets keep the previous bucket's price (forward fill).
    def ffill(carry, pos):
        pos = jnp.where(pos < 0, carry, pos)
        return pos, pos

    _, last_pos = jax.lax.scan(ffill, jnp.asarray(0, jnp.int32), last_pos)
    return prices[last_pos], bar_volumes


def vpin(
    prices: Float[Array, " bar"],
    volumes: Float[Array, " bar"],
    *,
    window: int = 50,
    sigma: float | None = None,
    df: float | None = None,
) -> Float[Array, " bar"]:
    """Compute rolling VPIN over pre-built volume bars.

    Pass bars from :func:`volume_bars`, not raw trades and not time bars --
    equal-volume bucketing is part of the definition.

    Parameters
    ----------
    prices, volumes:
        Bar close prices and bar volumes, each ``(bar,)``.
    window:
        Number of buckets ``n`` in the rolling average. Easley et al. use 50.
    sigma, df:
        Forwarded to
        :func:`~finax.microstructure.classification.bulk_volume_classification`.

    Returns
    -------
    ``(bar,)`` array. The first ``window - 1`` entries are ``nan``, since the
    window is not yet full.

    Examples
    --------
    Perfectly balanced flow gives VPIN near zero:

    >>> import jax.numpy as jnp
    >>> n = 200
    >>> flat = jnp.full((n,), 100.0)
    >>> out = vpin(flat, jnp.full((n,), 1000.0), window=50)
    >>> bool(out[-1] < 0.05)
    True

    A steadily rising price is maximally one-sided, so VPIN approaches 1:

    >>> rising = 100.0 + jnp.arange(n, dtype=jnp.float32)
    >>> out = vpin(rising, jnp.full((n,), 1000.0), window=50)
    >>> bool(out[-1] > 0.9)
    True

    The leading entries are undefined until the window fills:

    >>> bool(jnp.isnan(out[:49]).all()), bool(jnp.isfinite(out[49:]).all())
    (True, True)
    """
    prices = jnp.asarray(prices)
    volumes = jnp.asarray(volumes)
    if prices.shape != volumes.shape:
        raise ShapeError(
            f"prices {prices.shape} and volumes {volumes.shape} must have the same shape."
        )
    if window < 1:
        raise DataValidationError(f"window must be >= 1, got {window}.")
    if prices.shape[0] < window:
        raise DataValidationError(
            f"Need at least window={window} bars, got {prices.shape[0]}."
        )

    buy, sell = bulk_volume_classification(prices, volumes, sigma=sigma, df=df)
    imbalance = jnp.abs(buy - sell)

    # Rolling sums via cumulative sums: O(n) rather than O(n * window).
    def rolling_sum(x):
        c = jnp.concatenate([jnp.zeros((1,), x.dtype), jnp.cumsum(x)])
        return c[window:] - c[:-window]

    numerator = rolling_sum(imbalance)
    denominator = rolling_sum(volumes)
    values = numerator / jnp.maximum(denominator, 1e-12)

    pad = jnp.full((window - 1,), jnp.nan, values.dtype)
    return jnp.concatenate([pad, values])
