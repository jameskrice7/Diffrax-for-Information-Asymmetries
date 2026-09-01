"""Loss functions for differential-equation models."""

from __future__ import annotations

import jax.numpy as jnp

from .._typing import Array, Float

__all__ = [
    "mse",
    "mae",
    "gaussian_nll",
    "elbo",
    "quantile_loss",
]


def mse(
    predictions: Float[Array, "..."], targets: Float[Array, "..."]
) -> Float[Array, ""]:
    """Mean squared error, ignoring NaN targets.

    NaN-masking matters here: irregular panels are full of missing observations,
    and a single NaN target would otherwise turn the whole gradient into NaN.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> float(mse(jnp.array([1.0, 2.0]), jnp.array([1.0, 3.0])))
    0.5
    >>> float(mse(jnp.array([1.0, 2.0]), jnp.array([1.0, jnp.nan])))
    0.0
    """
    observed = ~jnp.isnan(targets)
    safe = jnp.where(observed, targets, 0.0)
    errors = jnp.where(observed, (predictions - safe) ** 2, 0.0)
    return jnp.sum(errors) / jnp.maximum(jnp.sum(observed), 1)


def mae(
    predictions: Float[Array, "..."], targets: Float[Array, "..."]
) -> Float[Array, ""]:
    """Mean absolute error, ignoring NaN targets.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> float(mae(jnp.array([1.0, 5.0]), jnp.array([1.0, 3.0])))
    1.0
    """
    observed = ~jnp.isnan(targets)
    safe = jnp.where(observed, targets, 0.0)
    errors = jnp.where(observed, jnp.abs(predictions - safe), 0.0)
    return jnp.sum(errors) / jnp.maximum(jnp.sum(observed), 1)


def gaussian_nll(
    predictions: Float[Array, "..."],
    targets: Float[Array, "..."],
    log_scale: Float[Array, "..."] | float = 0.0,
) -> Float[Array, ""]:
    """Negative Gaussian log-likelihood with a learnable observation scale.

    Parameterised by ``log_scale`` rather than the standard deviation so that
    unconstrained optimisation cannot drive the scale negative. Use this instead
    of :func:`mse` when the observation noise level is itself unknown -- fixing
    it implicitly at 1, as MSE does, distorts the trade-off against any KL or
    regularisation term measured in nats.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> x = jnp.array([0.0])
    >>> float(gaussian_nll(x, x))  # 0.5 * log(2 pi)
    0.9189...
    """
    scale = jnp.exp(log_scale)
    observed = ~jnp.isnan(targets)
    safe = jnp.where(observed, targets, 0.0)
    per_point = (
        0.5 * jnp.log(2 * jnp.pi) + log_scale + 0.5 * ((predictions - safe) / scale) ** 2
    )
    per_point = jnp.where(observed, per_point, 0.0)
    return jnp.sum(per_point) / jnp.maximum(jnp.sum(observed), 1)


def elbo(
    reconstruction: Float[Array, ""],
    kl: Float[Array, ""],
    *,
    beta: float = 1.0,
) -> Float[Array, ""]:
    """Negative evidence lower bound: ``reconstruction + beta * kl``.

    The loss to minimise when training a :class:`~finax.models.LatentSDE`, where
    ``reconstruction`` is a negative log-likelihood such as :func:`gaussian_nll`
    and ``kl`` is :attr:`~finax.models.LatentSDEOutput.kl`.

    ``beta`` is the beta-VAE weight. Values below 1 loosen the prior and are
    often needed early in training: with ``beta = 1`` a latent SDE tends to
    collapse to the prior before the decoder has learned anything useful.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> float(elbo(jnp.array(2.0), jnp.array(1.0), beta=0.5))
    2.5
    """
    return reconstruction + beta * kl


def quantile_loss(
    predictions: Float[Array, "... quantile"],
    targets: Float[Array, "..."],
    quantiles: Float[Array, " quantile"],
) -> Float[Array, ""]:
    """Pinball loss for simultaneous multi-quantile regression.

    Fitting several quantiles at once turns a point forecaster into a
    distributional one, which is usually what you want from a financial model:
    the tails carry the risk. The loss is asymmetric -- underprediction of the
    90th percentile is penalised nine times as heavily as overprediction.

    Parameters
    ----------
    predictions:
        Predicted quantiles, with the quantile axis last.
    targets:
        Observations, broadcast against all but the last axis of ``predictions``.
    quantiles:
        Quantile levels in ``(0, 1)``.

    Examples
    --------
    The 0.1 and 0.9 quantiles are each off by 1.0 in their penalised direction,
    contributing 0.1 apiece; the median is exact. Averaged over the three
    quantile-observation pairs that is ``0.2 / 3``:

    >>> import jax.numpy as jnp
    >>> q = jnp.array([0.1, 0.5, 0.9])
    >>> preds = jnp.array([[0.0, 1.0, 2.0]])
    >>> round(float(quantile_loss(preds, jnp.array([1.0]), q)), 5)
    0.06667
    """
    targets = jnp.expand_dims(targets, -1)
    observed = ~jnp.isnan(targets)
    safe = jnp.where(observed, targets, 0.0)
    errors = safe - predictions
    per_point = jnp.maximum(quantiles * errors, (quantiles - 1.0) * errors)
    per_point = jnp.where(observed, per_point, 0.0)
    return jnp.sum(per_point) / jnp.maximum(jnp.sum(observed) * quantiles.shape[0], 1)
