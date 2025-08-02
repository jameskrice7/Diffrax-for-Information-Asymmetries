"""Basic stochastic process simulators."""

from __future__ import annotations

try:  # pragma: no cover - optional at import
    import jax.numpy as jnp
    from jax import random
except Exception:  # pragma: no cover
    jnp = None  # type: ignore
    random = None  # type: ignore


def _check_jax() -> None:
    if jnp is None or random is None:
        raise ImportError("JAX is required for stochastic simulations")


def brownian_motion(key, steps: int, dt: float = 1.0, scale: float = 1.0):
    """Simulate a Brownian motion path.

    Parameters
    ----------
    key:
        JAX PRNGKey.
    steps:
        Number of time steps.
    dt:
        Time increment between steps.
    scale:
        Standard deviation multiplier.
    """

    _check_jax()
    increments = random.normal(key, (steps,)) * jnp.sqrt(dt) * scale
    return jnp.cumsum(increments)


def poisson_process(key, lam: float, steps: int, dt: float = 1.0):
    """Simulate a Poisson process."""

    _check_jax()
    counts = random.poisson(key, lam * dt, (steps,))
    return jnp.cumsum(counts)
