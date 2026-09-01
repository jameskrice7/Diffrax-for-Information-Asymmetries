"""Shared fixtures.

Note the ``x64`` fixture: several estimators are only accurate in double
precision, and JAX defaults to float32. Tests that need it request it explicitly
rather than flipping a global, because ``jax_enable_x64`` must be set before any
array is created and leaking it across tests causes confusing failures.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest


@pytest.fixture
def key():
    """A deterministic PRNG key."""
    return jr.PRNGKey(0)


@pytest.fixture
def x64():
    """Enable float64 for the duration of one test, then restore."""
    original = bool(jax.config.jax_enable_x64)
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", original)


@pytest.fixture
def irregular_series():
    """An irregularly sampled, partially observed two-channel series."""
    ts = jnp.array([0.0, 0.4, 1.7, 2.0, 3.9, 5.0])
    ys = jnp.array(
        [
            [1.0, 10.0],
            [jnp.nan, 11.0],
            [3.0, jnp.nan],
            [4.0, 13.0],
            [jnp.nan, jnp.nan],
            [6.0, 15.0],
        ]
    )
    return ts, ys


@pytest.fixture
def pin_sample():
    """Daily buy/sell counts simulated from a known PIN model.

    Returns ``(buys, sells, true_params)``.
    """
    from finax.microstructure import PINParams

    true = PINParams(alpha=0.35, delta=0.5, mu=90.0, eps_b=70.0, eps_s=70.0)
    k1, k2, k3, k4 = jr.split(jr.PRNGKey(42), 4)
    n = 500

    event = jr.bernoulli(k1, float(true.alpha), (n,))
    bad = jr.bernoulli(k2, float(true.delta), (n,))
    rate_b = float(true.eps_b) + float(true.mu) * (event & ~bad)
    rate_s = float(true.eps_s) + float(true.mu) * (event & bad)

    buys = jr.poisson(k3, rate_b).astype(jnp.float32)
    sells = jr.poisson(k4, rate_s).astype(jnp.float32)
    return buys, sells, true


@pytest.fixture
def price_frame():
    """A small OHLCV DataFrame with a DatetimeIndex."""
    pd = pytest.importorskip("pandas")
    import numpy as np

    rng = np.random.default_rng(0)
    index = pd.date_range("2024-01-01", periods=250, freq="D")
    close = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, 250)))
    return pd.DataFrame(
        {
            "open": close * (1 + rng.normal(0, 0.001, 250)),
            "high": close * (1 + np.abs(rng.normal(0, 0.005, 250))),
            "low": close * (1 - np.abs(rng.normal(0, 0.005, 250))),
            "close": close,
            "volume": rng.integers(1_000, 100_000, 250).astype(float),
        },
        index=index,
    )
