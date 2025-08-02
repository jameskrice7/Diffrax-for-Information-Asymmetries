"""Flax modules tailored for financial time-series data."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import linen as nn


class FinancialRNN(nn.Module):
    """Simple LSTM block for sequential financial features.

    Parameters
    ----------
    hidden_size:
        Number of hidden units in the LSTM cell.
    """

    hidden_size: int = 32

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply the LSTM to an input of shape ``(batch, time, features)``."""

        lstm = nn.OptimizedLSTMCell(self.hidden_size)
        batch = x.shape[0]
        carry = lstm.initialize_carry(jax.random.PRNGKey(0), (batch,), self.hidden_size)

        def step(carry, x_t):
            carry, y = lstm(carry, x_t)
            return carry, y

        _, ys = jax.lax.scan(step, carry, x.swapaxes(0, 1))
        return ys.swapaxes(0, 1)


class LogReturn(nn.Module):
    """Compute log returns from price series.

    Expects inputs of shape ``(batch, time)`` and returns ``(batch, time-1)``.
    """

    def __call__(self, prices: jnp.ndarray) -> jnp.ndarray:  # pragma: no cover - simple wrapper
        return jnp.diff(jnp.log(prices), axis=1)


__all__ = ["FinancialRNN", "LogReturn"]
