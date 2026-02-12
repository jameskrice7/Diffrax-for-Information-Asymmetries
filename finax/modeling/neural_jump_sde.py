"""Neural jump diffusion SDE models."""

from __future__ import annotations

from typing import Any, Callable

try:  # pragma: no cover
    import jax
    import jax.numpy as jnp
    import diffrax
except Exception:  # pragma: no cover
    jax = None  # type: ignore
    jnp = None  # type: ignore
    diffrax = None  # type: ignore


class NeuralJumpSDE:
    """Neural SDE with an explicit jump perturbation channel."""

    def __init__(
        self,
        drift: Callable[[Any, Any, Any], Any],
        diffusion: Callable[[Any, Any, Any], Any],
        jump: Callable[[Any, Any, Any], Any],
    ):
        self.drift = drift
        self.diffusion = diffusion
        self.jump = jump

    def simulate(
        self,
        y0: Any,
        t0: float,
        t1: float,
        *,
        key: Any,
        dt: float = 0.01,
        jump_intensity: float = 0.2,
    ) -> Any:
        """Simulate with Euler-style updates supporting discontinuities."""
        if jax is None or jnp is None:
            raise ImportError("JAX is required for jump SDE simulation.")

        n_steps = max(1, int((t1 - t0) / dt))
        keys = jax.random.split(key, n_steps)

        def step(y, k):
            kn, kj = jax.random.split(k)
            t = 0.0
            dw = jnp.sqrt(dt) * jax.random.normal(kn, shape=jnp.shape(y))
            y_next = y + self.drift(t, y, None) * dt + self.diffusion(t, y, None) * dw
            jump_mask = jax.random.bernoulli(kj, p=jnp.clip(jump_intensity * dt, 0.0, 1.0), shape=jnp.shape(y))
            y_next = y_next + jump_mask * self.jump(t, y_next, None)
            return y_next, y_next

        _, ys = jax.lax.scan(step, y0, keys)
        return jnp.concatenate([jnp.asarray(y0)[None, ...], ys], axis=0)
