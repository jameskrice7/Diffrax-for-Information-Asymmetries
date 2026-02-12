"""Neural SDE utilities for advanced financial time-series simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp


Array = jnp.ndarray


@dataclass(frozen=True)
class NSDEConfig:
    state_dim: int
    hidden_dims: tuple[int, ...] = (128, 128, 128)
    diffusion_floor: float = 1e-4


class MLP:
    """Minimal JAX MLP usable for large parameter counts (10k+)."""

    def __init__(self, layer_dims: Sequence[int], *, activation: Callable[[Array], Array] = jax.nn.swish):
        self.layer_dims = tuple(layer_dims)
        self.activation = activation

    def init(self, key: Any) -> list[dict[str, Array]]:
        params = []
        keys = jax.random.split(key, len(self.layer_dims) - 1)
        for k, (din, dout) in zip(keys, zip(self.layer_dims[:-1], self.layer_dims[1:])):
            wk, bk = jax.random.split(k)
            w = jax.random.normal(wk, (din, dout)) / jnp.sqrt(din)
            b = jnp.zeros((dout,))
            params.append({"w": w, "b": b})
        return params

    def apply(self, params: list[dict[str, Array]], x: Array) -> Array:
        for i, p in enumerate(params):
            x = x @ p["w"] + p["b"]
            if i < len(params) - 1:
                x = self.activation(x)
        return x


class NeuralFinancialSDE:
    """Composable neural SDE with optional jump and discontinuity support."""

    def __init__(
        self,
        config: NSDEConfig,
        *,
        drift_net: MLP | None = None,
        diffusion_net: MLP | None = None,
    ):
        self.config = config
        self.drift_net = drift_net or MLP((config.state_dim, *config.hidden_dims, config.state_dim))
        self.diffusion_net = diffusion_net or MLP((config.state_dim, *config.hidden_dims, config.state_dim))

    def init(self, key: Any) -> dict[str, Any]:
        k1, k2 = jax.random.split(key)
        return {
            "drift": self.drift_net.init(k1),
            "diffusion": self.diffusion_net.init(k2),
        }

    def step(
        self,
        params: dict[str, Any],
        x: Array,
        *,
        dt: float,
        key: Any,
        jump_intensity: float = 0.0,
        jump_scale: float = 0.0,
    ) -> Array:
        kn, kj, kz = jax.random.split(key, 3)
        drift = self.drift_net.apply(params["drift"], x)
        raw_diff = self.diffusion_net.apply(params["diffusion"], x)
        diffusion = jax.nn.softplus(raw_diff) + self.config.diffusion_floor
        eps = jax.random.normal(kn, shape=x.shape)
        jump_occurs = jax.random.bernoulli(kj, p=jnp.clip(jump_intensity * dt, 0.0, 1.0), shape=x.shape)
        jump = jump_occurs * jump_scale * jax.random.normal(kz, shape=x.shape)
        return x + drift * dt + diffusion * jnp.sqrt(dt) * eps + jump

    def simulate(
        self,
        params: dict[str, Any],
        x0: Array,
        *,
        n_steps: int,
        dt: float,
        key: Any,
        jump_intensity: float = 0.0,
        jump_scale: float = 0.0,
    ) -> Array:
        keys = jax.random.split(key, n_steps)

        def _scan_step(x, k):
            nx = self.step(
                params,
                x,
                dt=dt,
                key=k,
                jump_intensity=jump_intensity,
                jump_scale=jump_scale,
            )
            return nx, nx

        _, traj = jax.lax.scan(_scan_step, x0, keys)
        return jnp.concatenate([x0[None, ...], traj], axis=0)


def estimate_parameter_count(layer_dims: Sequence[int]) -> int:
    """Count dense parameters for architecture planning."""
    return int(sum((din * dout + dout) for din, dout in zip(layer_dims[:-1], layer_dims[1:])))
