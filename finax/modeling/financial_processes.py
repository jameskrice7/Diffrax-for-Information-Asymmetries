"""Financial process generators with jumps/discontinuities and regime changes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class JumpDiffusionConfig:
    drift: float = 0.05
    volatility: float = 0.2
    jump_intensity: float = 0.3
    jump_mean: float = -0.02
    jump_std: float = 0.08


@dataclass(frozen=True)
class RegimeSwitchingConfig:
    low_vol: float = 0.1
    high_vol: float = 0.35
    transition_up: float = 0.02
    transition_down: float = 0.05


def simulate_jump_diffusion(
    key: Any,
    *,
    n_steps: int,
    dt: float,
    s0: float = 1.0,
    config: JumpDiffusionConfig | None = None,
) -> jnp.ndarray:
    """Simulate Merton-style jump diffusion prices."""
    cfg = config or JumpDiffusionConfig()
    kb, kj, kz = jax.random.split(key, 3)
    d_w = jnp.sqrt(dt) * jax.random.normal(kb, shape=(n_steps,))
    jump_counts = jax.random.poisson(kj, lam=cfg.jump_intensity * dt, shape=(n_steps,))
    jump_sizes = cfg.jump_mean + cfg.jump_std * jax.random.normal(kz, shape=(n_steps,))
    jump_component = jump_counts * jump_sizes
    d_log_s = (cfg.drift - 0.5 * cfg.volatility**2) * dt + cfg.volatility * d_w + jump_component
    log_prices = jnp.cumsum(d_log_s) + jnp.log(s0)
    return jnp.concatenate([jnp.array([s0]), jnp.exp(log_prices)])


def simulate_regime_switching_process(
    key: Any,
    *,
    n_steps: int,
    dt: float,
    x0: float = 0.0,
    config: RegimeSwitchingConfig | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Simulate a 2-state volatility process with Markov switching."""
    cfg = config or RegimeSwitchingConfig()
    key_state, key_noise = jax.random.split(key)
    transitions = jax.random.uniform(key_state, shape=(n_steps,))
    noise = jax.random.normal(key_noise, shape=(n_steps,))

    def step(carry, inputs):
        state, x = carry
        u, eps = inputs
        p_up = jnp.where(state == 0, cfg.transition_up, 0.0)
        p_down = jnp.where(state == 1, cfg.transition_down, 0.0)
        next_state = jnp.where(
            state == 0,
            jnp.where(u < p_up, 1, 0),
            jnp.where(u < p_down, 0, 1),
        )
        sigma = jnp.where(next_state == 0, cfg.low_vol, cfg.high_vol)
        next_x = x + sigma * jnp.sqrt(dt) * eps
        return (next_state, next_x), (next_state, next_x)

    (_, _), (states, xs) = jax.lax.scan(step, (jnp.array(0), jnp.array(x0)), (transitions, noise))
    xs = jnp.concatenate([jnp.array([x0]), xs])
    states = jnp.concatenate([jnp.array([0]), states])
    return states, xs


def inject_discontinuities(
    series: jnp.ndarray,
    *,
    key: Any,
    intensity: float = 0.02,
    jump_scale: float = 0.05,
) -> jnp.ndarray:
    """Add synthetic discontinuities to a process for stress testing."""
    k1, k2 = jax.random.split(key)
    mask = jax.random.bernoulli(k1, p=intensity, shape=series.shape)
    shocks = jump_scale * jax.random.normal(k2, shape=series.shape)
    return series + mask * shocks
