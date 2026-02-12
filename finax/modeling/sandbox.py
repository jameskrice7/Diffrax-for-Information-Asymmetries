"""Experiment sandbox for comparing financial process families."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from .financial_processes import (
    JumpDiffusionConfig,
    RegimeSwitchingConfig,
    simulate_jump_diffusion,
    simulate_regime_switching_process,
)


@dataclass(frozen=True)
class SandboxScenario:
    name: str
    n_steps: int = 512
    dt: float = 1 / 252


class ProcessSandbox:
    """Utility class for scripted simulation sweeps inside Python workflows."""

    def __init__(self, seed: int = 0):
        self.key = jax.random.PRNGKey(seed)

    def run_jump_diffusion(self, scenario: SandboxScenario, config: JumpDiffusionConfig | None = None) -> dict[str, Any]:
        self.key, subkey = jax.random.split(self.key)
        prices = simulate_jump_diffusion(
            subkey,
            n_steps=scenario.n_steps,
            dt=scenario.dt,
            config=config,
        )
        log_returns = jnp.diff(jnp.log(prices))
        return {
            "scenario": scenario.name,
            "process": "jump_diffusion",
            "series": prices,
            "summary": {
                "mean_return": float(jnp.mean(log_returns)),
                "volatility": float(jnp.std(log_returns)),
            },
        }

    def run_regime_switching(self, scenario: SandboxScenario, config: RegimeSwitchingConfig | None = None) -> dict[str, Any]:
        self.key, subkey = jax.random.split(self.key)
        states, series = simulate_regime_switching_process(
            subkey,
            n_steps=scenario.n_steps,
            dt=scenario.dt,
            config=config,
        )
        state_changes = jnp.sum(states[1:] != states[:-1])
        return {
            "scenario": scenario.name,
            "process": "regime_switching",
            "series": series,
            "states": states,
            "summary": {
                "state_changes": int(state_changes),
                "volatility": float(jnp.std(jnp.diff(series))),
            },
        }
