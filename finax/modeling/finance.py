"""Finance-focused differential equation models."""

from __future__ import annotations

from typing import Any

try:  # pragma: no cover - handled at runtime
    import jax.numpy as jnp  # noqa: F401
    import diffrax  # noqa: F401
except Exception:  # pragma: no cover - optional at import
    jnp = None  # type: ignore
    diffrax = None  # type: ignore

from .neural_sde import NeuralSDE
from .neural_ode import NeuralODE


def geometric_brownian_motion(mu: float, sigma: float) -> NeuralSDE:
    """Create a geometric Brownian motion model for asset prices."""

    def drift(t, y, params):
        return mu * y

    def diffusion(t, y, params):
        return sigma * y

    return NeuralSDE(drift=drift, diffusion=diffusion)


def vasicek_rate(kappa: float, theta: float, sigma: float) -> NeuralSDE:
    """Create a Vasicek interest rate model."""

    def drift(t, r, params):
        return kappa * (theta - r)

    def diffusion(t, r, params):
        return sigma

    return NeuralSDE(drift=drift, diffusion=diffusion)


def logistic_growth(a: float, b: float) -> NeuralODE:
    """Create a logistic growth model for economic output."""

    def vector_field(t, y, params):
        return a * y * (1 - y / b)

    return NeuralODE(vector_field)
