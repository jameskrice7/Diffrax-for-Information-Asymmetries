"""Neural stochastic differential equation models."""

from __future__ import annotations

from typing import Any, Callable

try:  # pragma: no cover - optional at import time
    import jax.numpy as jnp
    import diffrax
except Exception:  # pragma: no cover
    jnp = None  # type: ignore
    diffrax = None  # type: ignore


class NeuralSDE:
    """SDE wrapper with user-provided drift and diffusion callables."""

    def __init__(self, drift: Callable[[Any, Any, Any], Any], diffusion: Callable[[Any, Any, Any], Any]):
        self.drift = drift
        self.diffusion = diffusion

    def simulate(
        self,
        y0: Any,
        t0: float,
        t1: float,
        *,
        key: Any,
        solver: Any | None = None,
        dt0: float = 0.1,
        saveat: Any | None = None,
        brownian_tol: float = 1e-3,
        **kwargs: Any,
    ) -> Any:
        """Simulate an SDE path with Diffrax."""
        if diffrax is None or jnp is None:
            raise ImportError("JAX and Diffrax are required for simulation.")

        if solver is None:  # pragma: no cover - default solver
            solver = diffrax.EulerHeun()

        brownian = diffrax.VirtualBrownianTree(
            t0=t0,
            t1=t1,
            tol=brownian_tol,
            shape=jnp.shape(y0),
            key=key,
        )
        term = diffrax.MultiTerm(
            diffrax.ODETerm(self.drift),
            diffrax.ControlTerm(self.diffusion, brownian),
        )
        solve_kwargs = dict(solver=solver, t0=t0, t1=t1, dt0=dt0, y0=y0, **kwargs)
        if saveat is not None:
            solve_kwargs["saveat"] = saveat
        return diffrax.diffeqsolve(term, **solve_kwargs)

    def plot(self, solution: Any, **kwargs: Any) -> Any:
        from ..visualization import plot_solution

        return plot_solution(solution, **kwargs)
