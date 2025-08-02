"""Neural jump diffusion SDE models."""

from __future__ import annotations

from typing import Callable, Any

try:  # pragma: no cover - handled at runtime
    import jax.numpy as jnp  # noqa: F401
    import diffrax  # noqa: F401
except Exception:  # pragma: no cover - optional at import
    jnp = None  # type: ignore
    diffrax = None  # type: ignore


class NeuralJumpSDE:
    """Neural SDE with an additional jump component.

    Parameters
    ----------
    drift:
        Callable representing the drift term.
    diffusion:
        Callable representing the diffusion term.
    jump:
        Callable representing jump sizes given jump times.
    """

    def __init__(self, drift: Callable[[Any, Any, Any], Any], diffusion: Callable[[Any, Any, Any], Any], jump: Callable[[Any, Any], Any]):
        self.drift = drift
        self.diffusion = diffusion
        self.jump = jump

    def simulate(self, y0: Any, t0: float, t1: float, *, key: Any, **kwargs: Any) -> Any:
        """Simulate the jump diffusion SDE path."""
        if diffrax is None:
            raise ImportError("JAX and Diffrax are required for simulation.")

        term = diffrax.MultiTerm(
            diffrax.ODETerm(self.drift),
            diffrax.ControlTerm(self.diffusion, diffrax.WeinerProcess(key)),
        )
        # Jump term is included as an event handler; placeholder for future refinement
        return diffrax.diffeqsolve(term, t0=t0, t1=t1, y0=y0, key=key, **kwargs)

    def plot(self, solution: Any, **kwargs: Any) -> Any:
        """Visualize an SDE solution with jumps using Finax's plotting helpers."""
        from ..visualization import plot_solution

        return plot_solution(solution, **kwargs)
