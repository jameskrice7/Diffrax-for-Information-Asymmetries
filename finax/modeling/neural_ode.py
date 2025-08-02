"""Neural ordinary differential equation models."""

from __future__ import annotations

from typing import Callable, Any

try:  # pragma: no cover - handled at runtime
    import jax.numpy as jnp  # noqa: F401
    import diffrax  # noqa: F401
except Exception:  # pragma: no cover - the dependencies are optional at import time
    jnp = None  # type: ignore
    diffrax = None  # type: ignore


class NeuralODE:
    """Basic neural ODE wrapper.

    Parameters
    ----------
    vector_field:
        Callable representing the derivative ``dy/dt = f(t, y, params)``.
    """

    def __init__(self, vector_field: Callable[[Any, Any, Any], Any]):
        self.vector_field = vector_field

    def solve(self, y0: Any, t0: float, t1: float, **kwargs: Any) -> Any:
        """Solve the ODE from ``t0`` to ``t1`` starting at ``y0``.

        This method requires JAX and Diffrax to be installed. It is a
        lightweight placeholder for future solver configuration.
        """
        if diffrax is None:
            raise ImportError("JAX and Diffrax are required for solving ODEs.")
        return diffrax.diffeqsolve(self.vector_field, t0=t0, t1=t1, y0=y0, **kwargs)


    def plot(self, solution: Any, **kwargs: Any) -> Any:
        """Visualize an ODE solution using Finax's plotting helpers."""
        from ..visualization import plot_solution

        return plot_solution(solution, **kwargs)

