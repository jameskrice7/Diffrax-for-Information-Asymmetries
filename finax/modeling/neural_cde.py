"""Neural controlled differential equation models."""

from __future__ import annotations

from typing import Callable, Any

try:  # pragma: no cover - handled at runtime
    import jax.numpy as jnp  # noqa: F401
    import diffrax  # noqa: F401
except Exception:  # pragma: no cover - optional at import
    jnp = None  # type: ignore
    diffrax = None  # type: ignore


class NeuralCDE:
    """Basic neural CDE wrapper.

    Parameters
    ----------
    vector_field:
        Callable ``f(t, y, u, params)`` representing the derivative driven by a control ``u``.
    control:
        Callable ``u(t)`` that produces the control signal.
    """

    def __init__(self, vector_field: Callable[[Any, Any, Any, Any], Any], control: Callable[[Any], Any]):
        self.vector_field = vector_field
        self.control = control

    def solve(self, y0: Any, t0: float, t1: float, **kwargs: Any) -> Any:
        """Solve the controlled differential equation."""
        if diffrax is None:
            raise ImportError("JAX and Diffrax are required for solving CDEs.")

        def vf(t, y, params):
            return self.vector_field(t, y, self.control(t), params)

        return diffrax.diffeqsolve(vf, t0=t0, t1=t1, y0=y0, **kwargs)

    def plot(self, solution: Any, **kwargs: Any) -> Any:
        """Visualize a CDE solution using Finax's plotting helpers."""
        from ..visualization import plot_solution

        return plot_solution(solution, **kwargs)
