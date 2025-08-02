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

    def solve(
        self,
        y0: Any,
        t0: float,
        t1: float,
        *,
        solver: Any | None = None,
        dt0: float = 0.1,
        **kwargs: Any,
    ) -> Any:
        """Solve the ODE from ``t0`` to ``t1`` starting at ``y0``.

        Parameters
        ----------
        y0:
            Initial state.
        t0, t1:
            Time interval of the solve.
        solver:
            Optional Diffrax solver; defaults to ``diffrax.Tsit5``.
        dt0:
            Initial step size for the solver.
        """

        if diffrax is None:
            raise ImportError("JAX and Diffrax are required for solving ODEs.")

        if solver is None:  # pragma: no cover - default solver
            solver = diffrax.Tsit5()

        term = diffrax.ODETerm(self.vector_field)
        return diffrax.diffeqsolve(
            term,
            solver=solver,
            t0=t0,
            t1=t1,
            dt0=dt0,
            y0=y0,
            **kwargs,
        )



    def plot(self, solution: Any, **kwargs: Any) -> Any:
        """Visualize an ODE solution using Finax's plotting helpers."""
        from ..visualization import plot_solution

        return plot_solution(solution, **kwargs)


    def validate(self, observed: Any, simulated: Any, lags: int = 20):
        """Run statistical tests on residuals between observed and simulated data.

        Parameters
        ----------
        observed, simulated:
            Arrays or sequences representing the actual data and the model output.
        lags:
            Number of lags for the Ljung-Box autocorrelation test.
        """
        import numpy as np
        from ..evaluation import (
            adf_test,
            kpss_test,
            ljung_box,
            jarque_bera_test,
            ks_test,
        )

        residuals = np.asarray(observed) - np.asarray(simulated)
        return {
            "adf": adf_test(residuals),
            "kpss": kpss_test(residuals),
            "jarque_bera": jarque_bera_test(residuals),
            "ljung_box": ljung_box(residuals, lags=lags),
            "ks": ks_test(residuals),
        }

