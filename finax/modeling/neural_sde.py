"""Neural stochastic differential equation models."""

from __future__ import annotations

from typing import Callable, Any

try:  # pragma: no cover - handled at runtime
    import jax.numpy as jnp  # noqa: F401
    import diffrax  # noqa: F401
except Exception:  # pragma: no cover - the dependencies are optional at import time
    jnp = None  # type: ignore
    diffrax = None  # type: ignore


class NeuralSDE:
    """Basic neural SDE wrapper with drift and diffusion terms."""

    def __init__(self, drift: Callable[[Any, Any, Any], Any], diffusion: Callable[[Any, Any, Any], Any]):
        self.drift = drift
        self.diffusion = diffusion

    def simulate(self, y0: Any, t0: float, t1: float, *, key: Any, **kwargs: Any) -> Any:
        """Simulate the SDE path.

        Parameters
        ----------
        y0: initial state
        t0, t1: time interval
        key: random key for stochastic integration
        """
        if diffrax is None:
            raise ImportError("JAX and Diffrax are required for simulation.")
        return diffrax.diffeqsolve(
            self.drift,
            t0=t0,
            t1=t1,
            y0=y0,
            args=(self.diffusion,),
            key=key,
            **kwargs,
        )

    def plot(self, solution: Any, **kwargs: Any) -> Any:
        """Visualize an SDE solution using Finax's plotting helpers."""
        from ..visualization import plot_solution

        return plot_solution(solution, **kwargs)

    def validate(self, observed: Any, simulated: Any, lags: int = 20):
        """Run statistical tests on residuals between observed and simulated paths.

        Parameters
        ----------
        observed, simulated:
            Arrays or sequences for the actual data and model output.
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

