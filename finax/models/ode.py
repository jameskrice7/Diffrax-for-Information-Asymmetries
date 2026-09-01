"""Neural ordinary differential equations."""

from __future__ import annotations

import diffrax
import equinox as eqx

from .._typing import Array, Float, PRNGKeyArray
from ..core.solve import SolveConfig, solve_ode
from .mlp import VectorFieldMLP

__all__ = ["NeuralODE"]


class NeuralODE(eqx.Module):
    """A neural ODE: ``dz/dt = f_theta(t, z)``.

    The vector field is any ``eqx.Module`` (or plain callable) with signature
    ``f(t, y, args) -> dy/dt``. Because :class:`NeuralODE` is itself an
    ``eqx.Module``, the whole model -- parameters included -- is a PyTree, so
    ``jax.jit``, ``jax.vmap`` and ``jax.grad`` apply to it directly.

    Parameters
    ----------
    vector_field:
        The drift function. Pass your own, or use :meth:`from_hyperparameters`
        to build a default MLP.
    config:
        Default :class:`~finax.core.solve.SolveConfig`. Overridable per call.

    Examples
    --------
    Wrapping an explicit vector field:

    >>> import jax.numpy as jnp
    >>> from finax.core import SolveConfig
    >>> model = NeuralODE(lambda t, y, args: -0.5 * y,
    ...                   config=SolveConfig(dt0=0.001))
    >>> y1 = model(jnp.array([1.0]), 0.0, 1.0)
    >>> bool(jnp.allclose(y1, jnp.exp(-0.5), atol=1e-4))
    True

    A learnable field, and gradients straight through the solve:

    >>> import jax, jax.random as jr
    >>> model = NeuralODE.from_hyperparameters(
    ...     state_size=2, width=16, depth=2, key=jr.PRNGKey(0),
    ...     config=SolveConfig(dt0=0.05))
    >>> loss = lambda m: jnp.sum(m(jnp.ones(2), 0.0, 1.0) ** 2)
    >>> grads = eqx.filter_grad(loss)(model)
    >>> bool(jnp.any(grads.vector_field.mlp.layers[0].weight != 0))
    True
    """

    vector_field: eqx.Module
    config: SolveConfig

    def __init__(self, vector_field, *, config: SolveConfig | None = None):
        self.vector_field = vector_field
        self.config = config if config is not None else SolveConfig()

    @classmethod
    def from_hyperparameters(
        cls,
        *,
        state_size: int,
        width: int = 64,
        depth: int = 2,
        key: PRNGKeyArray,
        activation=None,
        final_activation=None,
        config: SolveConfig | None = None,
    ) -> NeuralODE:
        """Build a :class:`NeuralODE` with a freshly-initialised MLP vector field.

        The final layer uses ``tanh`` by default, which bounds the drift and is
        the single most effective guard against the "stiffness blow-up" that
        makes untrained neural ODEs take enormous numbers of solver steps.
        """
        field = VectorFieldMLP(
            in_size=state_size,
            out_size=state_size,
            width=width,
            depth=depth,
            key=key,
            activation=activation,
            final_activation=final_activation,
        )
        return cls(field, config=config)

    def __call__(
        self,
        y0: Float[Array, " state"],
        t0: float,
        t1: float,
        *,
        ts: Float[Array, " time"] | None = None,
        args=None,
        config: SolveConfig | None = None,
    ) -> Float[Array, "..."]:
        """Integrate from ``t0`` to ``t1``.

        Returns the terminal state, or the states at ``ts`` if given.
        """
        cfg = config if config is not None else self.config
        if ts is not None:
            cfg = cfg.saving_at(ts)
        sol = solve_ode(self.vector_field, y0, t0, t1, args=args, config=cfg)
        return sol.ys if ts is not None else sol.ys[-1]

    def solve(
        self,
        y0,
        t0: float,
        t1: float,
        *,
        args=None,
        config: SolveConfig | None = None,
    ) -> diffrax.Solution:
        """Integrate and return the full Diffrax :class:`~diffrax.Solution`.

        Use this when you need ``sol.ts``, ``sol.stats`` or the dense
        interpolation rather than just the state array.
        """
        cfg = config if config is not None else self.config
        return solve_ode(self.vector_field, y0, t0, t1, args=args, config=cfg)
