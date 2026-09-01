"""Neural stochastic differential equations."""

from __future__ import annotations

from typing import Literal

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import lineax

from .._typing import Array, Float, PRNGKeyArray
from ..core.solve import SolveConfig, solve_sde
from ..errors import ShapeError
from .mlp import VectorFieldMLP

__all__ = ["NeuralSDE", "NoiseType"]

NoiseType = Literal["diagonal", "scalar", "general"]


class _DiagonalDiffusion(eqx.Module):
    """Wraps a network so its vector-shaped output is treated as diagonal noise."""

    net: eqx.Module
    floor: float = eqx.field(static=True)

    def __call__(self, t, y, args=None):
        return jax.nn.softplus(self.net(t, y, args)) + self.floor


class _GeneralDiffusion(eqx.Module):
    """Wraps a network so its flat output is reshaped to a ``(state, noise)`` matrix."""

    net: eqx.Module
    state_size: int = eqx.field(static=True)
    noise_size: int = eqx.field(static=True)

    def __call__(self, t, y, args=None):
        return self.net(t, y, args).reshape(self.state_size, self.noise_size)


class NeuralSDE(eqx.Module):
    """A neural SDE: ``dz = f_theta(t, z) dt + g_phi(t, z) dW``.

    Being an ``eqx.Module``, the model is a PyTree: ``jax.jit``, ``jax.vmap`` and
    ``equinox.filter_grad`` all work on it directly, and ``vmap`` over PRNG keys
    is the idiomatic way to draw Monte Carlo paths.

    Parameters
    ----------
    drift, diffusion:
        Callables ``f(t, y, args)``.
    noise_type:
        ``"diagonal"``
            ``diffusion`` returns an array shaped like ``y``; each state has its
            own independent Brownian driver. The common case.
        ``"scalar"``
            ``diffusion`` returns an array shaped like ``y``, but a *single*
            Brownian motion drives every component.
        ``"general"``
            ``diffusion`` returns a ``(state_size, noise_size)`` matrix, allowing
            correlated noise across states.
    noise_size:
        Number of independent Brownian factors. Required for ``"general"``;
        ignored otherwise.
    config:
        Default solve options. Defaults to ``dt0=0.01`` with ``ShARK``, a
        strong-order-1.0 solver.

    Examples
    --------
    Geometric Brownian motion has a closed-form mean, which the solver reproduces:

    >>> import jax, jax.numpy as jnp, jax.random as jr
    >>> from finax.core import SolveConfig
    >>> mu, sigma = 0.05, 0.2
    >>> model = NeuralSDE(lambda t, y, a: mu * y, lambda t, y, a: sigma * y,
    ...                   config=SolveConfig(dt0=0.002))
    >>> keys = jr.split(jr.PRNGKey(0), 2048)
    >>> paths = jax.vmap(lambda k: model(jnp.array([1.0]), 0.0, 1.0, key=k))(keys)
    >>> bool(abs(float(jnp.mean(paths)) - jnp.exp(mu)) < 0.03)
    True

    Zero diffusion recovers the ODE solution (to Euler--Maruyama's order-1
    accuracy in the drift):

    >>> det = NeuralSDE(lambda t, y, a: y, lambda t, y, a: jnp.zeros_like(y),
    ...                 config=SolveConfig(dt0=0.001))
    >>> bool(jnp.allclose(det(jnp.array([1.0]), 0.0, 1.0, key=jr.PRNGKey(0)),
    ...                   jnp.e, atol=5e-3))
    True
    """

    drift: eqx.Module
    diffusion: eqx.Module
    config: SolveConfig
    noise_type: NoiseType = eqx.field(static=True)
    noise_size: int | None = eqx.field(static=True)

    def __init__(
        self,
        drift,
        diffusion,
        *,
        noise_type: NoiseType = "diagonal",
        noise_size: int | None = None,
        config: SolveConfig | None = None,
    ):
        if noise_type == "general" and noise_size is None:
            raise ShapeError("noise_type='general' requires noise_size to be given.")
        if noise_type not in ("diagonal", "scalar", "general"):
            raise ValueError(
                f"Unknown noise_type {noise_type!r}; "
                "expected 'diagonal', 'scalar' or 'general'."
            )
        self.drift = drift
        self.diffusion = diffusion
        self.noise_type = noise_type
        self.noise_size = noise_size
        self.config = config if config is not None else SolveConfig()

    @classmethod
    def from_hyperparameters(
        cls,
        *,
        state_size: int,
        width: int = 64,
        depth: int = 2,
        key: PRNGKeyArray,
        noise_type: NoiseType = "diagonal",
        noise_size: int | None = None,
        diffusion_floor: float = 1e-4,
        config: SolveConfig | None = None,
    ) -> NeuralSDE:
        """Build a neural SDE with freshly-initialised MLP drift and diffusion.

        The diffusion output is passed through ``softplus`` and offset by
        ``diffusion_floor``, guaranteeing strict positivity. A diffusion that can
        reach zero makes the log-likelihood singular and is a common cause of
        NaN losses when training neural SDEs.
        """
        key_f, key_g = jax.random.split(key)
        drift = VectorFieldMLP(
            in_size=state_size, out_size=state_size, width=width, depth=depth, key=key_f
        )

        if noise_type == "general":
            if noise_size is None:
                raise ShapeError("noise_type='general' requires noise_size to be given.")
            raw = VectorFieldMLP(
                in_size=state_size,
                out_size=state_size * noise_size,
                width=width,
                depth=depth,
                key=key_g,
            )
            diffusion: eqx.Module = _GeneralDiffusion(
                net=raw, state_size=state_size, noise_size=noise_size
            )
        else:
            raw = VectorFieldMLP(
                in_size=state_size,
                out_size=state_size,
                width=width,
                depth=depth,
                key=key_g,
            )
            diffusion = _DiagonalDiffusion(net=raw, floor=diffusion_floor)

        return cls(
            drift,
            diffusion,
            noise_type=noise_type,
            noise_size=noise_size,
            config=config,
        )

    def _noise_shape(self, y0) -> tuple[int, ...]:
        if self.noise_type == "scalar":
            return ()
        if self.noise_type == "general":
            return (self.noise_size,)
        return jnp.shape(y0)

    def _diffusion_term(self, y0):
        """Return the diffusion callable Diffrax should contract with ``dW``."""
        if self.noise_type == "diagonal":
            # ControlTerm contracts a matrix with dW. Materialising a dense
            # diagonal matrix would be O(d^2) in both memory and flops; a Lineax
            # DiagonalLinearOperator keeps it O(d).
            def diag(t, y, args):
                return lineax.DiagonalLinearOperator(
                    jnp.atleast_1d(self.diffusion(t, y, args))
                )

            return diag
        return self.diffusion

    def solve(
        self,
        y0: Float[Array, " state"],
        t0: float,
        t1: float,
        *,
        key: PRNGKeyArray,
        args=None,
        config: SolveConfig | None = None,
    ) -> diffrax.Solution:
        """Simulate one path and return the full Diffrax :class:`~diffrax.Solution`."""
        cfg = config if config is not None else self.config
        return solve_sde(
            self.drift,
            self._diffusion_term(y0),
            y0,
            t0,
            t1,
            key=key,
            args=args,
            config=cfg,
            noise_shape=self._noise_shape(y0),
        )

    def __call__(
        self,
        y0: Float[Array, " state"],
        t0: float,
        t1: float,
        *,
        key: PRNGKeyArray,
        ts: Float[Array, " time"] | None = None,
        args=None,
        config: SolveConfig | None = None,
    ) -> Float[Array, "..."]:
        """Simulate one path, returning the terminal state or the states at ``ts``."""
        cfg = config if config is not None else self.config
        if ts is not None:
            cfg = cfg.saving_at(ts)
        sol = self.solve(y0, t0, t1, key=key, args=args, config=cfg)
        return sol.ys if ts is not None else sol.ys[-1]

    def sample(
        self,
        y0: Float[Array, " state"],
        t0: float,
        t1: float,
        *,
        key: PRNGKeyArray,
        n_paths: int,
        ts: Float[Array, " time"] | None = None,
        args=None,
        config: SolveConfig | None = None,
    ) -> Float[Array, "path ..."]:
        """Draw ``n_paths`` Monte Carlo paths in a single vectorised solve.

        This is a ``vmap`` over PRNG keys, so all paths are simulated in parallel
        on the accelerator rather than in a Python loop.

        Examples
        --------
        >>> import jax.numpy as jnp, jax.random as jr
        >>> from finax.core import SolveConfig
        >>> m = NeuralSDE(lambda t, y, a: jnp.zeros_like(y),
        ...               lambda t, y, a: jnp.ones_like(y),
        ...               config=SolveConfig(dt0=0.01))
        >>> paths = m.sample(jnp.zeros(1), 0.0, 1.0, key=jr.PRNGKey(0), n_paths=4096)
        >>> paths.shape
        (4096, 1)

        Standard Brownian motion at ``t=1`` has unit variance:

        >>> bool(abs(float(jnp.std(paths)) - 1.0) < 0.05)
        True
        """
        keys = jax.random.split(key, n_paths)
        return jax.vmap(
            lambda k: self(y0, t0, t1, key=k, ts=ts, args=args, config=config)
        )(keys)
