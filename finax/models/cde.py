"""Neural controlled differential equations for irregular time series.

A neural CDE evolves a hidden state ``z`` against an interpolated data path
``X``:

.. math:: z_t = z_{t_0} + \\int_{t_0}^{t} f_\\theta(z_s)\\, \\mathrm{d}X_s.

Because the integral is driven by ``dX`` rather than ``dt``, observation times
enter the model natively. That makes the neural CDE the right tool for
irregularly sampled and partially observed data -- exactly the shape of
market microstructure data, where quotes and trades arrive asynchronously.

See :mod:`finax.core.paths` for turning raw observations into the control path.
"""

from __future__ import annotations

import diffrax
import equinox as eqx
import jax

from .._typing import Array, Float, PRNGKeyArray
from ..core.paths import ControlPath
from ..core.solve import SolveConfig
from .mlp import LowRankTensorField, TensorFieldMLP

__all__ = ["NeuralCDE"]


class NeuralCDE(eqx.Module):
    """A neural CDE with initial-state encoder and terminal readout.

    Parameters
    ----------
    initial:
        Linear map from the first observation of the control path to the initial
        hidden state. Conditioning ``z_0`` on ``X_{t_0}`` (rather than using a
        fixed ``z_0``) is what lets the model see absolute levels, since the CDE
        integral itself only sees increments.
    field:
        Vector field returning a ``(hidden_size, control_size)`` matrix.
    readout:
        Linear map from the terminal hidden state to the output.
    config:
        Solve options.

    Examples
    --------
    >>> import jax.numpy as jnp, jax.random as jr
    >>> from finax.core import build_control_path, SolveConfig
    >>> ts = jnp.array([0.0, 0.7, 1.9, 3.0])
    >>> ys = jnp.array([[1.0], [jnp.nan], [3.0], [2.0]])
    >>> path = build_control_path(ts, ys)
    >>> model = NeuralCDE.from_hyperparameters(
    ...     input_size=path.n_channels, hidden_size=8, output_size=1,
    ...     key=jr.PRNGKey(0), config=SolveConfig(dt0=0.05))
    >>> model(path).shape
    (1,)

    Gradients flow through the solve to every parameter:

    >>> import equinox as eqx
    >>> g = eqx.filter_grad(lambda m: jnp.sum(m(path) ** 2))(model)
    >>> bool(jnp.any(g.readout.weight != 0))
    True
    """

    initial: eqx.nn.Linear
    field: eqx.Module
    readout: eqx.nn.Linear
    config: SolveConfig

    def __init__(self, initial, field, readout, *, config: SolveConfig | None = None):
        self.initial = initial
        self.field = field
        self.readout = readout
        self.config = config if config is not None else SolveConfig()

    @classmethod
    def from_hyperparameters(
        cls,
        *,
        input_size: int,
        hidden_size: int,
        output_size: int,
        width: int = 64,
        depth: int = 2,
        key: PRNGKeyArray,
        rank: int | None = None,
        config: SolveConfig | None = None,
    ) -> NeuralCDE:
        """Build a neural CDE with freshly-initialised components.

        Parameters
        ----------
        input_size:
            Channel count of the control path. Take it from
            :attr:`ControlPath.n_channels` rather than counting by hand -- the
            augmentation in :func:`~finax.core.paths.prepare_channels` adds
            time and mask channels.
        rank:
            If given, use a :class:`~finax.models.mlp.LowRankTensorField` of this
            rank instead of a dense field, cutting the final-layer parameter
            count from ``width * hidden * input`` to
            ``width * rank * (hidden + input)``. Recommended whenever
            ``hidden_size * input_size`` exceeds a few thousand.
        """
        key_i, key_f, key_r = jax.random.split(key, 3)
        initial = eqx.nn.Linear(input_size, hidden_size, key=key_i)
        if rank is None:
            field: eqx.Module = TensorFieldMLP(
                state_size=hidden_size,
                control_size=input_size,
                width=width,
                depth=depth,
                key=key_f,
            )
        else:
            field = LowRankTensorField(
                state_size=hidden_size,
                control_size=input_size,
                rank=rank,
                width=width,
                depth=depth,
                key=key_f,
            )
        readout = eqx.nn.Linear(hidden_size, output_size, key=key_r)
        return cls(initial, field, readout, config=config)

    def hidden_states(
        self,
        path: ControlPath,
        *,
        ts: Float[Array, " time"] | None = None,
        config: SolveConfig | None = None,
    ) -> Float[Array, "time hidden"]:
        """Return the hidden trajectory, for probing or sequence-to-sequence use.

        With ``ts=None`` this returns the single terminal hidden state, shaped
        ``(1, hidden)``.
        """
        cfg = config if config is not None else self.config
        if ts is not None:
            cfg = cfg.saving_at(ts)

        z0 = self.initial(path.evaluate(path.t0))
        term = diffrax.ControlTerm(
            lambda t, y, args: self.field(t, y, args), path.interpolation
        ).to_ode()
        sol = diffrax.diffeqsolve(
            term,
            cfg.solver if cfg.solver is not None else diffrax.Tsit5(),
            t0=path.t0,
            t1=path.t1,
            dt0=cfg.dt0,
            y0=z0,
            saveat=cfg.saveat,
            stepsize_controller=cfg.stepsize_controller,
            adjoint=cfg.adjoint,
            max_steps=cfg.max_steps,
            throw=cfg.throw,
        )
        return sol.ys

    def __call__(
        self,
        path: ControlPath,
        *,
        ts: Float[Array, " time"] | None = None,
        config: SolveConfig | None = None,
    ) -> Float[Array, "..."]:
        """Map a control path to an output.

        With ``ts=None`` returns a single ``(output_size,)`` prediction from the
        terminal state. With ``ts`` given, returns ``(len(ts), output_size)``,
        i.e. a prediction at every requested time.
        """
        zs = self.hidden_states(path, ts=ts, config=config)
        if ts is None:
            return self.readout(zs[-1])
        return jax.vmap(self.readout)(zs)
