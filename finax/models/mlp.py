"""Vector-field networks.

Thin, opinionated wrappers over ``equinox.nn.MLP`` that adapt it to the
``f(t, y, args)`` signature Diffrax expects, and that default to the
initialisation choices which make neural differential equations actually train.
"""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp

from .._typing import Array, Float, PRNGKeyArray

__all__ = ["VectorFieldMLP", "TensorFieldMLP", "LowRankTensorField"]


class VectorFieldMLP(eqx.Module):
    """An MLP with a Diffrax-compatible ``f(t, y, args)`` signature.

    Parameters
    ----------
    in_size, out_size:
        State dimensions.
    width, depth:
        Hidden layer width and number of hidden layers.
    activation:
        Hidden activation. Defaults to ``jax.nn.softplus``, which is smooth --
        unlike ReLU, whose kinks make adaptive solvers reject steps and make
        higher-order solvers lose their convergence order.
    final_activation:
        Output activation. Defaults to ``jnp.tanh`` to bound the drift; see
        :meth:`NeuralODE.from_hyperparameters` for why this matters.
    include_time:
        Concatenate ``t`` onto the input, making the field non-autonomous.

    Examples
    --------
    >>> import jax.random as jr, jax.numpy as jnp
    >>> f = VectorFieldMLP(in_size=3, out_size=3, width=8, depth=1, key=jr.PRNGKey(0))
    >>> f(0.0, jnp.ones(3), None).shape
    (3,)
    """

    mlp: eqx.nn.MLP
    include_time: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        in_size: int,
        out_size: int,
        width: int = 64,
        depth: int = 2,
        key: PRNGKeyArray,
        activation: Callable | None = None,
        final_activation: Callable | None = None,
        include_time: bool = True,
    ):
        self.include_time = include_time
        self.mlp = eqx.nn.MLP(
            in_size=in_size + (1 if include_time else 0),
            out_size=out_size,
            width_size=width,
            depth=depth,
            activation=jax.nn.softplus if activation is None else activation,
            final_activation=jnp.tanh if final_activation is None else final_activation,
            key=key,
        )

    def __call__(self, t, y: Float[Array, " state"], args=None) -> Float[Array, " state"]:
        if self.include_time:
            y = jnp.concatenate([jnp.broadcast_to(jnp.asarray(t, y.dtype), (1,)), y])
        return self.mlp(y)


class TensorFieldMLP(eqx.Module):
    """A CDE vector field producing the full ``(state, control)`` matrix.

    A neural CDE integrates ``dz = f_theta(z) dX``, so ``f_theta(z)`` must be a
    matrix of shape ``(state_size, control_size)``. Producing it densely means
    the final linear layer has ``width * state_size * control_size`` weights --
    the cubic parameter growth that makes deep or wide neural CDEs impractical.

    Use this when ``state_size * control_size`` is small; otherwise prefer
    :class:`LowRankTensorField`.

    Examples
    --------
    >>> import jax.random as jr, jax.numpy as jnp
    >>> f = TensorFieldMLP(state_size=4, control_size=3, width=8, depth=1,
    ...                    key=jr.PRNGKey(0))
    >>> f(0.0, jnp.ones(4), None).shape
    (4, 3)
    """

    mlp: eqx.nn.MLP
    state_size: int = eqx.field(static=True)
    control_size: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        state_size: int,
        control_size: int,
        width: int = 64,
        depth: int = 2,
        key: PRNGKeyArray,
        activation: Callable | None = None,
    ):
        self.state_size = state_size
        self.control_size = control_size
        self.mlp = eqx.nn.MLP(
            in_size=state_size,
            out_size=state_size * control_size,
            width_size=width,
            depth=depth,
            activation=jax.nn.softplus if activation is None else activation,
            final_activation=jnp.tanh,
            key=key,
        )

    def __call__(self, t, y, args=None) -> Float[Array, "state control"]:
        return self.mlp(y).reshape(self.state_size, self.control_size)


class LowRankTensorField(eqx.Module):
    """A CDE vector field factorised as ``U(z) @ V(z).T`` to avoid cubic growth.

    The dense field of :class:`TensorFieldMLP` needs
    ``width * state_size * control_size`` final-layer parameters. Factorising

    .. math:: f_\\theta(z) = U(z) V(z)^\\top, \\quad
              U \\in \\mathbb{R}^{d \\times r},\\; V \\in \\mathbb{R}^{m \\times r}

    reduces that to ``width * r * (state_size + control_size)``. For a
    128-dimensional state, 64 control channels and ``width=128`` that is a drop
    from ~1.05M parameters to ~98k at ``r=4`` -- a 10x reduction -- and the
    resulting field is still a universal approximator of rank-``r`` dynamics.

    This directly addresses the limitation noted in the neural CDE literature
    that "the vector field must output a matrix, leading to cubic parameter
    growth in the final layer, which makes deep or stacked Neural CDE
    architectures impractical".

    Parameters
    ----------
    rank:
        Factorisation rank ``r``. ``r >= min(state_size, control_size)`` recovers
        the full expressiveness of a dense field.

    Examples
    --------
    >>> import jax.random as jr, jax.numpy as jnp
    >>> f = LowRankTensorField(state_size=128, control_size=64, rank=4,
    ...                        width=128, depth=1, key=jr.PRNGKey(0))
    >>> f(0.0, jnp.ones(128), None).shape
    (128, 64)

    The parameter saving versus a dense field of the same width:

    >>> dense = TensorFieldMLP(state_size=128, control_size=64, width=128,
    ...                        depth=1, key=jr.PRNGKey(0))
    >>> count = lambda m: sum(x.size for x in jax.tree_util.tree_leaves(
    ...     eqx.filter(m, eqx.is_inexact_array)))
    >>> count(dense), count(f)
    (1073280, 132096)
    >>> count(dense) > 8 * count(f)
    True
    """

    u_net: eqx.nn.MLP
    v_net: eqx.nn.MLP
    state_size: int = eqx.field(static=True)
    control_size: int = eqx.field(static=True)
    rank: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        state_size: int,
        control_size: int,
        rank: int = 4,
        width: int = 64,
        depth: int = 2,
        key: PRNGKeyArray,
        activation: Callable | None = None,
    ):
        if rank < 1:
            raise ValueError(f"rank must be >= 1, got {rank}.")
        self.state_size = state_size
        self.control_size = control_size
        self.rank = rank
        act = jax.nn.softplus if activation is None else activation
        key_u, key_v = jax.random.split(key)
        self.u_net = eqx.nn.MLP(
            in_size=state_size,
            out_size=state_size * rank,
            width_size=width,
            depth=depth,
            activation=act,
            final_activation=jnp.tanh,
            key=key_u,
        )
        self.v_net = eqx.nn.MLP(
            in_size=state_size,
            out_size=control_size * rank,
            width_size=width,
            depth=depth,
            activation=act,
            final_activation=jnp.tanh,
            key=key_v,
        )

    def __call__(self, t, y, args=None) -> Float[Array, "state control"]:
        u = self.u_net(y).reshape(self.state_size, self.rank)
        v = self.v_net(y).reshape(self.control_size, self.rank)
        # Scale by 1/sqrt(rank) so the output magnitude does not grow with rank.
        return (u @ v.T) / jnp.sqrt(self.rank)
