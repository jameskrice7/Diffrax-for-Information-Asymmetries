"""A single, reusable description of *how* to solve a differential equation.

Diffrax's ``diffeqsolve`` takes a dozen knobs (solver, step-size controller,
adjoint, ``saveat``, ``max_steps``, ...). Threading all of those through every
model's ``__call__`` produces unreadable signatures and silent defaults that
differ per call site. :class:`SolveConfig` bundles them into one PyTree that can
be stored, swapped and passed around.

It also fixes two defaults that bite people constantly:

* ``max_steps`` defaults to 4096 in Diffrax and raises when exceeded. Long or
  stiff solves hit this and the error is not obvious. :class:`SolveConfig`
  surfaces it as a first-class field.
* Gradients through an SDE need care. ``RecursiveCheckpointAdjoint`` (the
  Diffrax default) is right for fixed-step SDE solves;
  :meth:`SolveConfig.for_backprop_through_long_solve` swaps in
  ``BacksolveAdjoint`` for O(1)-memory training.

Ito versus Stratonovich
-----------------------
:func:`solve_sde` interprets ``drift``/``diffusion`` in the **Ito** sense, which
is the convention finance uses (``dS = mu S dt + sigma S dW`` for geometric
Brownian motion). The default solver is therefore ``diffrax.Euler``
(Euler--Maruyama), which is Ito-correct for any noise structure.

Faster solvers exist but each carries an assumption, and applying one whose
assumption is violated gives a silently wrong answer rather than an error:

===========================  =============  ==============================
Solver                       Strong order   Requires
===========================  =============  ==============================
``diffrax.Euler``            0.5            nothing (the safe default)
``diffrax.ItoMilstein``      1.0            commutative noise
``diffrax.ShARK``            1.5            *additive* noise (``g`` free of ``y``)
``diffrax.SRA1``             1.5            additive noise
``diffrax.Heun``             1.0            **Stratonovich** coefficients
===========================  =============  ==============================

Use :meth:`SolveConfig.for_additive_noise` to opt into the additive-noise fast
path, and :mod:`finax.diagnostics` to *verify* empirically that a solver
achieves the convergence order you expect on your problem.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import diffrax
import equinox as eqx
import jax.numpy as jnp

from .._typing import Array, Float, PRNGKeyArray, PyTree

__all__ = ["SolveConfig", "solve_ode", "solve_sde"]


class SolveConfig(eqx.Module):
    """Bundle of Diffrax solve options.

    Attributes
    ----------
    solver:
        Diffrax solver instance. Defaults are set per call site, since the right
        default differs for ODEs (``Tsit5``) and SDEs (``ShARK``).
    dt0:
        Initial (or fixed) step size. ``None`` asks an adaptive controller to
        pick one; that requires ``stepsize_controller`` to be adaptive.
    stepsize_controller:
        Defaults to ``ConstantStepSize()``.
    saveat:
        Where to record output. Defaults to the terminal value only.
    adjoint:
        Differentiation strategy.
    max_steps:
        Step budget. Raise it for long horizons or fine ``dt0``.
    throw:
        If ``False``, a failed solve returns NaNs and a non-zero ``result``
        instead of raising. Necessary under ``vmap`` when some batch elements
        may legitimately fail.

    Examples
    --------
    >>> import diffrax
    >>> cfg = SolveConfig(dt0=0.01, max_steps=100_000)
    >>> cfg.dt0
    0.01
    >>> dense = cfg.saving_dense()
    >>> dense.max_steps  # other fields are preserved
    100000
    """

    solver: diffrax.AbstractSolver | None = None
    dt0: float | None = 0.01
    stepsize_controller: diffrax.AbstractStepSizeController = eqx.field(
        default_factory=diffrax.ConstantStepSize
    )
    saveat: diffrax.SaveAt = eqx.field(default_factory=lambda: diffrax.SaveAt(t1=True))
    adjoint: diffrax.AbstractAdjoint = eqx.field(
        default_factory=diffrax.RecursiveCheckpointAdjoint
    )
    max_steps: int | None = eqx.field(static=True, default=4096)
    throw: bool = eqx.field(static=True, default=True)

    # -- Convenience constructors ------------------------------------------
    #
    # These use dataclasses.replace rather than eqx.tree_at because some fields
    # (max_steps, throw) are static, and tree_at only reaches PyTree leaves.

    def saving_at(self, ts: Float[Array, " time"]) -> SolveConfig:
        """Return a copy that records the solution at ``ts``."""
        return dataclasses.replace(self, saveat=diffrax.SaveAt(ts=jnp.asarray(ts)))

    def saving_dense(self) -> SolveConfig:
        """Return a copy that builds a dense (continuously-queryable) solution."""
        return dataclasses.replace(self, saveat=diffrax.SaveAt(dense=True))

    def saving_steps(self) -> SolveConfig:
        """Return a copy that records every accepted step."""
        return dataclasses.replace(self, saveat=diffrax.SaveAt(steps=True))

    def adaptive(self, *, rtol: float = 1e-3, atol: float = 1e-6) -> SolveConfig:
        """Return a copy using a PID adaptive step-size controller.

        Only appropriate for ODEs and for SDE solvers that expose an error
        estimate. Fixed-step is the norm for SDEs.
        """
        return dataclasses.replace(
            self,
            stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
            dt0=None,
        )

    def for_backprop_through_long_solve(self) -> SolveConfig:
        """Return a copy using ``BacksolveAdjoint`` for O(1)-memory gradients.

        Trades memory for compute and introduces some gradient error, but makes
        very long solves trainable. This is the "optimise-then-discretise"
        route from Chen et al. (2018).

        .. warning::
           ``BacksolveAdjoint`` is implemented as a ``custom_vjp``, so it
           **cannot differentiate with respect to values closed over by the
           vector field**. Attempting it raises
           ``jax.errors.CustomVJPException``. Any parameter you need gradients
           for must arrive either through the ``args`` argument of
           :func:`solve_ode` / :func:`solve_sde`, or as a field of an
           ``eqx.Module`` vector field -- which is how every model in
           :mod:`finax.models` is built, so they are unaffected.

           This restriction does not apply to the default
           ``RecursiveCheckpointAdjoint``.
        """
        return dataclasses.replace(self, adjoint=diffrax.BacksolveAdjoint())

    def with_steps_for(
        self, t0: float, t1: float, *, safety: float = 1.25
    ) -> SolveConfig:
        """Return a copy whose ``max_steps`` is large enough to span ``[t0, t1]``.

        Diffrax defaults ``max_steps`` to 4096 and *raises* when it is exceeded.
        With a fixed step size the requirement is exactly ``(t1 - t0) / dt0``,
        so there is no reason to guess: this computes it and adds a safety
        margin. ``max_steps`` is never reduced.

        Examples
        --------
        >>> cfg = SolveConfig(dt0=0.0005).with_steps_for(0.0, 3.15)
        >>> cfg.max_steps
        7876
        >>> SolveConfig(dt0=0.1).with_steps_for(0.0, 1.0).max_steps  # never shrinks
        4096
        """
        if self.dt0 is None:
            raise ValueError(
                "with_steps_for needs a fixed dt0; an adaptive controller sizes "
                "its own steps."
            )
        needed = int(abs(t1 - t0) / abs(self.dt0) * safety) + 1
        return dataclasses.replace(self, max_steps=max(needed, self.max_steps or 0))

    def for_additive_noise(self) -> SolveConfig:
        """Return a copy using ``diffrax.ShARK`` (strong order 1.5).

        Only valid when the diffusion does not depend on the state. Diffrax
        checks this and raises if it is violated, so the failure mode is loud.
        """
        return dataclasses.replace(self, solver=diffrax.ShARK())


def _levy_area_for(solver: diffrax.AbstractSolver):
    """Pick the cheapest Levy area a solver will accept.

    Solvers advertise their requirement via ``minimal_levy_area``. Reading it
    means users never have to match Brownian-motion configuration to solver by
    hand -- a mismatch otherwise surfaces as an opaque error deep inside the
    solver's ``init``.
    """
    minimal = getattr(solver, "minimal_levy_area", None)
    if minimal is None:
        return diffrax.BrownianIncrement
    if issubclass(minimal, diffrax.AbstractSpaceTimeTimeLevyArea):
        return diffrax.SpaceTimeTimeLevyArea
    if issubclass(minimal, diffrax.AbstractSpaceTimeLevyArea):
        return diffrax.SpaceTimeLevyArea
    return diffrax.BrownianIncrement


def _resolve(
    config: SolveConfig | None, default_solver: diffrax.AbstractSolver
) -> SolveConfig:
    config = SolveConfig() if config is None else config
    if config.solver is None:
        config = eqx.tree_at(
            lambda c: c.solver, config, default_solver, is_leaf=lambda x: x is None
        )
    return config


def solve_ode(
    vector_field,
    y0: PyTree,
    t0: float,
    t1: float,
    *,
    args: PyTree = None,
    config: SolveConfig | None = None,
) -> diffrax.Solution:
    """Solve ``dy/dt = vector_field(t, y, args)`` on ``[t0, t1]``.

    Defaults to ``Tsit5``, a good general-purpose explicit Runge-Kutta method.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> sol = solve_ode(lambda t, y, args: -y, jnp.array(1.0), 0.0, 1.0,
    ...                 config=SolveConfig(dt0=0.001))
    >>> bool(jnp.allclose(sol.ys[-1], jnp.exp(-1.0), atol=1e-4))
    True
    """
    cfg = _resolve(config, diffrax.Tsit5())
    return diffrax.diffeqsolve(
        diffrax.ODETerm(vector_field),
        cfg.solver,
        t0=t0,
        t1=t1,
        dt0=cfg.dt0,
        y0=y0,
        args=args,
        saveat=cfg.saveat,
        stepsize_controller=cfg.stepsize_controller,
        adjoint=cfg.adjoint,
        max_steps=cfg.max_steps,
        throw=cfg.throw,
    )


def solve_sde(
    drift,
    diffusion,
    y0: PyTree,
    t0: float,
    t1: float,
    *,
    key: PRNGKeyArray,
    args: PyTree = None,
    config: SolveConfig | None = None,
    noise_shape: tuple[int, ...] | None = None,
    brownian_tol: float | None = None,
    levy_area: Any | None = None,
) -> diffrax.Solution:
    """Solve the Ito SDE ``dy = drift(t,y) dt + diffusion(t,y) dW`` on ``[t0, t1]``.

    Defaults to ``diffrax.Euler`` (Euler--Maruyama), which is correct for any
    noise structure under the Ito convention. See the module docstring for the
    faster solvers and the assumptions they carry.

    Parameters
    ----------
    drift, diffusion:
        Callables ``f(t, y, args)``. ``diffusion`` returns something that can be
        contracted with a Brownian increment: a ``(state, noise)`` matrix, or a
        ``lineax`` linear operator (use ``lineax.DiagonalLinearOperator`` for
        diagonal noise -- it is O(d) rather than O(d^2)).
    key:
        PRNG key for the Brownian path.
    noise_shape:
        Shape of the driving Brownian motion. Defaults to the shape of ``y0``
        (i.e. diagonal noise). Set it explicitly when the diffusion is a matrix
        mapping ``m`` Brownian factors into ``d`` states.
    brownian_tol:
        Resolution of the ``VirtualBrownianTree``. Defaults to ``dt0 / 10``,
        keeping the Brownian path finer than the solver steps.
    levy_area:
        Levy-area type to simulate. By default it is read off the solver's
        ``minimal_levy_area``, so it always matches.

    Examples
    --------
    Zero diffusion must reproduce the ODE solution:

    >>> import jax.numpy as jnp, jax.random as jr
    >>> sol = solve_sde(lambda t, y, a: y, lambda t, y, a: jnp.zeros_like(y),
    ...                 jnp.array(1.0), 0.0, 1.0, key=jr.PRNGKey(0),
    ...                 config=SolveConfig(dt0=0.001))
    >>> bool(jnp.allclose(sol.ys[-1], jnp.e, atol=1e-2))
    True
    """
    cfg = _resolve(config, diffrax.Euler())

    if noise_shape is None:
        noise_shape = jnp.shape(y0)
    if brownian_tol is None:
        brownian_tol = (cfg.dt0 / 10.0) if cfg.dt0 is not None else 1e-3
    if levy_area is None:
        levy_area = _levy_area_for(cfg.solver)

    brownian = diffrax.VirtualBrownianTree(
        t0=t0,
        t1=t1,
        tol=brownian_tol,
        shape=noise_shape,
        key=key,
        levy_area=levy_area,
    )
    terms = diffrax.MultiTerm(
        diffrax.ODETerm(drift),
        diffrax.ControlTerm(diffusion, brownian),
    )
    return diffrax.diffeqsolve(
        terms,
        cfg.solver,
        t0=t0,
        t1=t1,
        dt0=cfg.dt0,
        y0=y0,
        args=args,
        saveat=cfg.saveat,
        stepsize_controller=cfg.stepsize_controller,
        adjoint=cfg.adjoint,
        max_steps=cfg.max_steps,
        throw=cfg.throw,
    )
