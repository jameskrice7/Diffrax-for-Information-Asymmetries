"""Neural jump-diffusion SDEs.

Prices move continuously most of the time and discontinuously at news events.
A pure diffusion cannot produce the heavy tails that result; a jump term can.

The simulation here uses a fixed grid with a compensated compound-Poisson jump
channel. On each step of length ``dt`` the number of jumps is drawn
``Poisson(lambda * dt)`` and their aggregate size is drawn from the learned jump
distribution. This is the standard Euler scheme for jump-diffusions and is
convergent as ``dt -> 0``; it is used in preference to exact jump-time
simulation because a fixed grid is what makes the whole thing ``jit``-able,
``vmap``-able and differentiable.

Note that the jump *counts* are discrete and therefore not reparameterisable:
gradients flow to the drift, diffusion and jump-size parameters, but not to the
jump intensity through the counting process. Use a score-function estimator, or
:meth:`NeuralJumpSDE.expected_jump_compensator`, if you need intensity gradients.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from .._typing import Array, Float, PRNGKeyArray
from .mlp import VectorFieldMLP

__all__ = ["NeuralJumpSDE"]


class NeuralJumpSDE(eqx.Module):
    """``dz = f(t,z) dt + g(t,z) dW + h(t,z) dN`` with ``N`` a Poisson process.

    Parameters
    ----------
    drift, diffusion, jump:
        Callables ``f(t, y, args)``. ``jump`` returns the size of a single jump
        given the pre-jump state.
    log_intensity:
        Learnable log jump intensity ``log(lambda)``, in jumps per unit time.
        Parameterised in logs so the intensity stays positive under
        unconstrained optimisation.
    compensate:
        Subtract ``lambda * E[jump] * dt`` from the drift so the jump channel is
        a martingale. This is what you want when the jump term should add tail
        risk without shifting the expected return -- the standard convention in
        Merton-style models.

    Examples
    --------
    >>> import jax, jax.numpy as jnp, jax.random as jr
    >>> m = NeuralJumpSDE(
    ...     drift=lambda t, y, a: jnp.zeros_like(y),
    ...     diffusion=lambda t, y, a: 0.1 * jnp.ones_like(y),
    ...     jump=lambda t, y, a: -0.05 * jnp.ones_like(y),
    ...     log_intensity=jnp.log(jnp.array(5.0)))
    >>> paths = m.sample(jnp.zeros(1), n_steps=200, dt=0.005,
    ...                  key=jr.PRNGKey(0), n_paths=2048)
    >>> paths.shape
    (2048, 201, 1)

    With compensation on, the terminal mean stays near zero even though every
    jump is negative:

    >>> bool(abs(float(jnp.mean(paths[:, -1]))) < 0.02)
    True

    Without compensation the jumps drag the mean down to ``lambda * size * T``:

    >>> import dataclasses
    >>> m2 = dataclasses.replace(m, compensate=False)
    >>> raw = m2.sample(jnp.zeros(1), n_steps=200, dt=0.005,
    ...                 key=jr.PRNGKey(0), n_paths=2048)
    >>> bool(abs(float(jnp.mean(raw[:, -1])) - (5.0 * -0.05 * 1.0)) < 0.03)
    True
    """

    drift: eqx.Module
    diffusion: eqx.Module
    jump: eqx.Module
    log_intensity: Float[Array, ""]
    compensate: bool = eqx.field(static=True, default=True)

    @classmethod
    def from_hyperparameters(
        cls,
        *,
        state_size: int,
        width: int = 64,
        depth: int = 2,
        key: PRNGKeyArray,
        intensity: float = 1.0,
        compensate: bool = True,
    ) -> NeuralJumpSDE:
        """Build a neural jump SDE with freshly-initialised networks."""
        k_f, k_g, k_h = jax.random.split(key, 3)
        return cls(
            drift=VectorFieldMLP(
                in_size=state_size, out_size=state_size, width=width, depth=depth, key=k_f
            ),
            diffusion=VectorFieldMLP(
                in_size=state_size, out_size=state_size, width=width, depth=depth, key=k_g
            ),
            jump=VectorFieldMLP(
                in_size=state_size, out_size=state_size, width=width, depth=depth, key=k_h
            ),
            log_intensity=jnp.log(jnp.asarray(intensity, jnp.float32)),
            compensate=compensate,
        )

    @property
    def intensity(self) -> Float[Array, ""]:
        """Jump intensity ``lambda``, in expected jumps per unit time."""
        return jnp.exp(self.log_intensity)

    def expected_jump_compensator(self, t, y) -> Float[Array, " state"]:
        """``lambda * h(t, y)``: the drift correction that makes jumps a martingale.

        Exposed separately because it is differentiable in the intensity, unlike
        the sampled counting process.
        """
        return self.intensity * self.jump(t, y, None)

    def step(
        self,
        y: Float[Array, " state"],
        t: float,
        dt: float,
        *,
        key: PRNGKeyArray,
    ) -> Float[Array, " state"]:
        """Advance one Euler step of length ``dt``."""
        key_w, key_n = jax.random.split(key)
        f = self.drift(t, y, None)
        g = self.diffusion(t, y, None)
        h = self.jump(t, y, None)

        if self.compensate:
            f = f - self.intensity * h

        dw = jnp.sqrt(dt) * jax.random.normal(key_w, y.shape, y.dtype)
        n_jumps = jax.random.poisson(key_n, self.intensity * dt, y.shape).astype(y.dtype)
        return y + f * dt + g * dw + h * n_jumps

    def simulate(
        self,
        y0: Float[Array, " state"],
        *,
        n_steps: int,
        dt: float,
        key: PRNGKeyArray,
        t0: float = 0.0,
    ) -> Float[Array, "time state"]:
        """Simulate one path on a fixed grid.

        Returns ``n_steps + 1`` states, including ``y0``.
        """
        keys = jax.random.split(key, n_steps)
        ts = t0 + dt * jnp.arange(n_steps)

        def body(y, inputs):
            t, k = inputs
            y_next = self.step(y, t, dt, key=k)
            return y_next, y_next

        _, ys = jax.lax.scan(body, y0, (ts, keys))
        return jnp.concatenate([y0[None, ...], ys], axis=0)

    def sample(
        self,
        y0: Float[Array, " state"],
        *,
        n_steps: int,
        dt: float,
        key: PRNGKeyArray,
        n_paths: int,
        t0: float = 0.0,
    ) -> Float[Array, "path time state"]:
        """Simulate ``n_paths`` independent paths in one vectorised call."""
        keys = jax.random.split(key, n_paths)
        return jax.vmap(
            lambda k: self.simulate(y0, n_steps=n_steps, dt=dt, key=k, t0=t0)
        )(keys)
