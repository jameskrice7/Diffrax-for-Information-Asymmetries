import pathlib
import sys

import diffrax
import jax.numpy as jnp
import jax.random as jr

from finax.modeling.neural_ode import NeuralODE
from finax.modeling.neural_sde import NeuralSDE


def test_neural_ode_exponential():
    def vector_field(t, y, args):
        return y

    model = NeuralODE(vector_field)
    sol = model.solve(y0=jnp.array(1.0), t0=0.0, t1=1.0, dt0=0.1)
    assert jnp.allclose(sol.ys[-1], jnp.exp(1.0), atol=1e-2)


def test_neural_sde_deterministic():
    def drift(t, y, args):
        return y

    def diffusion(t, y, args):
        return 0.0

    model = NeuralSDE(drift, diffusion)
    key = jr.PRNGKey(0)
    sol = model.simulate(
        y0=jnp.array(1.0),
        t0=0.0,
        t1=1.0,
        dt0=0.01,
        key=key,
    )
    assert jnp.allclose(sol.ys[-1], jnp.exp(1.0), atol=2e-2)
