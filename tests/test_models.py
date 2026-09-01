"""Tests for the neural differential equation models.

The recurring theme: verify against a closed-form solution where one exists, and
verify the JAX transformation properties (jit/vmap/grad) that motivated making
every model an ``equinox.Module``.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from finax.core import SolveConfig, build_control_path
from finax.models import (
    LatentSDE,
    LowRankTensorField,
    NeuralCDE,
    NeuralJumpSDE,
    NeuralODE,
    NeuralSDE,
    TensorFieldMLP,
)


class TestNeuralODE:
    def test_matches_analytic_exponential(self):
        model = NeuralODE(lambda t, y, a: -0.5 * y, config=SolveConfig(dt0=0.001))
        result = model(jnp.array([1.0]), 0.0, 1.0)
        assert bool(jnp.allclose(result, jnp.exp(-0.5), atol=1e-4))

    def test_matches_analytic_harmonic_oscillator(self):
        """A system with a known periodic solution catches sign and coupling errors."""

        def field(t, y, args):
            return jnp.array([y[1], -y[0]])

        config = SolveConfig(dt0=0.0005).with_steps_for(0.0, float(jnp.pi))
        model = NeuralODE(field, config=config)
        result = model(jnp.array([1.0, 0.0]), 0.0, jnp.pi)
        # After half a period the state is negated.
        assert bool(jnp.allclose(result, jnp.array([-1.0, 0.0]), atol=1e-3))

    def test_saves_at_requested_times(self):
        model = NeuralODE(lambda t, y, a: -y, config=SolveConfig(dt0=0.001))
        ts = jnp.linspace(0.0, 1.0, 11)
        out = model(jnp.array([1.0]), 0.0, 1.0, ts=ts)
        assert out.shape == (11, 1)
        assert bool(jnp.allclose(out[:, 0], jnp.exp(-ts), atol=1e-3))

    def test_is_a_pytree(self, key):
        model = NeuralODE.from_hyperparameters(state_size=2, width=8, depth=1, key=key)
        leaves = jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_inexact_array))
        assert len(leaves) > 0

    def test_gradients_flow_to_parameters(self, key):
        model = NeuralODE.from_hyperparameters(
            state_size=2, width=8, depth=1, key=key, config=SolveConfig(dt0=0.05)
        )
        grads = eqx.filter_grad(lambda m: jnp.sum(m(jnp.ones(2), 0.0, 1.0) ** 2))(model)
        flat = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_inexact_array))
        assert any(bool(jnp.any(g != 0)) for g in flat)

    def test_vmaps_over_initial_conditions(self, key):
        model = NeuralODE(lambda t, y, a: -y, config=SolveConfig(dt0=0.01))
        y0s = jnp.arange(1.0, 6.0)[:, None]
        out = jax.vmap(lambda y: model(y, 0.0, 1.0))(y0s)
        assert out.shape == (5, 1)
        assert bool(jnp.allclose(out[:, 0], y0s[:, 0] * jnp.exp(-1.0), atol=1e-2))


class TestNeuralSDE:
    def test_zero_diffusion_reduces_to_ode(self, key):
        model = NeuralSDE(
            lambda t, y, a: y,
            lambda t, y, a: jnp.zeros_like(y),
            config=SolveConfig(dt0=0.0005),
        )
        result = model(jnp.array([1.0]), 0.0, 1.0, key=key)
        assert bool(jnp.allclose(result, jnp.e, atol=5e-3))

    def test_gbm_mean_matches_analytic(self, key):
        mu, sigma = 0.05, 0.2
        model = NeuralSDE(
            lambda t, y, a: mu * y,
            lambda t, y, a: sigma * y,
            config=SolveConfig(dt0=0.002),
        )
        paths = model.sample(jnp.array([1.0]), 0.0, 1.0, key=key, n_paths=8192)
        assert abs(float(jnp.mean(paths)) - float(jnp.exp(mu))) < 0.02

    def test_brownian_motion_variance_matches_analytic(self, key):
        model = NeuralSDE(
            lambda t, y, a: jnp.zeros_like(y),
            lambda t, y, a: jnp.ones_like(y),
            config=SolveConfig(dt0=0.005),
        )
        paths = model.sample(jnp.zeros(1), 0.0, 2.0, key=key, n_paths=8192)
        # Var(W_2) = 2.
        assert abs(float(jnp.var(paths)) - 2.0) < 0.1

    def test_sample_is_deterministic_given_key(self, key):
        model = NeuralSDE(
            lambda t, y, a: 0.1 * y,
            lambda t, y, a: 0.2 * y,
            config=SolveConfig(dt0=0.01),
        )
        a = model.sample(jnp.ones(1), 0.0, 1.0, key=key, n_paths=16)
        b = model.sample(jnp.ones(1), 0.0, 1.0, key=key, n_paths=16)
        assert bool(jnp.array_equal(a, b))

    def test_different_keys_give_different_paths(self):
        model = NeuralSDE(
            lambda t, y, a: jnp.zeros_like(y),
            lambda t, y, a: jnp.ones_like(y),
            config=SolveConfig(dt0=0.01),
        )
        a = model(jnp.zeros(1), 0.0, 1.0, key=jr.PRNGKey(0))
        b = model(jnp.zeros(1), 0.0, 1.0, key=jr.PRNGKey(1))
        assert not bool(jnp.allclose(a, b))

    def test_general_noise_shape(self, key):
        """A (state, noise) diffusion matrix must produce correlated states."""
        model = NeuralSDE(
            lambda t, y, a: jnp.zeros_like(y),
            lambda t, y, a: jnp.array([[1.0, 0.0], [0.9, 0.436]]),
            noise_type="general",
            noise_size=2,
            config=SolveConfig(dt0=0.01),
        )
        paths = model.sample(jnp.zeros(2), 0.0, 1.0, key=key, n_paths=4096)
        assert paths.shape == (4096, 2)
        corr = jnp.corrcoef(paths.T)[0, 1]
        assert abs(float(corr) - 0.9) < 0.05

    def test_diffusion_floor_keeps_diffusion_positive(self, key):
        model = NeuralSDE.from_hyperparameters(
            state_size=3, width=8, depth=1, key=key, diffusion_floor=1e-3
        )
        value = model.diffusion(0.0, jnp.zeros(3), None)
        assert bool(jnp.all(value >= 1e-3))

    def test_gradients_flow_through_solve(self, key):
        model = NeuralSDE.from_hyperparameters(
            state_size=2, width=8, depth=1, key=key, config=SolveConfig(dt0=0.05)
        )

        def loss(m):
            return jnp.sum(m(jnp.ones(2), 0.0, 1.0, key=jr.PRNGKey(1)) ** 2)

        grads = eqx.filter_grad(loss)(model)
        flat = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_inexact_array))
        assert any(bool(jnp.any(jnp.isfinite(g) & (g != 0))) for g in flat)

    def test_rejects_general_noise_without_size(self):
        from finax.errors import ShapeError

        with pytest.raises(ShapeError):
            NeuralSDE(lambda t, y, a: y, lambda t, y, a: y, noise_type="general")

    def test_rejects_unknown_noise_type(self):
        with pytest.raises(ValueError, match="Unknown noise_type"):
            NeuralSDE(lambda t, y, a: y, lambda t, y, a: y, noise_type="banana")


class TestNeuralCDE:
    def test_output_shape(self, key, irregular_series):
        ts, ys = irregular_series
        path = build_control_path(ts, ys)
        model = NeuralCDE.from_hyperparameters(
            input_size=path.n_channels,
            hidden_size=8,
            output_size=3,
            key=key,
            config=SolveConfig(dt0=0.05),
        )
        assert model(path).shape == (3,)

    def test_sequence_output(self, key, irregular_series):
        ts, ys = irregular_series
        path = build_control_path(ts, ys)
        model = NeuralCDE.from_hyperparameters(
            input_size=path.n_channels,
            hidden_size=6,
            output_size=2,
            key=key,
            config=SolveConfig(dt0=0.05),
        )
        query = jnp.linspace(float(path.t0), float(path.t1), 7)
        assert model(path, ts=query).shape == (7, 2)

    def test_gradients_flow(self, key, irregular_series):
        ts, ys = irregular_series
        path = build_control_path(ts, ys)
        model = NeuralCDE.from_hyperparameters(
            input_size=path.n_channels,
            hidden_size=6,
            output_size=1,
            key=key,
            config=SolveConfig(dt0=0.05),
        )
        grads = eqx.filter_grad(lambda m: jnp.sum(m(path) ** 2))(model)
        assert bool(jnp.any(grads.readout.weight != 0))

    def test_vmaps_over_a_batch_of_paths(self, key):
        ts = jnp.stack([jnp.linspace(0.0, 1.0, 6)] * 4)
        ys = jr.normal(key, (4, 6, 2))
        paths = jax.vmap(build_control_path)(ts, ys)
        model = NeuralCDE.from_hyperparameters(
            input_size=5,  # 1 time + 2 value + 2 mask
            hidden_size=6,
            output_size=1,
            key=key,
            config=SolveConfig(dt0=0.05),
        )
        out = jax.vmap(model)(paths)
        assert out.shape == (4, 1)

    def test_low_rank_field_cuts_parameters(self, key):
        dense = TensorFieldMLP(state_size=64, control_size=32, width=64, depth=1, key=key)
        low_rank = LowRankTensorField(
            state_size=64, control_size=32, rank=2, width=64, depth=1, key=key
        )

        def count(module):
            return sum(
                x.size
                for x in jax.tree_util.tree_leaves(
                    eqx.filter(module, eqx.is_inexact_array)
                )
            )

        assert count(low_rank) < count(dense) / 4
        assert low_rank(0.0, jnp.ones(64), None).shape == (64, 32)

    def test_low_rank_field_rejects_bad_rank(self, key):
        with pytest.raises(ValueError, match="rank must be >= 1"):
            LowRankTensorField(state_size=4, control_size=2, rank=0, key=key)


class TestLatentSDE:
    def test_shapes_and_non_negative_kl(self, key):
        ts = jnp.linspace(0.0, 1.0, 10)
        ys = jnp.sin(3 * ts)[:, None]
        path = build_control_path(ts, ys)
        model = LatentSDE.from_hyperparameters(
            input_size=path.n_channels,
            latent_size=4,
            output_size=1,
            context_size=4,
            width=16,
            key=key,
            config=SolveConfig(dt0=0.02),
        )
        out = model(path, ts=ts, key=jr.PRNGKey(1))
        assert out.outputs.shape == (10, 1)
        assert out.latents.shape == (10, 4)
        # KL is an integral of a squared quantity, so it cannot be negative.
        assert float(out.kl) >= 0.0

    def test_kl_gradients_reach_both_drifts(self, key):
        ts = jnp.linspace(0.0, 1.0, 8)
        path = build_control_path(ts, jnp.cos(ts)[:, None])
        model = LatentSDE.from_hyperparameters(
            input_size=path.n_channels,
            latent_size=3,
            output_size=1,
            context_size=3,
            width=8,
            key=key,
            config=SolveConfig(dt0=0.05),
        )
        grads = eqx.filter_grad(lambda m: m(path, ts=ts, key=jr.PRNGKey(2)).kl)(model)
        assert bool(jnp.any(grads.prior_drift.mlp.layers[0].weight != 0))
        assert bool(jnp.any(grads.posterior_drift.mlp.layers[0].weight != 0))

    def test_prior_sampling_runs(self, key):
        ts = jnp.linspace(0.0, 1.0, 6)
        path = build_control_path(ts, jnp.ones((6, 1)))
        model = LatentSDE.from_hyperparameters(
            input_size=path.n_channels,
            latent_size=3,
            output_size=2,
            context_size=3,
            width=8,
            key=key,
            config=SolveConfig(dt0=0.05),
        )
        out = model.sample_prior(jnp.zeros(3), 0.0, 1.0, key=jr.PRNGKey(3), ts=ts)
        assert out.shape == (6, 2)


class TestNeuralJumpSDE:
    def test_compensated_jumps_preserve_the_mean(self, key):
        model = NeuralJumpSDE(
            drift=lambda t, y, a: jnp.zeros_like(y),
            diffusion=lambda t, y, a: 0.1 * jnp.ones_like(y),
            jump=lambda t, y, a: -0.05 * jnp.ones_like(y),
            log_intensity=jnp.log(jnp.array(5.0)),
            compensate=True,
        )
        paths = model.sample(jnp.zeros(1), n_steps=200, dt=0.005, key=key, n_paths=4096)
        assert abs(float(jnp.mean(paths[:, -1]))) < 0.02

    def test_uncompensated_jumps_shift_the_mean(self, key):
        import dataclasses

        model = NeuralJumpSDE(
            drift=lambda t, y, a: jnp.zeros_like(y),
            diffusion=lambda t, y, a: 0.1 * jnp.ones_like(y),
            jump=lambda t, y, a: -0.05 * jnp.ones_like(y),
            log_intensity=jnp.log(jnp.array(5.0)),
            compensate=True,
        )
        raw = dataclasses.replace(model, compensate=False)
        paths = raw.sample(jnp.zeros(1), n_steps=200, dt=0.005, key=key, n_paths=4096)
        # lambda * jump_size * T = 5 * -0.05 * 1.
        assert abs(float(jnp.mean(paths[:, -1])) + 0.25) < 0.03

    def test_jumps_produce_excess_kurtosis(self, key):
        """The whole point of a jump term is fat tails."""
        model = NeuralJumpSDE(
            drift=lambda t, y, a: jnp.zeros_like(y),
            diffusion=lambda t, y, a: 0.05 * jnp.ones_like(y),
            jump=lambda t, y, a: 0.5 * jnp.ones_like(y),
            log_intensity=jnp.log(jnp.array(1.0)),
            compensate=True,
        )
        terminal = model.sample(
            jnp.zeros(1), n_steps=100, dt=0.01, key=key, n_paths=8192
        )[:, -1, 0]
        z = (terminal - jnp.mean(terminal)) / jnp.std(terminal)
        assert float(jnp.mean(z**4)) > 3.5

    def test_output_shape(self, key):
        model = NeuralJumpSDE.from_hyperparameters(state_size=3, width=8, key=key)
        paths = model.sample(jnp.zeros(3), n_steps=20, dt=0.01, key=key, n_paths=7)
        assert paths.shape == (7, 21, 3)
