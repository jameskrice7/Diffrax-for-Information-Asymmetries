"""Tests for training, losses and calibration."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from finax.errors import DataValidationError
from finax.inference import (
    dataloader,
    elbo,
    fit,
    fit_gbm,
    fit_mle,
    fit_ou,
    gaussian_nll,
    mae,
    mse,
    quantile_loss,
)
from finax.processes import GeometricBrownianMotion, OrnsteinUhlenbeck


class TestLosses:
    def test_mse_ignores_nan_targets(self):
        preds = jnp.array([1.0, 2.0, 3.0])
        targets = jnp.array([1.0, jnp.nan, 5.0])
        # Only the first and third contribute: (0 + 4) / 2.
        assert abs(float(mse(preds, targets)) - 2.0) < 1e-6

    def test_mse_gradient_is_finite_with_nan_targets(self):
        """A NaN target must not poison the gradient."""
        targets = jnp.array([1.0, jnp.nan, 5.0])
        grad = jax.grad(lambda p: mse(p, targets))(jnp.array([1.0, 2.0, 3.0]))
        assert bool(jnp.all(jnp.isfinite(grad)))

    def test_mae_ignores_nan_targets(self):
        preds = jnp.array([1.0, 10.0, 3.0])
        targets = jnp.array([2.0, jnp.nan, 5.0])
        assert abs(float(mae(preds, targets)) - 1.5) < 1e-6

    def test_gaussian_nll_is_minimised_at_the_true_scale(self):
        key = jr.PRNGKey(0)
        true_scale = 0.5
        targets = true_scale * jr.normal(key, (20_000,))
        preds = jnp.zeros_like(targets)

        at_truth = gaussian_nll(preds, targets, jnp.log(true_scale))
        for wrong in (0.2, 1.5):
            assert float(at_truth) < float(gaussian_nll(preds, targets, jnp.log(wrong)))

    def test_elbo_weights_the_kl(self):
        assert float(elbo(jnp.array(2.0), jnp.array(4.0), beta=0.25)) == 3.0

    def test_quantile_loss_is_asymmetric(self):
        """Underpredicting the 90th percentile costs 9x overpredicting it."""
        q = jnp.array([0.9])
        under = quantile_loss(jnp.array([[0.0]]), jnp.array([1.0]), q)
        over = quantile_loss(jnp.array([[2.0]]), jnp.array([1.0]), q)
        assert abs(float(under) / float(over) - 9.0) < 1e-4

    def test_quantile_loss_is_zero_when_exact(self):
        q = jnp.array([0.1, 0.5, 0.9])
        preds = jnp.array([[1.0, 1.0, 1.0]])
        assert float(quantile_loss(preds, jnp.array([1.0]), q)) == 0.0


class TestFit:
    def test_recovers_known_linear_relationship(self):
        key = jr.PRNGKey(0)
        x = jr.normal(key, (512, 3))
        true_w = jnp.array([2.0, -3.0, 0.5])
        y = (x @ true_w)[:, None]

        model = eqx.nn.Linear(3, 1, key=jr.PRNGKey(1))
        loss = lambda m, xb, yb: jnp.mean((jax.vmap(m)(xb) - yb) ** 2)  # noqa: E731
        result = fit(model, loss, (x, y), steps=3000, learning_rate=0.05)

        assert float(result.train_losses[-1]) < 1e-4
        assert bool(jnp.allclose(result.model.weight[0], true_w, atol=0.02))

    def test_loss_decreases(self):
        key = jr.PRNGKey(0)
        x = jr.normal(key, (128, 2))
        y = jnp.sum(x, axis=1, keepdims=True)
        model = eqx.nn.Linear(2, 1, key=jr.PRNGKey(1))
        loss = lambda m, xb, yb: jnp.mean((jax.vmap(m)(xb) - yb) ** 2)  # noqa: E731
        result = fit(model, loss, (x, y), steps=500, learning_rate=0.05)
        assert float(result.train_losses[-1]) < float(result.train_losses[0])

    def test_minibatching_runs(self):
        key = jr.PRNGKey(0)
        x = jr.normal(key, (200, 2))
        y = jnp.sum(x, axis=1, keepdims=True)
        model = eqx.nn.Linear(2, 1, key=jr.PRNGKey(1))
        loss = lambda m, xb, yb: jnp.mean((jax.vmap(m)(xb) - yb) ** 2)  # noqa: E731
        result = fit(
            model,
            loss,
            (x, y),
            steps=200,
            batch_size=32,
            key=jr.PRNGKey(2),
            learning_rate=0.05,
        )
        assert len(result.train_losses) == 200
        assert float(result.train_losses[-1]) < float(result.train_losses[0])

    def test_early_stopping_triggers(self):
        key = jr.PRNGKey(0)
        x = jr.normal(key, (200, 2))
        y = jnp.sum(x, axis=1, keepdims=True)
        model = eqx.nn.Linear(2, 1, key=jr.PRNGKey(1))
        loss = lambda m, xb, yb: jnp.mean((jax.vmap(m)(xb) - yb) ** 2)  # noqa: E731
        result = fit(
            model,
            loss,
            (x[:150], y[:150]),
            validation_data=(x[150:], y[150:]),
            steps=20_000,
            validate_every=10,
            patience=3,
            learning_rate=0.05,
        )
        assert result.stopped_early
        assert len(result.train_losses) < 20_000

    def test_returns_best_model_not_last(self):
        key = jr.PRNGKey(0)
        x = jr.normal(key, (100, 2))
        y = jnp.sum(x, axis=1, keepdims=True)
        model = eqx.nn.Linear(2, 1, key=jr.PRNGKey(1))
        loss = lambda m, xb, yb: jnp.mean((jax.vmap(m)(xb) - yb) ** 2)  # noqa: E731
        result = fit(
            model,
            loss,
            (x, y),
            validation_data=(x, y),
            steps=300,
            validate_every=10,
            learning_rate=0.05,
        )
        best = float(min(result.val_losses))
        assert abs(float(loss(result.model, x, y)) - best) < 1e-6

    def test_patience_requires_validation_data(self):
        model = eqx.nn.Linear(2, 1, key=jr.PRNGKey(0))
        with pytest.raises(DataValidationError, match="patience requires"):
            fit(model, lambda m, x: jnp.array(0.0), (jnp.ones((4, 2)),), patience=2)

    def test_batching_requires_a_key(self):
        model = eqx.nn.Linear(2, 1, key=jr.PRNGKey(0))
        with pytest.raises(DataValidationError, match="requires a PRNG key"):
            fit(model, lambda m, x: jnp.array(0.0), (jnp.ones((8, 2)),), batch_size=2)


class TestDataloader:
    def test_yields_correct_shapes(self):
        x = jnp.arange(20.0)[:, None]
        y = jnp.arange(20.0)[:, None] * 2
        loader = dataloader((x, y), batch_size=5, key=jr.PRNGKey(0))
        bx, by = next(loader)
        assert bx.shape == (5, 1) and by.shape == (5, 1)

    def test_keeps_arrays_aligned(self):
        x = jnp.arange(20.0)[:, None]
        y = x * 3.0
        loader = dataloader((x, y), batch_size=5, key=jr.PRNGKey(0))
        bx, by = next(loader)
        assert bool(jnp.allclose(by, bx * 3.0))

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(DataValidationError, match="leading dimension"):
            next(
                dataloader(
                    (jnp.ones((10, 1)), jnp.ones((5, 1))), batch_size=2, key=jr.PRNGKey(0)
                )
            )

    def test_rejects_oversized_batch(self):
        with pytest.raises(DataValidationError, match="exceeds dataset size"):
            next(dataloader((jnp.ones((4, 1)),), batch_size=8, key=jr.PRNGKey(0)))


class TestCalibration:
    def test_fit_gbm_recovers_parameters(self, key):
        ts = jnp.linspace(0, 20, 10_001)
        truth = GeometricBrownianMotion(mu=0.08, sigma=0.3)
        path = truth.sample(jnp.array(100.0), ts=ts, key=key, n_paths=1)[0]
        result = fit_gbm(path, ts)
        # sigma is precisely identified from quadratic variation.
        assert abs(float(result.process.sigma) - 0.3) < 0.01
        # mu converges much more slowly; a wide band is honest.
        assert abs(float(result.process.mu) - 0.08) < 0.15

    def test_fit_ou_recovers_parameters(self, key):
        ts = jnp.linspace(0, 400, 40_001)
        truth = OrnsteinUhlenbeck(kappa=1.5, theta=0.03, sigma=0.2)
        path = truth.sample(jnp.array(0.03), ts=ts, key=key, n_paths=1)[0]
        result = fit_ou(path, ts)
        assert abs(float(result.process.kappa) - 1.5) < 0.15
        assert abs(float(result.process.theta) - 0.03) < 0.02
        assert abs(float(result.process.sigma) - 0.2) < 0.01

    def test_fit_mle_matches_closed_form(self, key):
        ts = jnp.linspace(0, 10, 5001)
        path = GeometricBrownianMotion(mu=0.05, sigma=0.25).sample(
            jnp.array(100.0), ts=ts, key=key, n_paths=1
        )[0]

        closed_form = fit_gbm(path, ts)
        iterative = fit_mle(
            lambda r: GeometricBrownianMotion(mu=r[0], sigma=jax.nn.softplus(r[1])),
            jnp.array([0.0, -1.0]),
            lambda p: p.log_likelihood(path, ts),
            steps=2000,
        )
        assert (
            abs(float(iterative.process.sigma) - float(closed_form.process.sigma)) < 0.01
        )

    def test_calibrated_likelihood_beats_the_start(self, key):
        ts = jnp.linspace(0, 5, 2001)
        path = GeometricBrownianMotion(mu=0.1, sigma=0.4).sample(
            jnp.array(100.0), ts=ts, key=key, n_paths=1
        )[0]
        result = fit_mle(
            lambda r: GeometricBrownianMotion(mu=r[0], sigma=jax.nn.softplus(r[1])),
            jnp.array([0.0, 0.0]),
            lambda p: p.log_likelihood(path, ts),
            steps=1500,
        )
        assert float(result.history[-1]) > float(result.history[0])
