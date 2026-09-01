"""Tests for the classical process library.

Exact samplers have exactly known moments, so these are sharp tests rather than
loose sanity checks.
"""

from __future__ import annotations

import jax.numpy as jnp

from finax.processes import (
    CoxIngersollRoss,
    GeometricBrownianMotion,
    Heston,
    MertonJumpDiffusion,
    OrnsteinUhlenbeck,
)


class TestGBM:
    def test_mean_matches_analytic(self, key):
        mu, sigma, s0, t = 0.06, 0.25, 100.0, 1.0
        paths = GeometricBrownianMotion(mu=mu, sigma=sigma).sample(
            jnp.array(s0), ts=jnp.linspace(0, t, 253), key=key, n_paths=100_000
        )
        assert abs(float(jnp.mean(paths[:, -1])) - s0 * float(jnp.exp(mu * t))) < 0.5

    def test_variance_matches_analytic(self, key):
        mu, sigma, s0 = 0.0, 0.3, 1.0
        paths = GeometricBrownianMotion(mu=mu, sigma=sigma).sample(
            jnp.array(s0), ts=jnp.linspace(0, 1, 253), key=key, n_paths=200_000
        )
        # Var = S0^2 e^{2 mu t}(e^{sigma^2 t} - 1).
        expected = float(jnp.exp(sigma**2) - 1.0)
        assert abs(float(jnp.var(paths[:, -1])) - expected) < 0.02

    def test_log_returns_are_gaussian(self, key):
        paths = GeometricBrownianMotion(mu=0.05, sigma=0.2).sample(
            jnp.array(100.0), ts=jnp.linspace(0, 1, 253), key=key, n_paths=20_000
        )
        lr = jnp.log(paths[:, -1] / 100.0)
        z = (lr - jnp.mean(lr)) / jnp.std(lr)
        assert abs(float(jnp.mean(z**3))) < 0.1  # no skew
        assert abs(float(jnp.mean(z**4)) - 3.0) < 0.15  # Gaussian kurtosis

    def test_stays_positive(self, key):
        paths = GeometricBrownianMotion(mu=-0.5, sigma=1.5).sample(
            jnp.array(1.0), ts=jnp.linspace(0, 5, 500), key=key, n_paths=1000
        )
        assert bool(jnp.all(paths > 0.0))

    def test_log_likelihood_peaks_at_truth(self, key):
        ts = jnp.linspace(0, 5, 2001)
        truth = GeometricBrownianMotion(mu=0.08, sigma=0.3)
        path = truth.sample(jnp.array(100.0), ts=ts, key=key, n_paths=1)[0]
        best = truth.log_likelihood(path, ts)
        for wrong in [
            GeometricBrownianMotion(mu=0.08, sigma=0.6),
            GeometricBrownianMotion(mu=0.08, sigma=0.15),
        ]:
            assert float(best) > float(wrong.log_likelihood(path, ts))


class TestOU:
    def test_reverts_to_theta(self, key):
        theta = 0.05
        paths = OrnsteinUhlenbeck(kappa=3.0, theta=theta, sigma=0.1).sample(
            jnp.array(0.9), ts=jnp.linspace(0, 20, 2001), key=key, n_paths=5000
        )
        assert abs(float(jnp.mean(paths[:, -1])) - theta) < 0.01

    def test_stationary_variance_matches_analytic(self, key):
        kappa, sigma = 2.0, 0.15
        paths = OrnsteinUhlenbeck(kappa=kappa, theta=0.0, sigma=sigma).sample(
            jnp.array(0.0), ts=jnp.linspace(0, 30, 3001), key=key, n_paths=20_000
        )
        expected = sigma**2 / (2 * kappa)
        assert abs(float(jnp.var(paths[:, -1])) - expected) < 0.0005

    def test_mean_reversion_speed(self, key):
        """E[X_t] = theta + (X_0 - theta) e^{-kappa t}."""
        kappa, theta, x0, t = 1.5, 0.0, 1.0, 1.0
        ts = jnp.linspace(0, t, 501)
        paths = OrnsteinUhlenbeck(kappa=kappa, theta=theta, sigma=0.05).sample(
            jnp.array(x0), ts=ts, key=key, n_paths=20_000
        )
        expected = theta + (x0 - theta) * float(jnp.exp(-kappa * t))
        assert abs(float(jnp.mean(paths[:, -1])) - expected) < 0.005


class TestCIR:
    def test_stays_non_negative(self, key):
        paths = CoxIngersollRoss(kappa=2.0, theta=0.04, sigma=0.15).sample(
            jnp.array(0.04), ts=jnp.linspace(0, 10, 1001), key=key, n_paths=2000
        )
        assert bool(jnp.all(paths >= 0.0))

    def test_stays_non_negative_when_feller_violated(self, key):
        """The regime where naive Euler produces NaN via sqrt of a negative."""
        process = CoxIngersollRoss(kappa=0.4, theta=0.01, sigma=1.0)
        assert not bool(process.feller_satisfied)
        paths = process.sample(
            jnp.array(0.01), ts=jnp.linspace(0, 5, 501), key=key, n_paths=2000
        )
        assert bool(jnp.all(paths >= 0.0))
        assert bool(jnp.all(jnp.isfinite(paths)))

    def test_mean_reverts_to_theta(self, key):
        theta = 0.05
        paths = CoxIngersollRoss(kappa=3.0, theta=theta, sigma=0.2).sample(
            jnp.array(0.2), ts=jnp.linspace(0, 15, 1501), key=key, n_paths=10_000
        )
        assert abs(float(jnp.mean(paths[:, -1])) - theta) < 0.005

    def test_stationary_variance_matches_analytic(self, key):
        """Stationary variance of CIR is sigma^2 theta / (2 kappa)."""
        kappa, theta, sigma = 4.0, 0.06, 0.2
        paths = CoxIngersollRoss(kappa=kappa, theta=theta, sigma=sigma).sample(
            jnp.array(theta), ts=jnp.linspace(0, 20, 2001), key=key, n_paths=40_000
        )
        expected = sigma**2 * theta / (2 * kappa)
        assert abs(float(jnp.var(paths[:, -1])) - expected) < 5e-5

    def test_feller_condition_flag(self):
        assert bool(CoxIngersollRoss(kappa=2.0, theta=0.1, sigma=0.1).feller_satisfied)
        assert not bool(
            CoxIngersollRoss(kappa=0.1, theta=0.01, sigma=1.0).feller_satisfied
        )


class TestHeston:
    def test_variance_stays_non_negative(self, key):
        _, v = Heston(mu=0.03, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7).sample(
            jnp.array(jnp.log(100.0)),
            jnp.array(0.04),
            ts=jnp.linspace(0, 2, 505),
            key=key,
            n_paths=2000,
        )
        assert bool(jnp.all(v >= 0.0))

    def test_negative_rho_gives_negative_skew(self, key):
        """The leverage effect: this is what Heston is for."""
        log_s, _ = Heston(mu=0.0, kappa=2.0, theta=0.04, xi=0.5, rho=-0.8).sample(
            jnp.array(0.0),
            jnp.array(0.04),
            ts=jnp.linspace(0, 1, 253),
            key=key,
            n_paths=20_000,
        )
        z = (log_s[:, -1] - jnp.mean(log_s[:, -1])) / jnp.std(log_s[:, -1])
        assert float(jnp.mean(z**3)) < -0.15

    def test_positive_rho_gives_positive_skew(self, key):
        log_s, _ = Heston(mu=0.0, kappa=2.0, theta=0.04, xi=0.5, rho=0.8).sample(
            jnp.array(0.0),
            jnp.array(0.04),
            ts=jnp.linspace(0, 1, 253),
            key=key,
            n_paths=20_000,
        )
        z = (log_s[:, -1] - jnp.mean(log_s[:, -1])) / jnp.std(log_s[:, -1])
        assert float(jnp.mean(z**3)) > 0.15

    def test_diffusion_matrix_has_correct_correlation(self):
        model = Heston(mu=0.0, kappa=1.0, theta=0.04, xi=0.3, rho=-0.6)
        g = model.diffusion(0.0, jnp.array([0.0, 0.04]))
        assert g.shape == (2, 2)
        # Row correlation must equal rho.
        cov = g @ g.T
        implied = cov[0, 1] / jnp.sqrt(cov[0, 0] * cov[1, 1])
        assert abs(float(implied) - (-0.6)) < 1e-5


class TestMerton:
    def test_compensated_drift_preserves_the_mean(self, key):
        mu, s0 = 0.05, 100.0
        paths = MertonJumpDiffusion(
            mu=mu, sigma=0.2, intensity=2.0, jump_mean=-0.15, jump_std=0.2
        ).sample(jnp.array(s0), ts=jnp.linspace(0, 1, 253), key=key, n_paths=200_000)
        assert abs(float(jnp.mean(paths[:, -1])) - s0 * float(jnp.exp(mu))) < 1.0

    def test_jumps_create_excess_kurtosis(self, key):
        ts = jnp.linspace(0, 1, 253)
        jumpy = MertonJumpDiffusion(
            mu=0.0, sigma=0.1, intensity=3.0, jump_mean=-0.1, jump_std=0.2
        ).sample(jnp.array(100.0), ts=ts, key=key, n_paths=50_000)
        r = jnp.log(jumpy[:, -1] / 100.0)
        z = (r - jnp.mean(r)) / jnp.std(r)
        assert float(jnp.mean(z**4)) > 3.3

    def test_zero_intensity_reduces_to_gbm(self, key):
        ts = jnp.linspace(0, 1, 253)
        merton = MertonJumpDiffusion(
            mu=0.05, sigma=0.2, intensity=0.0, jump_mean=0.0, jump_std=0.1
        ).sample(jnp.array(100.0), ts=ts, key=key, n_paths=20_000)
        gbm = GeometricBrownianMotion(mu=0.05, sigma=0.2).sample(
            jnp.array(100.0), ts=ts, key=key, n_paths=20_000
        )
        assert abs(float(jnp.mean(merton[:, -1])) - float(jnp.mean(gbm[:, -1]))) < 0.5
        assert abs(float(jnp.std(merton[:, -1])) - float(jnp.std(gbm[:, -1]))) < 0.5

    def test_stays_positive(self, key):
        paths = MertonJumpDiffusion(
            mu=0.0, sigma=0.3, intensity=5.0, jump_mean=-0.3, jump_std=0.3
        ).sample(jnp.array(50.0), ts=jnp.linspace(0, 3, 500), key=key, n_paths=2000)
        assert bool(jnp.all(paths > 0.0))


def test_all_processes_expose_drift_and_diffusion():
    """Every process must be usable as a Diffrax term."""
    processes = [
        GeometricBrownianMotion(mu=0.05, sigma=0.2),
        OrnsteinUhlenbeck(kappa=1.0, theta=0.0, sigma=0.1),
        CoxIngersollRoss(kappa=1.0, theta=0.04, sigma=0.1),
        MertonJumpDiffusion(
            mu=0.05, sigma=0.2, intensity=1.0, jump_mean=0.0, jump_std=0.1
        ),
    ]
    for process in processes:
        y = jnp.array(0.5)
        assert jnp.isfinite(process.drift(0.0, y)).all()
        assert jnp.isfinite(process.diffusion(0.0, y)).all()
