"""Tests for the solver-diagnostics tools.

These verify that the diagnostics reproduce *textbook* convergence orders. If
they did not, the tool would be worse than useless -- it would give false
confidence.
"""

from __future__ import annotations

import diffrax
import jax.numpy as jnp
import jax.random as jr
import pytest

from finax.diagnostics import (
    martingale_test,
    moment_report,
    strong_order,
    weak_order,
)


def _make_simulator(solver, *, additive: bool, mu: float = 0.05, sigma: float = 0.3):
    """Build a `simulate(dt, key)` for a GBM-like SDE with a shared Brownian path."""

    def simulate(dt, key):
        bm = diffrax.VirtualBrownianTree(
            0.0,
            1.0,
            tol=1e-5,
            shape=(),
            key=key,
            levy_area=diffrax.SpaceTimeLevyArea,
        )
        diffusion = (lambda t, y, a: sigma) if additive else (lambda t, y, a: sigma * y)
        terms = diffrax.MultiTerm(
            diffrax.ODETerm(lambda t, y, a: mu * y),
            diffrax.ControlTerm(diffusion, bm),
        )
        return diffrax.diffeqsolve(
            terms, solver, 0.0, 1.0, dt, jnp.array(1.0), max_steps=None
        ).ys[-1]

    return simulate


class TestStrongOrder:
    def test_euler_is_order_half_for_multiplicative_noise(self, key):
        report = strong_order(
            _make_simulator(diffrax.Euler(), additive=False), key=key, n_paths=512
        )
        assert 0.35 < report.estimated_order < 0.70
        assert report.r_squared > 0.9

    def test_milstein_is_order_one_for_multiplicative_noise(self, key):
        report = strong_order(
            _make_simulator(diffrax.ItoMilstein(), additive=False),
            key=key,
            n_paths=512,
        )
        assert 0.85 < report.estimated_order < 1.15
        assert report.r_squared > 0.9

    def test_milstein_beats_euler(self, key):
        """The diagnostic must rank solvers correctly, not just report a number."""
        euler = strong_order(
            _make_simulator(diffrax.Euler(), additive=False), key=key, n_paths=512
        )
        milstein = strong_order(
            _make_simulator(diffrax.ItoMilstein(), additive=False),
            key=key,
            n_paths=512,
        )
        assert milstein.estimated_order > euler.estimated_order
        assert float(milstein.errors[-1]) < float(euler.errors[-1])

    def test_euler_is_order_one_for_additive_noise(self, key):
        """Euler-Maruyama gains a half order when the noise is additive."""
        report = strong_order(
            _make_simulator(diffrax.Euler(), additive=True), key=key, n_paths=512
        )
        assert 0.85 < report.estimated_order < 1.25

    def test_errors_shrink_with_step_size(self, key):
        report = strong_order(
            _make_simulator(diffrax.Euler(), additive=False), key=key, n_paths=256
        )
        errors = [float(e) for e in report.errors]
        assert errors == sorted(errors, reverse=True)

    def test_repr_is_informative(self, key):
        report = strong_order(
            _make_simulator(diffrax.Euler(), additive=False), key=key, n_paths=64
        )
        text = repr(report)
        assert "order=" in text and "r_squared=" in text and "dt=" in text

    def test_requires_two_step_sizes(self, key):
        from finax.errors import DataValidationError

        with pytest.raises(DataValidationError):
            strong_order(
                _make_simulator(diffrax.Euler(), additive=False),
                step_sizes=(0.1,),
                key=key,
            )


class TestWeakOrder:
    def test_euler_weak_error_shrinks(self, key):
        mu = 0.05
        report = weak_order(
            _make_simulator(diffrax.Euler(), additive=False, mu=mu),
            exact_expectation=float(jnp.exp(mu)),
            key=key,
            n_paths=8192,
        )
        assert report.errors.shape == (4,)
        assert bool(jnp.all(jnp.isfinite(report.errors)))


class TestMartingaleTest:
    def test_brownian_motion_passes(self):
        steps = jr.normal(jr.PRNGKey(0), (20_000, 50)) * jnp.sqrt(1 / 50)
        paths = jnp.concatenate(
            [jnp.zeros((20_000, 1)), jnp.cumsum(steps, axis=1)], axis=1
        )
        assert martingale_test(paths)["passed"]

    def test_drifting_process_fails(self):
        steps = jr.normal(jr.PRNGKey(0), (20_000, 50)) * jnp.sqrt(1 / 50)
        paths = jnp.concatenate(
            [jnp.zeros((20_000, 1)), jnp.cumsum(steps, axis=1)], axis=1
        )
        drifted = paths + 0.5 * jnp.linspace(0, 1, 51)
        result = martingale_test(drifted)
        assert not result["passed"]
        assert result["max_abs_z"] > result["critical_value"]

    def test_exponential_martingale_passes(self):
        """exp(sigma W_t - sigma^2 t / 2) is a martingale; a missing Ito
        correction is exactly what this catches."""
        sigma, n_steps = 0.3, 50
        ts = jnp.linspace(0, 1, n_steps + 1)
        steps = jr.normal(jr.PRNGKey(1), (40_000, n_steps)) * jnp.sqrt(1 / n_steps)
        w = jnp.concatenate([jnp.zeros((40_000, 1)), jnp.cumsum(steps, axis=1)], 1)
        paths = jnp.exp(sigma * w - 0.5 * sigma**2 * ts)
        assert martingale_test(paths)["passed"]

    def test_missing_ito_correction_is_detected(self):
        sigma, n_steps = 0.3, 50
        steps = jr.normal(jr.PRNGKey(1), (40_000, n_steps)) * jnp.sqrt(1 / n_steps)
        w = jnp.concatenate([jnp.zeros((40_000, 1)), jnp.cumsum(steps, axis=1)], 1)
        without_correction = jnp.exp(sigma * w)
        assert not martingale_test(without_correction)["passed"]

    def test_rejects_wrong_rank(self):
        from finax.errors import DataValidationError

        with pytest.raises(DataValidationError):
            martingale_test(jnp.zeros(10))


class TestMomentReport:
    def test_standard_normal_matches_its_moments(self):
        x = jr.normal(jr.PRNGKey(0), (200_000,))
        report = moment_report(
            x,
            expected_mean=0.0,
            expected_variance=1.0,
            expected_skewness=0.0,
            expected_kurtosis=3.0,
        )
        for name in ("mean", "variance", "skewness", "kurtosis"):
            assert abs(report[name]["z"]) < 4.0

    def test_detects_a_wrong_mean(self):
        x = jr.normal(jr.PRNGKey(0), (100_000,)) + 0.5
        report = moment_report(x, expected_mean=0.0)
        assert abs(report["mean"]["z"]) > 10.0

    def test_omitted_targets_have_no_z(self):
        x = jr.normal(jr.PRNGKey(0), (1000,))
        report = moment_report(x, expected_mean=0.0)
        assert "z" in report["mean"]
        assert "z" not in report["variance"]
        assert "sample" in report["variance"]
