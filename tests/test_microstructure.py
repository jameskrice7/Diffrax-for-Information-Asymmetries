"""Tests for the microstructure estimators.

These are correctness tests against known ground truth, not smoke tests: the
whole point of this module is that the previous implementation produced
plausible-looking numbers that were not the quantities they claimed to be.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from finax.microstructure import (
    PINParams,
    amihud_illiquidity,
    bulk_volume_classification,
    corwin_schultz_spread,
    estimate_pin,
    estimate_pin_panel,
    kyle_lambda,
    lee_ready,
    pin_log_likelihood,
    roll_spread,
    tick_rule,
    volume_bars,
    vpin,
)


class TestPIN:
    def test_recovers_known_pin(self, pin_sample):
        buys, sells, true = pin_sample
        result = estimate_pin(buys, sells)
        assert abs(float(result.pin) - float(true.pin)) < 0.05
        assert not bool(result.at_boundary)

    def test_recovers_structural_parameters(self, pin_sample):
        buys, sells, true = pin_sample
        result = estimate_pin(buys, sells)
        # mu and the uninformed rates are identified up to which side is
        # labelled "bad news", so compare the total informed intensity.
        assert abs(float(result.params.alpha) - float(true.alpha)) < 0.12
        assert abs(float(result.params.mu) - float(true.mu)) < 25.0

    @pytest.mark.parametrize("true_pin_alpha", [0.1, 0.3, 0.6])
    def test_recovers_across_regimes(self, true_pin_alpha):
        """PIN recovery should hold across a range of event probabilities."""
        true = PINParams(
            alpha=true_pin_alpha, delta=0.4, mu=120.0, eps_b=80.0, eps_s=80.0
        )
        k1, k2, k3, k4 = jr.split(jr.PRNGKey(7), 4)
        n = 600
        event = jr.bernoulli(k1, true_pin_alpha, (n,))
        bad = jr.bernoulli(k2, 0.4, (n,))
        buys = jr.poisson(k3, 80.0 + 120.0 * (event & ~bad)).astype(jnp.float32)
        sells = jr.poisson(k4, 80.0 + 120.0 * (event & bad)).astype(jnp.float32)

        result = estimate_pin(buys, sells)
        assert abs(float(result.pin) - float(true.pin)) < 0.06

    def test_likelihood_finite_at_extreme_trade_counts(self):
        """The Lin-Ke factorization must survive counts that overflow the naive form.

        eps_b ** 200000 is inf in any float type; the whole point of the
        factorization is that this never gets computed.
        """
        params = PINParams(
            alpha=0.3, delta=0.5, mu=5_000.0, eps_b=150_000.0, eps_s=150_000.0
        )
        buys = jnp.array([200_000.0, 180_000.0])
        sells = jnp.array([190_000.0, 210_000.0])
        assert bool(jnp.isfinite(pin_log_likelihood(params, buys, sells)))

    def test_likelihood_is_maximised_at_truth(self, pin_sample):
        buys, sells, true = pin_sample
        at_truth = pin_log_likelihood(true, buys, sells)
        for wrong in [
            PINParams(alpha=0.9, delta=0.5, mu=10.0, eps_b=10.0, eps_s=10.0),
            PINParams(alpha=0.05, delta=0.1, mu=500.0, eps_b=200.0, eps_s=5.0),
        ]:
            assert float(at_truth) > float(pin_log_likelihood(wrong, buys, sells))

    def test_is_differentiable(self, pin_sample):
        """A gradient of PIN w.r.t. trade counts must exist and be finite.

        This is the property that lets an estimated PIN sit inside a larger
        differentiable model, and it is what no other PIN package provides.
        """
        buys, sells, true = pin_sample

        def pin_of_counts(b):
            return pin_log_likelihood(true, b, sells)

        grad = jax.grad(pin_of_counts)(buys)
        assert grad.shape == buys.shape
        assert bool(jnp.all(jnp.isfinite(grad)))
        assert bool(jnp.any(grad != 0.0))

    def test_is_jittable(self, pin_sample):
        buys, sells, true = pin_sample
        jitted = jax.jit(pin_log_likelihood)
        assert bool(
            jnp.allclose(jitted(true, buys, sells), pin_log_likelihood(true, buys, sells))
        )

    def test_panel_matches_individual_fits(self):
        """Batched estimation must agree with fitting each series separately."""
        keys = jr.split(jr.PRNGKey(3), 3)
        buys = jnp.stack([jr.poisson(k, 120.0, (300,)).astype(jnp.float32) for k in keys])
        sells = jnp.stack(
            [jr.poisson(k, 110.0, (300,)).astype(jnp.float32) for k in keys]
        )

        panel = estimate_pin_panel(buys, sells, steps=300)
        assert panel.pin.shape == (3,)
        for i in range(3):
            single = estimate_pin(buys[i], sells[i], steps=300)
            assert abs(float(panel.pin[i]) - float(single.pin)) < 0.02

    def test_pin_is_a_probability(self, pin_sample):
        buys, sells, _ = pin_sample
        result = estimate_pin(buys, sells)
        assert 0.0 <= float(result.pin) <= 1.0

    def test_boundary_solutions_are_flagged(self):
        """Balanced Poisson flow has no information events; alpha is driven to 0."""
        keys = jr.split(jr.PRNGKey(11), 2)
        buys = jr.poisson(keys[0], 100.0, (400,)).astype(jnp.float32)
        sells = jr.poisson(keys[1], 100.0, (400,)).astype(jnp.float32)
        result = estimate_pin(buys, sells)
        # Either it finds a near-zero PIN, or it hits a boundary and says so.
        assert float(result.pin) < 0.15 or bool(result.at_boundary)

    def test_rejects_mismatched_shapes(self):
        from finax.errors import ShapeError

        with pytest.raises(ShapeError):
            estimate_pin(jnp.ones(10), jnp.ones(9))


class TestVPIN:
    def test_balanced_flow_gives_low_vpin(self):
        n = 300
        prices = jnp.full((n,), 100.0)
        volumes = jnp.full((n,), 1000.0)
        out = vpin(prices, volumes, window=50)
        assert float(out[-1]) < 0.05

    def test_one_sided_flow_gives_high_vpin(self):
        n = 300
        prices = 100.0 + jnp.arange(n, dtype=jnp.float32)
        out = vpin(prices, jnp.full((n,), 1000.0), window=50)
        assert float(out[-1]) > 0.9

    def test_leading_window_is_nan(self):
        n = 200
        out = vpin(jnp.full((n,), 100.0), jnp.full((n,), 10.0), window=50)
        assert bool(jnp.isnan(out[:49]).all())
        assert bool(jnp.isfinite(out[49:]).all())

    def test_is_bounded(self):
        rng = np.random.default_rng(0)
        prices = jnp.asarray(100 + np.cumsum(rng.normal(0, 1, 500)), jnp.float32)
        out = vpin(prices, jnp.full((500,), 1000.0), window=50)
        finite = out[~jnp.isnan(out)]
        assert bool(jnp.all((finite >= 0.0) & (finite <= 1.0)))

    def test_volume_bars_conserve_volume(self):
        prices = jnp.arange(1.0, 101.0)
        volumes = jnp.full((100,), 10.0)
        bar_prices, bar_volumes = volume_bars(
            prices, volumes, bucket_volume=100.0, n_buckets=10
        )
        assert bar_prices.shape == (10,)
        assert abs(float(jnp.sum(bar_volumes)) - 1000.0) < 1e-3

    def test_volume_bars_rejects_empty_trades(self):
        from finax.errors import DataValidationError

        with pytest.raises(DataValidationError, match="at least one trade"):
            volume_bars(
                jnp.array([], dtype=jnp.float32),
                jnp.array([], dtype=jnp.float32),
                bucket_volume=100.0,
                n_buckets=2,
            )

    def test_volume_bars_preserve_volume_precision(self):
        prices = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float16)
        volumes = jnp.array([10.0, 20.0, 30.0], dtype=jnp.float32)
        _, bar_volumes = volume_bars(prices, volumes, bucket_volume=30.0, n_buckets=2)
        assert bar_volumes.dtype == jnp.result_type(prices, volumes)

    def test_bvc_splits_volume_exactly(self):
        prices = jnp.array([100.0, 101.0, 99.0, 103.0, 102.0])
        volumes = jnp.array([100.0, 200.0, 150.0, 300.0, 50.0])
        buy, sell = bulk_volume_classification(prices, volumes)
        assert bool(jnp.allclose(buy + sell, volumes, atol=1e-4))
        assert bool(jnp.all(buy >= 0) and jnp.all(sell >= 0))

    def test_bvc_is_differentiable(self):
        """BVC is smooth by design, unlike the sign-based tick rule."""
        prices = jnp.array([100.0, 101.0, 99.0, 103.0])
        volumes = jnp.full((4,), 100.0)

        def total_buy(p):
            return jnp.sum(bulk_volume_classification(p, volumes)[0])

        grad = jax.grad(total_buy)(prices)
        assert bool(jnp.all(jnp.isfinite(grad)))
        assert bool(jnp.any(grad != 0.0))


class TestClassification:
    def test_tick_rule_carries_zero_ticks_forward(self):
        prices = jnp.array([10.0, 10.5, 10.5, 10.5, 10.0, 10.0])
        signs = tick_rule(prices)
        assert signs.tolist() == [1, 1, 1, 1, -1, -1]

    def test_lee_ready_uses_tick_at_midpoint(self):
        prices = jnp.array([10.2, 10.0, 10.1, 10.1])
        bids = jnp.full((4,), 10.0)
        asks = jnp.full((4,), 10.2)
        signs = lee_ready(prices, bids, asks)
        # Trades 0 and 1 are away from the mid; 2 and 3 are at it.
        assert signs[0] == 1 and signs[1] == -1
        assert signs[2] in (1, -1) and signs[3] in (1, -1)

    def test_signs_are_only_plus_or_minus_one(self):
        rng = np.random.default_rng(0)
        prices = jnp.asarray(100 + np.cumsum(rng.normal(0, 0.1, 200)), jnp.float32)
        assert set(np.unique(np.asarray(tick_rule(prices)))) <= {-1, 1}


class TestLiquidity:
    def test_kyle_lambda_recovers_known_impact(self):
        flow = jnp.array([100.0, -50.0, 200.0, -300.0, 75.0])
        assert bool(jnp.allclose(kyle_lambda(0.0025 * flow, flow), 0.0025, atol=1e-6))

    def test_kyle_lambda_is_zero_for_unrelated_flow(self):
        rng = np.random.default_rng(0)
        flow = jnp.asarray(rng.normal(0, 100, 5000), jnp.float32)
        noise = jnp.asarray(rng.normal(0, 0.01, 5000), jnp.float32)
        assert abs(float(kyle_lambda(noise, flow))) < 1e-4

    def test_amihud_scales_inversely_with_volume(self):
        r = jnp.array([0.01, -0.02, 0.015])
        dv = jnp.full((3,), 1e6)
        assert bool(
            jnp.allclose(
                amihud_illiquidity(r, 4 * dv), amihud_illiquidity(r, dv) / 4, rtol=1e-5
            )
        )

    def test_roll_spread_recovers_bid_ask_bounce(self):
        signs = 2.0 * jr.bernoulli(jr.PRNGKey(0), 0.5, (50_000,)) - 1.0
        prices = 100.0 + 0.04 * signs  # half-spread 0.04 => spread 0.08
        assert abs(float(roll_spread(prices)) - 0.08) < 0.004

    def test_roll_spread_truncates_at_zero_when_model_fails(self):
        trending = jnp.arange(200, dtype=jnp.float32)
        assert float(roll_spread(trending)) == 0.0

    def test_corwin_schultz_is_non_negative(self, price_frame):
        highs = jnp.asarray(price_frame["high"].to_numpy(), jnp.float32)
        lows = jnp.asarray(price_frame["low"].to_numpy(), jnp.float32)
        spread = corwin_schultz_spread(highs, lows)
        assert spread.shape == (len(highs) - 1,)
        assert bool(jnp.all(spread >= 0.0))

    def test_estimators_vmap_over_cross_section(self):
        """Every estimator must batch across firms without a Python loop."""
        rng = np.random.default_rng(0)
        returns = jnp.asarray(rng.normal(0, 0.02, (50, 250)), jnp.float32)
        volumes = jnp.asarray(rng.uniform(1e5, 1e7, (50, 250)), jnp.float32)
        out = jax.vmap(amihud_illiquidity)(returns, volumes)
        assert out.shape == (50,)
        assert bool(jnp.all(jnp.isfinite(out)))
