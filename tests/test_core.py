"""Tests for control-path construction and solve configuration."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from finax.core import (
    SolveConfig,
    build_control_path,
    fill_forward,
    pad_ragged,
    prepare_channels,
    solve_ode,
    solve_sde,
)
from finax.errors import ShapeError


class TestFillForward:
    def test_propagates_last_observation(self):
        ys = jnp.array([[1.0], [jnp.nan], [jnp.nan], [4.0], [jnp.nan]])
        assert fill_forward(ys)[:, 0].tolist() == [1.0, 1.0, 1.0, 4.0, 4.0]

    def test_leaves_leading_nans(self):
        ys = jnp.array([[jnp.nan], [2.0]])
        out = fill_forward(ys)
        assert bool(jnp.isnan(out[0, 0]))
        assert float(out[1, 0]) == 2.0

    def test_channels_are_independent(self):
        ys = jnp.array([[1.0, 10.0], [jnp.nan, 20.0], [3.0, jnp.nan]])
        out = fill_forward(ys)
        assert out.tolist() == [[1.0, 10.0], [1.0, 20.0], [3.0, 20.0]]

    def test_rejects_wrong_rank(self):
        with pytest.raises(ShapeError):
            fill_forward(jnp.ones(5))

    def test_is_jittable(self):
        ys = jnp.array([[1.0], [jnp.nan], [3.0]])
        assert bool(jnp.allclose(jax.jit(fill_forward)(ys), fill_forward(ys)))


class TestPrepareChannels:
    def test_output_width(self, irregular_series):
        ts, ys = irregular_series
        out = prepare_channels(ts, ys)
        # time + 2 values + 2 masks.
        assert out.shape == (6, 5)

    def test_never_returns_nan(self, irregular_series):
        ts, ys = irregular_series
        assert bool(jnp.all(jnp.isfinite(prepare_channels(ts, ys))))

    def test_time_is_channel_zero(self, irregular_series):
        ts, ys = irregular_series
        assert bool(jnp.allclose(prepare_channels(ts, ys)[:, 0], ts))

    def test_cumulative_mask_counts_observations(self):
        ts = jnp.array([0.0, 1.0, 2.0, 3.0])
        ys = jnp.array([[1.0], [jnp.nan], [3.0], [4.0]])
        out = prepare_channels(ts, ys)
        # Observations at t=0, 2, 3 => running counts 1,1,2,3.
        assert out[:, -1].tolist() == [1.0, 1.0, 2.0, 3.0]

    def test_binary_mask_option(self):
        ts = jnp.array([0.0, 1.0, 2.0])
        ys = jnp.array([[1.0], [jnp.nan], [3.0]])
        out = prepare_channels(ts, ys, cumulative_mask=False)
        assert out[:, -1].tolist() == [1.0, 0.0, 1.0]

    def test_never_observed_channel_becomes_zero(self):
        ts = jnp.array([0.0, 1.0])
        ys = jnp.array([[jnp.nan], [jnp.nan]])
        out = prepare_channels(ts, ys)
        assert bool(jnp.all(out[:, 1] == 0.0))

    def test_can_disable_augmentation(self, irregular_series):
        ts, ys = irregular_series
        out = prepare_channels(ts, ys, append_time=False, append_mask=False)
        assert out.shape == (6, 2)

    def test_rejects_length_mismatch(self):
        with pytest.raises(ShapeError):
            prepare_channels(jnp.zeros(3), jnp.zeros((4, 1)))


class TestPadRagged:
    def test_pads_to_longest(self):
        a = (jnp.array([0.0, 1.0, 2.0]), jnp.array([[1.0], [2.0], [3.0]]))
        b = (jnp.array([0.0, 1.0]), jnp.array([[4.0], [5.0]]))
        ts, ys, lengths = pad_ragged([a, b])
        assert ts.shape == (2, 3)
        assert ys.shape == (2, 3, 1)
        assert lengths.tolist() == [3, 2]

    def test_repeats_final_timestamp_not_nan(self):
        a = (jnp.array([0.0, 1.0, 2.0]), jnp.zeros((3, 1)))
        b = (jnp.array([0.0, 5.0]), jnp.zeros((2, 1)))
        ts, _ys, _ = pad_ragged([a, b])
        # Non-decreasing times are required by Diffrax; padding must not break it.
        assert ts[1].tolist() == [0.0, 5.0, 5.0]
        assert bool(jnp.all(jnp.diff(ts, axis=1) >= 0))

    def test_pads_values_with_nan(self):
        a = (jnp.array([0.0, 1.0]), jnp.zeros((2, 1)))
        b = (jnp.array([0.0]), jnp.zeros((1, 1)))
        _, ys, _ = pad_ragged([a, b])
        assert bool(jnp.isnan(ys[1, 1, 0]))

    def test_rejects_mismatched_channels(self):
        a = (jnp.array([0.0]), jnp.zeros((1, 2)))
        b = (jnp.array([0.0]), jnp.zeros((1, 3)))
        with pytest.raises(ShapeError, match="same channel count"):
            pad_ragged([a, b])

    def test_rejects_empty(self):
        with pytest.raises(ShapeError):
            pad_ragged([])


class TestBuildControlPath:
    @pytest.mark.parametrize("method", ["hermite", "linear", "rectilinear"])
    def test_all_methods_produce_a_usable_path(self, method, irregular_series):
        ts, ys = irregular_series
        path = build_control_path(ts, ys, method=method)
        assert path.n_channels == 5
        assert float(path.t0) == 0.0
        assert float(path.t1) == 5.0
        value = path.evaluate(path.t0, path.t1)
        assert value.shape == (5,)
        assert bool(jnp.all(jnp.isfinite(value)))

    def test_interpolation_passes_through_observations(self):
        """At an observation time the path must equal the observed value."""
        ts = jnp.array([0.0, 1.0, 2.0, 3.0])
        ys = jnp.array([[5.0], [7.0], [2.0], [9.0]])
        path = build_control_path(ts, ys, append_mask=False)
        for i, t in enumerate(ts):
            assert abs(float(path.evaluate(t)[1]) - float(ys[i, 0])) < 1e-4

    def test_rectilinear_doubles_knots(self, irregular_series):
        ts, ys = irregular_series
        path = build_control_path(ts, ys, method="rectilinear")
        # Rectilinear expansion still spans the same interval.
        assert float(path.t1) == 5.0

    def test_vmaps_over_a_batch(self):
        ts = jnp.stack([jnp.linspace(0.0, 1.0, 5)] * 3)
        ys = jr.normal(jr.PRNGKey(0), (3, 5, 2))
        paths = jax.vmap(build_control_path)(ts, ys)
        assert paths.t1.shape == (3,)
        assert paths.n_channels == 5

    def test_rejects_unknown_method(self, irregular_series):
        ts, ys = irregular_series
        with pytest.raises(ValueError, match="Unknown interpolation method"):
            build_control_path(ts, ys, method="cubic-spline")


class TestSolveConfig:
    def test_defaults_save_terminal_value(self):
        sol = solve_ode(lambda t, y, a: -y, jnp.array(1.0), 0.0, 1.0)
        assert sol.ys is not None
        assert sol.ys.shape == (1,)

    def test_saving_at_preserves_other_fields(self):
        cfg = SolveConfig(dt0=0.01, max_steps=99_999)
        assert cfg.saving_at(jnp.linspace(0, 1, 5)).max_steps == 99_999

    def test_with_steps_for_computes_requirement(self):
        assert SolveConfig(dt0=0.001).with_steps_for(0.0, 10.0).max_steps == 12_501

    def test_with_steps_for_never_shrinks(self):
        assert SolveConfig(dt0=0.5).with_steps_for(0.0, 1.0).max_steps == 4096

    def test_with_steps_for_rejects_adaptive(self):
        with pytest.raises(ValueError, match="needs a fixed dt0"):
            SolveConfig(dt0=None).with_steps_for(0.0, 1.0)

    def test_adaptive_controller_solves(self):
        cfg = SolveConfig().adaptive(rtol=1e-8, atol=1e-10)
        sol = solve_ode(lambda t, y, a: -y, jnp.array(1.0), 0.0, 1.0, config=cfg)
        assert abs(float(sol.ys[-1]) - float(jnp.exp(-1.0))) < 1e-5

    def test_levy_area_is_selected_from_the_solver(self):
        """ShARK needs space-time Levy area; the config must supply it unprompted."""
        cfg = SolveConfig(dt0=0.01).for_additive_noise()
        sol = solve_sde(
            lambda t, y, a: -y,
            lambda t, y, a: jnp.array(0.2),  # additive: independent of y
            jnp.array(1.0),
            0.0,
            1.0,
            key=jr.PRNGKey(0),
            config=cfg,
        )
        assert bool(jnp.isfinite(sol.ys[-1]))

    def test_backsolve_adjoint_gives_gradients(self):
        """Parameters must reach the vector field via `args`, not by closure.

        BacksolveAdjoint is a custom_vjp and cannot differentiate closed-over
        values; passing through `args` is the documented way round it.
        """
        cfg = SolveConfig(dt0=0.01).for_backprop_through_long_solve()

        def loss(scale):
            sol = solve_ode(
                lambda t, y, a: -a * y,
                jnp.array(1.0),
                0.0,
                1.0,
                args=scale,
                config=cfg,
            )
            return jnp.sum(sol.ys**2)

        grad = jax.grad(loss)(jnp.array(1.0))
        assert bool(jnp.isfinite(grad)) and float(grad) != 0.0

    def test_backsolve_adjoint_rejects_closed_over_parameters(self):
        """The documented failure mode should be the documented exception."""
        cfg = SolveConfig(dt0=0.01).for_backprop_through_long_solve()

        def loss(scale):
            sol = solve_ode(
                lambda t, y, a: -scale * y, jnp.array(1.0), 0.0, 1.0, config=cfg
            )
            return jnp.sum(sol.ys**2)

        # CustomVJPException is not exported from a public JAX namespace, so
        # match on the message instead of the type.
        with pytest.raises(Exception, match="custom_vjp"):
            jax.grad(loss)(jnp.array(1.0))
