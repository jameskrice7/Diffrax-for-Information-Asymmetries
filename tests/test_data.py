"""Tests for the data layer, including the pandas-to-JAX bridge."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from finax.errors import DataValidationError

pd = pytest.importorskip("pandas")

from finax.data import (  # noqa: E402
    align_frames,
    bollinger_bands,
    clip_outliers,
    compute_bid_ask_spread,
    detect_outliers,
    event_flags,
    fill_missing,
    macd,
    panel_to_batch,
    realized_volatility,
    resample_ohlcv,
    returns,
    rsi,
    to_arrays,
    winsorize,
)


class TestPackageImports:
    def test_data_module_imports(self):
        """Regression: finax.data raised NameError on import in 0.1.0."""
        import finax.data

        assert hasattr(finax.data, "load_csv")

    def test_every_public_name_is_importable(self):
        import finax.data

        for name in finax.data.__all__:
            assert hasattr(finax.data, name), f"{name} is exported but missing"


class TestToArrays:
    def test_datetime_index_becomes_elapsed_days(self):
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-03", "2024-01-08"]),
                "price": [10.0, 11.0, 12.0],
            }
        )
        ts, ys = to_arrays(df, time_column="date")
        assert ts.tolist() == [0.0, 2.0, 7.0]
        assert ys.shape == (3, 1)

    def test_missing_values_survive_as_nan(self):
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                "price": [10.0, None],
            }
        )
        _, ys = to_arrays(df, time_column="date")
        assert bool(jnp.isnan(ys[1, 0]))

    def test_time_unit_scaling(self):
        df = pd.DataFrame(
            {
                "t": pd.to_datetime(["2024-01-01 00:00", "2024-01-01 06:00"]),
                "p": [1.0, 2.0],
            }
        )
        ts_days, _ = to_arrays(df, time_column="t", time_unit="D")
        ts_hours, _ = to_arrays(df, time_column="t", time_unit="h")
        assert abs(float(ts_days[1]) - 0.25) < 1e-6
        assert abs(float(ts_hours[1]) - 6.0) < 1e-4

    def test_rejects_unsorted_time(self):
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-03", "2024-01-01"]),
                "price": [1.0, 2.0],
            }
        )
        with pytest.raises(DataValidationError, match="non-decreasing"):
            to_arrays(df, time_column="date")

    def test_rejects_missing_time_column(self):
        df = pd.DataFrame({"price": [1.0]})
        with pytest.raises(DataValidationError, match="not in columns"):
            to_arrays(df, time_column="nope")

    def test_rejects_frame_without_numeric_columns(self):
        df = pd.DataFrame({"date": pd.to_datetime(["2024-01-01"]), "tag": ["a"]})
        with pytest.raises(DataValidationError, match="No numeric"):
            to_arrays(df, time_column="date")


class TestPanelToBatch:
    def test_pads_ragged_entities(self):
        df = pd.DataFrame(
            {
                "ticker": ["A", "A", "A", "B", "B"],
                "date": pd.to_datetime(
                    ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-01", "2024-01-02"]
                ),
                "price": [1.0, 2.0, 3.0, 10.0, 11.0],
            }
        )
        ts, ys, entities = panel_to_batch(df, entity_column="ticker", time_column="date")
        assert ts.shape == (2, 3)
        assert ys.shape == (2, 3, 1)
        assert entities == ["A", "B"]
        assert bool(jnp.isnan(ys[1, 2, 0]))

    def test_padded_times_stay_non_decreasing(self):
        df = pd.DataFrame(
            {
                "id": ["A", "A", "A", "B"],
                "t": pd.to_datetime(
                    ["2024-01-01", "2024-01-05", "2024-01-09", "2024-01-01"]
                ),
                "v": [1.0, 2.0, 3.0, 9.0],
            }
        )
        ts, _, _ = panel_to_batch(df, entity_column="id", time_column="t")
        assert bool(jnp.all(jnp.diff(ts, axis=1) >= 0))


class TestReturns:
    def test_simple_returns(self):
        r = returns(jnp.array([100.0, 110.0, 121.0]), log=False)
        assert bool(jnp.allclose(r, jnp.array([0.1, 0.1]), atol=1e-5))

    def test_log_returns_are_additive(self):
        prices = jnp.array([100.0, 110.0, 121.0, 133.1])
        lr = returns(prices)
        total = jnp.log(prices[-1] / prices[0])
        assert bool(jnp.allclose(jnp.sum(lr), total, atol=1e-5))


class TestCleaning:
    def test_fill_missing_respects_limit(self):
        df = pd.DataFrame({"x": [1.0, np.nan, np.nan, 4.0]})
        assert np.isnan(fill_missing(df, limit=1)["x"].iloc[2])
        assert fill_missing(df)["x"].iloc[2] == 1.0

    def test_robust_outlier_detection_catches_masking(self):
        """A single huge outlier inflates the classical std enough to hide itself."""
        df = pd.DataFrame({"x": [1.0, 1.1, 0.9, 1.05, 0.95, 100.0]})
        assert bool(detect_outliers(df)["x"].iloc[-1])
        assert not bool(detect_outliers(df, robust=False)["x"].iloc[-1])

    def test_clip_outliers_replaces_with_nan(self):
        df = pd.DataFrame({"x": [1.0, 1.1, 0.9, 1.05, 0.95, 100.0]})
        assert np.isnan(clip_outliers(df)["x"].iloc[-1])

    def test_winsorize_clips_to_quantiles(self):
        df = pd.DataFrame({"x": list(range(101))})
        out = winsorize(df, lower=0.05, upper=0.95)
        assert float(out["x"].min()) == 5.0
        assert float(out["x"].max()) == 95.0

    def test_fill_missing_rejects_unknown_method(self):
        with pytest.raises(DataValidationError):
            fill_missing(pd.DataFrame({"x": [1.0]}), method="magic")


class TestFeatures:
    def test_rsi_is_bounded(self, price_frame):
        values = rsi(price_frame["close"]).dropna()
        assert bool(((values >= 0) & (values <= 100)).all())

    def test_rsi_saturates_on_monotone_series(self):
        assert float(rsi(pd.Series(range(1, 60), dtype=float)).iloc[-1]) == 100.0
        assert float(rsi(pd.Series(range(60, 1, -1), dtype=float)).iloc[-1]) == 0.0

    def test_macd_columns(self, price_frame):
        out = macd(price_frame["close"])
        assert list(out.columns) == ["macd", "signal", "hist"]
        # hist is macd minus signal, by definition.
        assert np.allclose(
            (out["macd"] - out["signal"]).to_numpy(), out["hist"].to_numpy()
        )

    def test_macd_rejects_fast_above_slow(self):
        with pytest.raises(DataValidationError):
            macd(pd.Series([1.0, 2.0]), fast=26, slow=12)

    def test_bollinger_ordering(self, price_frame):
        out = bollinger_bands(price_frame["close"]).dropna()
        assert bool((out["upper"] >= out["middle"]).all())
        assert bool((out["middle"] >= out["lower"]).all())

    def test_realized_volatility_recovers_known_sigma(self):
        rng = np.random.default_rng(0)
        sigma = 0.012
        s = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, sigma, 4000))))
        rv = realized_volatility(s, 1000)
        assert abs(float(rv.iloc[-1]) - sigma) < 0.001

    def test_event_flags(self):
        df = pd.DataFrame(
            {"p": [1.0, 2.0, 3.0]},
            index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        )
        events = pd.DataFrame(
            {"date": pd.to_datetime(["2024-01-02"]), "event": ["earnings"]}
        )
        assert event_flags(df, events)["earnings"].tolist() == [0, 1, 0]

    def test_event_flags_requires_datetime_index(self):
        with pytest.raises(DataValidationError, match="DatetimeIndex"):
            event_flags(
                pd.DataFrame({"p": [1.0]}),
                pd.DataFrame({"date": pd.to_datetime(["2024-01-01"]), "event": ["x"]}),
            )


class TestOHLC:
    def test_resample_from_tick_prices(self):
        index = pd.date_range("2024-01-01", periods=48, freq="h")
        ticks = pd.DataFrame({"price": np.arange(48.0), "volume": 1.0}, index=index)
        bars = resample_ohlcv(ticks, "D", price_column="price")
        assert bars[["open", "high", "low", "close"]].iloc[0].tolist() == [
            0.0,
            23.0,
            0.0,
            23.0,
        ]
        assert bars["volume"].tolist() == [24.0, 24.0]

    def test_resample_preserves_ohlc_semantics(self, price_frame):
        weekly = resample_ohlcv(price_frame, "W")
        assert bool((weekly["high"] >= weekly["low"]).all())
        assert bool((weekly["high"] >= weekly["close"]).all())
        assert bool((weekly["low"] <= weekly["close"]).all())

    def test_volume_is_conserved(self, price_frame):
        monthly = resample_ohlcv(price_frame, "ME")
        assert (
            abs(float(monthly["volume"].sum()) - float(price_frame["volume"].sum())) < 1.0
        )

    def test_relative_spread(self):
        df = pd.DataFrame({"bid": [99.0], "ask": [101.0]})
        assert float(compute_bid_ask_spread(df).iloc[0]) == 2.0
        assert abs(float(compute_bid_ask_spread(df, relative=True).iloc[0]) - 0.02) < 1e-9

    def test_missing_columns_raise(self):
        with pytest.raises(DataValidationError, match="missing columns"):
            compute_bid_ask_spread(pd.DataFrame({"bid": [1.0]}))

    def test_resample_without_ohlc_suggests_price_column(self):
        df = pd.DataFrame({"price": [1.0]}, index=pd.to_datetime(["2024-01-01"]))
        with pytest.raises(DataValidationError, match="price_column"):
            resample_ohlcv(df, "D")


class TestAlignFrames:
    def test_inner_join(self):
        a = pd.DataFrame({"x": [1, 2, 3]}, index=[1, 2, 3])
        b = pd.DataFrame({"y": [4, 5]}, index=[2, 3])
        ra, rb = align_frames(a, b)
        assert ra.index.tolist() == [2, 3] == rb.index.tolist()

    def test_outer_join(self):
        a = pd.DataFrame({"x": [1, 2]}, index=[1, 2])
        b = pd.DataFrame({"y": [4, 5]}, index=[3, 4])
        ra, _ = align_frames(a, b, how="outer")
        assert ra.index.tolist() == [1, 2, 3, 4]

    def test_rejects_bad_how(self):
        a = pd.DataFrame({"x": [1]})
        with pytest.raises(DataValidationError):
            align_frames(a, a, how="cross")
