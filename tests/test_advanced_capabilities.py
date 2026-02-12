import pandas as pd
import pathlib
import sys

import jax.random as jr

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from finax.app import SiteLaunchConfig, dashboard_payload, render_dashboard_html
from finax.data import (
    densify_sparse_frame,
    load_sparse_csv,
    select_high_dimensional_block,
)
from finax.modeling import simulate_paths, simulate_sparse_factor_process, summarize_sparse_structure
from finax.nlp import keyword_intensity, text_feature_frame
from finax.visualization import summarize_statistics, web_series_payload


def test_sparse_import_roundtrip(tmp_path):
    df = pd.DataFrame({"id": [1, 2, 3], "a": [0, 0, 1], "b": [0, 2, 0]})
    path = tmp_path / "sparse.csv"
    df.to_csv(path, index=False)

    sparse_df = load_sparse_csv(path)
    restored = densify_sparse_frame(sparse_df)
    pd.testing.assert_frame_equal(restored, df)


def test_high_dim_selection_and_simulation():
    sim = simulate_sparse_factor_process(steps=20, dimensions=12, latent_factors=3)
    block = select_high_dimensional_block(sim, min_numeric_columns=10)
    summary = summarize_sparse_structure(block)

    assert block.shape == (20, 12)
    assert summary["columns"] == 12
    assert 0 <= summary["zero_ratio"] <= 1


def test_text_feature_generation():
    df = pd.DataFrame({"notes": ["Strong upward momentum", "Liquidity risk is elevated"]})
    scores = keyword_intensity(df["notes"], ["momentum", "risk"])
    features = text_feature_frame(df, text_column="notes", keywords=["momentum", "risk"])

    assert (scores > 0).all()
    assert "keyword_intensity" in features.columns
    assert "text_length" in features.columns


def test_website_prep_payload():
    cfg = SiteLaunchConfig(app_name="Finax Web")
    assert cfg.to_dict()["app_name"] == "Finax Web"

    df = pd.DataFrame({"x": [1, 2], "text": ["a", "b"]})
    payload = dashboard_payload(df)
    assert payload["rows"] == 2
    assert payload["columns"] == 2
    assert len(payload["preview"]) == 2


def test_web_payload_summary_and_html():
    df = pd.DataFrame({"value": [1.0, 2.0, 3.0], "label": ["a", "b", "c"]})
    payload = dashboard_payload(df)

    assert "summary" in payload
    assert "chart_data" in payload
    assert "value" in payload["summary"]
    assert payload["chart_data"]["series"]["value"] == [1.0, 2.0, 3.0]

    html_output = render_dashboard_html(payload, title="Finax Preview")
    assert "<html>" in html_output
    assert "Finax Preview" in html_output


def test_web_series_payload_and_stats_helpers():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [3, 1, 2]})
    payload = web_series_payload(df, max_points=2)
    stats = summarize_statistics(df)

    assert payload["series"]["x"] == [1, 2]
    assert "y" in stats


def test_simulate_paths_runs():
    class DummyModel:
        def __init__(self):
            self.keys = []

        def simulate(self, *, key, scale=1.0):
            self.keys.append(key)
            return float(scale)

    model = DummyModel()
    outputs = simulate_paths(model, n_paths=3, key=jr.PRNGKey(0), scale=2.0)
    assert outputs == [2.0, 2.0, 2.0]
    assert len(model.keys) == 3
