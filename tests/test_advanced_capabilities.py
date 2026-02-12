import pandas as pd
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from finax.app import SiteLaunchConfig, dashboard_payload
from finax.data import (
    densify_sparse_frame,
    load_sparse_csv,
    select_high_dimensional_block,
)
from finax.modeling import simulate_sparse_factor_process, summarize_sparse_structure
from finax.nlp import keyword_intensity, text_feature_frame


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
