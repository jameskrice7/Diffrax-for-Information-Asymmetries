import pandas as pd
import pytest
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from finax.data import (
    load_csv,
    load_json,
    load_parquet,
    load_excel,
    load_hdf5,
    load_sqlite,
)


def sample_df():
    return pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=3, freq="D"),
            "price": [1.0, 2.0, 3.0],
            "volume": [100, 200, 300],
        }
    )


def test_load_csv(tmp_path):
    df = sample_df()
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    loaded = load_csv(path, parse_dates=["date"])
    pd.testing.assert_frame_equal(loaded, df)


def test_load_json(tmp_path):
    df = sample_df()
    path = tmp_path / "data.json"
    df.to_json(path, orient="records")
    loaded = load_json(path)
    # pandas read_json sorts by column; ensure same order
    loaded = loaded[df.columns]
    loaded["date"] = pd.to_datetime(loaded["date"]).astype("datetime64[us]")
    loaded["price"] = loaded["price"].astype(float)
    df = df.copy()
    df["date"] = df["date"].astype("datetime64[us]")
    pd.testing.assert_frame_equal(loaded, df)


def test_load_parquet(tmp_path):
    pytest.importorskip("pyarrow")
    df = sample_df()
    path = tmp_path / "data.parquet"
    df.to_parquet(path)
    loaded = load_parquet(path)
    assert loaded.equals(df)


def test_load_excel(tmp_path):
    pytest.importorskip("openpyxl")
    df = sample_df()
    path = tmp_path / "data.xlsx"
    df.to_excel(path, index=False)
    loaded = load_excel(path)
    assert loaded.equals(df)


def test_load_hdf5(tmp_path):
    pytest.importorskip("tables")
    df = sample_df()
    path = tmp_path / "data.h5"
    df.to_hdf(path, key="data")
    loaded = load_hdf5(path)
    assert loaded.equals(df)


def test_load_sqlite(tmp_path):
    df = sample_df()
    path = tmp_path / "data.db"
    import sqlite3

    with sqlite3.connect(path) as conn:
        df.to_sql("prices", conn, index=False)
    query = "SELECT * FROM prices"
    loaded = load_sqlite(path, query)
    # read_sql returns columns as text; cast date
    loaded["date"] = pd.to_datetime(loaded["date"]).astype("datetime64[us]")
    df = df.copy()
    df["date"] = df["date"].astype("datetime64[us]")
    assert loaded.equals(df)
