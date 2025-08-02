import pathlib
import sys

import pandas as pd
import pandas.testing as tm

from finax.data.ingestion import load_csv, load_json


def test_load_csv(tmp_path, sample_df):
    file = tmp_path / "sample.csv"
    sample_df.to_csv(file, index=False)
    loaded = load_csv(str(file), parse_dates=["date"])
    tm.assert_frame_equal(loaded, sample_df)


def test_load_json(tmp_path, sample_df):
    file = tmp_path / "sample.json"
    sample_df.to_json(file, orient="records", date_format="iso")
    loaded = load_json(str(file))
    loaded["value"] = loaded["value"].astype(float)
    tm.assert_frame_equal(loaded, sample_df)
