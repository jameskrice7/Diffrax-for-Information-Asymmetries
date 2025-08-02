import pathlib
import sys

import pandas as pd
import pytest

# Ensure package root on path for direct module imports
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Small synthetic dataset for data loader tests."""
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=3),
        "value": [1.0, 2.0, 3.0],
    })
