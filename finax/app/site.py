"""Website-launch preparation helpers for Finax dashboards."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import pandas as pd


@dataclass
class SiteLaunchConfig:
    """Configuration scaffold for a Finax web deployment."""

    app_name: str = "Finax Studio"
    host: str = "0.0.0.0"
    port: int = 8501
    enable_auth: bool = False
    default_dataset_limit: int = 100_000

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def dashboard_payload(df: pd.DataFrame, *, max_rows: int = 2000) -> dict[str, Any]:
    """Create JSON-serializable payloads for web frontends."""

    sample = df.head(max_rows).copy()
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "preview": sample.to_dict(orient="records"),
        "column_types": {k: str(v) for k, v in df.dtypes.items()},
    }
