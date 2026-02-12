"""Website-launch preparation helpers for Finax dashboards."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Iterable

import html

import pandas as pd

from ..visualization import summarize_statistics, web_series_payload


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


def dashboard_payload(
    df: pd.DataFrame,
    *,
    max_rows: int = 2000,
    include_summary: bool = True,
    include_charts: bool = True,
) -> dict[str, Any]:
    """Create JSON-serializable payloads for web frontends."""

    sample = df.head(max_rows).copy()
    payload = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "preview": sample.to_dict(orient="records"),
        "column_types": {k: str(v) for k, v in df.dtypes.items()},
    }
    if include_summary:
        payload["summary"] = summarize_statistics(sample)
    if include_charts:
        payload["chart_data"] = web_series_payload(sample)
    return payload


def render_dashboard_html(payload: dict[str, Any], *, title: str = "Finax Dashboard") -> str:
    """Render a simple HTML dashboard preview for web launches."""
    columns: Iterable[str] = payload.get("column_types", {}).keys()
    rows: list[dict[str, Any]] = payload.get("preview", [])

    header = "".join(f"<th>{html.escape(str(col))}</th>" for col in columns)
    body = ""
    for row in rows:
        cells = "".join(f"<td>{html.escape(str(row.get(col, '')))}</td>" for col in columns)
        body += f"<tr>{cells}</tr>"

    summary_items = ""
    for col, stats in payload.get("summary", {}).items():
        stat_text = ", ".join(f"{k}: {v:.3f}" for k, v in stats.items())
        summary_items += f"<li><strong>{html.escape(col)}</strong> — {html.escape(stat_text)}</li>"

    summary_html = f"<ul>{summary_items}</ul>" if summary_items else "<p>No summary available.</p>"

    return (
        "<!doctype html>"
        f"<html><head><meta charset='utf-8'><title>{html.escape(title)}</title>"
        "<style>body{font-family:Arial,sans-serif;margin:2rem;}table{border-collapse:collapse;width:100%;}"
        "th,td{border:1px solid #ddd;padding:0.5rem;}th{background:#f5f5f5;text-align:left;}</style>"
        "</head><body>"
        f"<h1>{html.escape(title)}</h1>"
        "<h2>Summary</h2>"
        f"{summary_html}"
        "<h2>Preview</h2>"
        f"<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"
        "</body></html>"
    )
