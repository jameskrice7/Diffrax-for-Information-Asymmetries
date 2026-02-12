"""Plotting utilities using matplotlib and seaborn."""

from __future__ import annotations

from typing import Any, Iterable, Optional

import base64
import io

import numpy as np
import pandas as pd

try:  # pragma: no cover - handled at runtime
    import matplotlib.pyplot as plt  # noqa: F401
    import seaborn as sns  # noqa: F401
except Exception:  # pragma: no cover - the dependencies are optional at import time
    plt = None  # type: ignore
    sns = None  # type: ignore


def _require_viz() -> None:
    if plt is None or sns is None:
        raise ImportError(
            "Matplotlib and Seaborn are required for visualization. Install Finax with the "
            "'visualization' extra, e.g. `pip install finax[visualization]`."
        )


def plot_time_series(data: Any, *, columns: Optional[Iterable[str]] = None, ax: Optional[Any] = None, **kwargs: Any) -> Any:
    """Plot one or more time series.

    Parameters
    ----------
    data:
        Time-indexed data structure such as ``pandas.DataFrame`` or array-like.
    columns:
        Optional iterable specifying which columns to plot (for DataFrame input).
    ax:
        Existing matplotlib axes; if ``None`` one is created.
    """
    _require_viz()
    df = pd.DataFrame(data)
    if columns is not None:
        df = df[list(columns)]
    if ax is None:
        ax = plt.gca()
    sns.lineplot(data=df, ax=ax, **kwargs)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    return ax


def plot_distribution(data: Any, *, ax: Optional[Any] = None, bins: int = 50, **kwargs: Any) -> Any:
    """Plot a distribution histogram using Seaborn."""
    _require_viz()
    if ax is None:
        ax = plt.gca()
    sns.histplot(np.asarray(data), bins=bins, ax=ax, **kwargs)
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    return ax


def plot_training_history(history: Any, *, ax: Optional[Any] = None, **kwargs: Any) -> Any:
    """Plot a training loss curve."""
    _require_viz()
    history_arr = np.asarray(history)
    if ax is None:
        ax = plt.gca()
    sns.lineplot(x=np.arange(len(history_arr)), y=history_arr, ax=ax, **kwargs)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    return ax


def plot_solution(solution: Any, *, ax: Optional[Any] = None, **kwargs: Any) -> Any:
    """Plot the solution from ``diffrax.diffeqsolve`` returned by Finax models."""
    _require_viz()
    if ax is None:
        ax = plt.gca()
    ts = np.asarray(getattr(solution, "ts", None))
    ys = np.asarray(getattr(solution, "ys", None))
    if ts.ndim == 0 or ys.ndim == 0:
        raise ValueError("Solution object must have 'ts' and 'ys' attributes.")
    if ys.ndim == 1:
        sns.lineplot(x=ts, y=ys, ax=ax, **kwargs)
    else:
        df = pd.DataFrame(ys, columns=[f"y{i}" for i in range(ys.shape[1])])
        df.insert(0, "t", ts)
        melted = df.melt(id_vars="t", var_name="variable", value_name="value")
        sns.lineplot(data=melted, x="t", y="value", hue="variable", ax=ax, **kwargs)
    ax.set_xlabel("Time")
    ax.set_ylabel("State")
    return ax


def _coerce_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def summarize_statistics(data: Any) -> dict[str, dict[str, float]]:
    """Summarize numeric columns for quick statistical dashboards."""
    df = pd.DataFrame(data)
    numeric = df.select_dtypes(include=[np.number])
    summary: dict[str, dict[str, float]] = {}
    for col in numeric.columns:
        series = numeric[col].dropna()
        if series.empty:
            continue
        summary[col] = {
            "mean": float(series.mean()),
            "std": float(series.std(ddof=1)),
            "min": float(series.min()),
            "max": float(series.max()),
            "p10": float(series.quantile(0.1)),
            "p90": float(series.quantile(0.9)),
        }
    return summary


def web_series_payload(
    data: Any,
    *,
    columns: Optional[Iterable[str]] = None,
    max_points: int = 2000,
) -> dict[str, Any]:
    """Create lightweight series payloads for web charting."""
    df = pd.DataFrame(data)
    if columns is not None:
        df = df[list(columns)]
    df = df.head(max_points)
    index = [str(v) for v in df.index]
    series = {
        col: [_coerce_scalar(v) for v in df[col].tolist()]
        for col in df.columns
    }
    return {"index": index, "series": series}


def figure_to_base64(fig: Any, *, format: str = "png") -> str:
    """Encode a Matplotlib figure to base64 for web delivery."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format=format, bbox_inches="tight")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    buffer.close()
    return encoded
