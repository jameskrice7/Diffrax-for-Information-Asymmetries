"""Plotting utilities using matplotlib and seaborn."""

from __future__ import annotations

from typing import Any, Iterable, Optional

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
