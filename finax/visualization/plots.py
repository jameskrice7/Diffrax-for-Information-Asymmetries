"""Plotting helpers for paths, ensembles and diagnostics.

Matplotlib only -- seaborn is not required. Every function takes an optional
``ax`` and returns it, so plots compose into subplot grids.
"""

from __future__ import annotations

import base64
import io
from typing import Any

import numpy as np

from .._typing import Array, Float
from ..errors import DataValidationError, require

__all__ = [
    "plot_paths",
    "plot_fan_chart",
    "plot_training_history",
    "plot_convergence",
    "plot_solution",
    "figure_to_base64",
]


def _plt():
    return require("matplotlib.pyplot", purpose="plotting")


def plot_paths(
    ts: Float[Array, " time"],
    paths: Float[Array, "path time"],
    *,
    ax: Any = None,
    max_paths: int = 50,
    alpha: float = 0.3,
    color: str = "C0",
    label: str | None = None,
):
    """Plot a sample of Monte Carlo paths.

    Only ``max_paths`` are drawn: a few thousand overlapping lines render as an
    illegible solid block and take a long time to rasterise. Use
    :func:`plot_fan_chart` to show the whole ensemble.

    Examples
    --------
    >>> import matplotlib; matplotlib.use("Agg")
    >>> import jax.numpy as jnp, jax.random as jr
    >>> ts = jnp.linspace(0, 1, 100)
    >>> paths = jr.normal(jr.PRNGKey(0), (200, 100)).cumsum(axis=1)
    >>> ax = plot_paths(ts, paths)
    >>> len(ax.lines)
    50
    """
    plt = _plt()
    ts = np.asarray(ts)
    paths = np.atleast_2d(np.asarray(paths))
    if paths.shape[-1] != ts.shape[0]:
        raise DataValidationError(
            f"paths last axis ({paths.shape[-1]}) must match len(ts) ({ts.shape[0]})."
        )

    if ax is None:
        _, ax = plt.subplots()
    for i, path in enumerate(paths[:max_paths]):
        ax.plot(
            ts, path, color=color, alpha=alpha, lw=0.8, label=label if i == 0 else None
        )
    ax.set_xlabel("Time")
    ax.set_ylabel("State")
    return ax


def plot_fan_chart(
    ts: Float[Array, " time"],
    paths: Float[Array, "path time"],
    *,
    ax: Any = None,
    quantiles: tuple[float, ...] = (0.05, 0.25, 0.5, 0.75, 0.95),
    color: str = "C0",
    label: str | None = None,
):
    """Plot the predictive distribution of an ensemble as nested quantile bands.

    The right way to show a stochastic forecast: the median as a line, with
    symmetric quantile pairs shaded at increasing opacity. Conveys the whole
    distribution without drawing a single path.

    Parameters
    ----------
    quantiles:
        Levels to shade, symmetric about the median.

    Examples
    --------
    >>> import matplotlib; matplotlib.use("Agg")
    >>> import jax.numpy as jnp, jax.random as jr
    >>> ts = jnp.linspace(0, 1, 50)
    >>> paths = jr.normal(jr.PRNGKey(0), (500, 50)).cumsum(axis=1)
    >>> ax = plot_fan_chart(ts, paths)
    >>> len(ax.collections)  # two shaded bands for five quantiles
    2
    """
    plt = _plt()
    ts = np.asarray(ts)
    paths = np.atleast_2d(np.asarray(paths))
    if ax is None:
        _, ax = plt.subplots()

    levels = sorted(quantiles)
    values = np.quantile(paths, levels, axis=0)

    n_pairs = len(levels) // 2
    for i in range(n_pairs):
        ax.fill_between(
            ts,
            values[i],
            values[-(i + 1)],
            color=color,
            alpha=0.15 + 0.15 * i,
            lw=0,
        )
    if len(levels) % 2 == 1:
        ax.plot(ts, values[n_pairs], color=color, lw=1.5, label=label)

    ax.set_xlabel("Time")
    ax.set_ylabel("State")
    return ax


def plot_training_history(
    train_losses: Float[Array, " step"],
    val_losses: Float[Array, " eval"] | None = None,
    *,
    ax: Any = None,
    log_scale: bool = True,
    validate_every: int = 1,
):
    """Plot training and validation loss curves.

    Log-scaled by default: training losses routinely span several orders of
    magnitude, and on a linear axis everything after the first few steps is a
    flat line against the axis.

    Examples
    --------
    >>> import matplotlib; matplotlib.use("Agg")
    >>> import jax.numpy as jnp
    >>> ax = plot_training_history(jnp.exp(-jnp.linspace(0, 5, 100)))
    >>> ax.get_yscale()
    'log'
    """
    plt = _plt()
    train_losses = np.asarray(train_losses)
    if ax is None:
        _, ax = plt.subplots()

    ax.plot(np.arange(len(train_losses)), train_losses, label="train", lw=1.0)
    if val_losses is not None and len(val_losses):
        val_losses = np.asarray(val_losses)
        x = (np.arange(1, len(val_losses) + 1)) * validate_every
        ax.plot(x, val_losses, label="validation", lw=1.5, marker="o", ms=3)
        ax.legend()

    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    return ax


def plot_convergence(report: Any, *, ax: Any = None, reference_orders=(0.5, 1.0)):
    """Plot a :class:`~finax.diagnostics.ConvergenceReport` on log-log axes.

    Reference slopes are drawn so the measured order can be read off against the
    theoretical ones by eye.

    Examples
    --------
    >>> import matplotlib; matplotlib.use("Agg")
    >>> import jax.numpy as jnp
    >>> from finax.diagnostics import ConvergenceReport
    >>> r = ConvergenceReport(step_sizes=jnp.array([0.1, 0.05, 0.025]),
    ...                       errors=jnp.array([0.1, 0.07, 0.05]),
    ...                       estimated_order=0.5, r_squared=0.99)
    >>> ax = plot_convergence(r)
    >>> ax.get_xscale()
    'log'
    """
    plt = _plt()
    if ax is None:
        _, ax = plt.subplots()

    dts = np.asarray(report.step_sizes)
    errors = np.asarray(report.errors)
    ax.loglog(dts, errors, "o-", label=f"measured (order {report.estimated_order:.2f})")

    for order in reference_orders:
        scaled = errors[0] * (dts / dts[0]) ** order
        ax.loglog(dts, scaled, "--", lw=0.8, alpha=0.6, label=f"order {order}")

    ax.set_xlabel("Step size")
    ax.set_ylabel("Error")
    ax.legend()
    return ax


def plot_solution(solution: Any, *, ax: Any = None, **kwargs: Any):
    """Plot a Diffrax :class:`~diffrax.Solution`.

    Examples
    --------
    >>> import matplotlib; matplotlib.use("Agg")
    >>> import diffrax, jax.numpy as jnp
    >>> sol = diffrax.diffeqsolve(
    ...     diffrax.ODETerm(lambda t, y, a: -y), diffrax.Tsit5(), 0.0, 1.0, 0.01,
    ...     jnp.array([1.0]), saveat=diffrax.SaveAt(ts=jnp.linspace(0, 1, 20)))
    >>> ax = plot_solution(sol)
    >>> len(ax.lines)
    1
    """
    plt = _plt()
    ts = getattr(solution, "ts", None)
    ys = getattr(solution, "ys", None)
    if ts is None or ys is None:
        raise DataValidationError("Solution must have 'ts' and 'ys' attributes.")

    ts = np.asarray(ts)
    ys = np.asarray(ys)
    if ax is None:
        _, ax = plt.subplots()

    if ys.ndim == 1:
        ax.plot(ts, ys, **kwargs)
    else:
        for i in range(ys.shape[1]):
            ax.plot(ts, ys[:, i], label=f"y{i}", **kwargs)
        if ys.shape[1] > 1:
            ax.legend()

    ax.set_xlabel("Time")
    ax.set_ylabel("State")
    return ax


def figure_to_base64(fig: Any, *, img_format: str = "png", dpi: int = 100) -> str:
    """Encode a Matplotlib figure as a base64 string for embedding in HTML.

    Examples
    --------
    >>> import matplotlib; matplotlib.use("Agg")
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> _ = ax.plot([0, 1], [0, 1])
    >>> len(figure_to_base64(fig)) > 100
    True
    """
    buffer = io.BytesIO()
    fig.savefig(buffer, format=img_format, bbox_inches="tight", dpi=dpi)
    try:
        return base64.b64encode(buffer.getvalue()).decode("ascii")
    finally:
        buffer.close()
