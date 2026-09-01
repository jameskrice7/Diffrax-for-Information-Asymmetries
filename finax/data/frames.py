"""The bridge from pandas to JAX.

Everything upstream of this module speaks DataFrames; everything downstream
speaks JAX arrays. Getting across that boundary correctly -- preserving
irregular timestamps, keeping missingness as information rather than silently
imputing it, and turning a ragged panel into rectangular batched arrays -- is
where most of the friction in a real workflow lives.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np

from .._typing import Array, Float
from ..errors import DataValidationError, require

__all__ = [
    "to_arrays",
    "panel_to_batch",
    "returns",
    "align_frames",
]


def to_arrays(
    df: Any,
    *,
    time_column: str | None = None,
    columns: list[str] | None = None,
    time_unit: str = "D",
) -> tuple[Float[Array, " time"], Float[Array, "time channel"]]:
    """Convert a DataFrame into ``(ts, ys)`` arrays ready for :mod:`finax.core.paths`.

    Parameters
    ----------
    df:
        Source DataFrame.
    time_column:
        Column holding timestamps. If ``None``, the index is used.
    columns:
        Value columns to extract. Defaults to all numeric columns except the
        time column.
    time_unit:
        Unit for converting datetimes to floats: ``"D"`` days, ``"h"`` hours,
        ``"m"`` minutes, ``"s"`` seconds. Times are measured **from the first
        observation**, not from the epoch -- an epoch-based float64 timestamp in
        seconds is around 1.7e9, and subtracting two of those in float32 loses
        all intraday resolution.

    Returns
    -------
    ``(ts, ys)`` with ``ts`` shaped ``(time,)`` and ``ys`` shaped
    ``(time, channel)``. Missing values are preserved as NaN so that
    :func:`~finax.core.paths.prepare_channels` can encode them as masks.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     "date": pd.to_datetime(["2024-01-01", "2024-01-03", "2024-01-08"]),
    ...     "price": [10.0, None, 12.0],
    ... })
    >>> ts, ys = to_arrays(df, time_column="date")
    >>> ts.tolist()  # days elapsed since the first observation
    [0.0, 2.0, 7.0]
    >>> bool(jnp.isnan(ys[1, 0]))  # missingness survives the conversion
    True
    """
    pd = require("pandas", purpose="DataFrame conversion")
    if not isinstance(df, pd.DataFrame):
        raise DataValidationError(f"Expected a DataFrame, got {type(df).__name__}.")

    if time_column is not None:
        if time_column not in df.columns:
            raise DataValidationError(
                f"time_column {time_column!r} not in columns {list(df.columns)}."
            )
        time_values = df[time_column]
        frame = df.drop(columns=[time_column])
    else:
        time_values = df.index.to_series()
        frame = df

    if columns is not None:
        missing = set(columns) - set(frame.columns)
        if missing:
            raise DataValidationError(f"Columns not found: {sorted(missing)}.")
        frame = frame[columns]
    else:
        frame = frame.select_dtypes(include=[np.number])

    if frame.shape[1] == 0:
        raise DataValidationError("No numeric value columns found.")

    if pd.api.types.is_datetime64_any_dtype(time_values):
        deltas = time_values - time_values.iloc[0]
        ts = deltas.dt.total_seconds().to_numpy(dtype=np.float64)
        divisor = {"D": 86400.0, "h": 3600.0, "m": 60.0, "s": 1.0}
        if time_unit not in divisor:
            raise DataValidationError(
                f"time_unit must be one of {sorted(divisor)}, got {time_unit!r}."
            )
        ts = ts / divisor[time_unit]
    else:
        ts = time_values.to_numpy(dtype=np.float64)
        ts = ts - ts[0]

    if np.any(np.diff(ts) < 0):
        raise DataValidationError(
            "Timestamps must be non-decreasing; sort the frame first."
        )

    return jnp.asarray(ts, jnp.float32), jnp.asarray(
        frame.to_numpy(dtype=np.float64), jnp.float32
    )


def panel_to_batch(
    df: Any,
    *,
    entity_column: str,
    time_column: str,
    value_columns: list[str] | None = None,
    time_unit: str = "D",
) -> tuple[Float[Array, "entity time"], Float[Array, "entity time channel"], list[Any]]:
    """Convert a long-format panel into batched, rectangular arrays.

    Real panels are ragged -- firms enter and exit, tickers have different
    histories. This groups by entity, converts each to ``(ts, ys)``, and pads to
    a common length using :func:`~finax.core.paths.pad_ragged`.

    Parameters
    ----------
    df:
        Long-format frame with one row per (entity, time).
    entity_column:
        Column identifying the entity (e.g. ticker).
    time_column:
        Column holding timestamps.
    value_columns:
        Value columns. Defaults to all numeric columns.
    time_unit:
        Passed to :func:`to_arrays`.

    Returns
    -------
    ``(ts, ys, entities)`` where ``ts`` is ``(entity, time)``, ``ys`` is
    ``(entity, time, channel)`` and ``entities`` lists the entity labels in row
    order.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     "ticker": ["A", "A", "A", "B", "B"],
    ...     "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03",
    ...                             "2024-01-01", "2024-01-02"]),
    ...     "price": [1.0, 2.0, 3.0, 10.0, 11.0],
    ... })
    >>> ts, ys, entities = panel_to_batch(df, entity_column="ticker",
    ...                                   time_column="date")
    >>> ts.shape, ys.shape, entities
    ((2, 3), (2, 3, 1), ['A', 'B'])

    B is one day shorter, so its padded slot is NaN and will be masked as
    unobserved downstream:

    >>> bool(jnp.isnan(ys[1, 2, 0]))
    True
    """
    require("pandas", purpose="panel conversion")
    from ..core.paths import pad_ragged

    if entity_column not in df.columns:
        raise DataValidationError(
            f"entity_column {entity_column!r} not in columns {list(df.columns)}."
        )

    if value_columns is None:
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        value_columns = [c for c in numeric if c not in (entity_column, time_column)]
    if not value_columns:
        raise DataValidationError("No numeric value columns found.")

    entities, series = [], []
    for name, group in df.groupby(entity_column, sort=True):
        group = group.sort_values(time_column)
        ts, ys = to_arrays(
            group,
            time_column=time_column,
            columns=value_columns,
            time_unit=time_unit,
        )
        entities.append(name)
        series.append((ts, ys))

    ts_batch, ys_batch, _ = pad_ragged(series)
    return ts_batch, ys_batch, entities


def returns(
    prices: Float[Array, "... time"],
    *,
    log: bool = True,
    axis: int = -1,
) -> Float[Array, "... time-1"]:
    """Compute returns from a price array.

    Parameters
    ----------
    prices:
        Price array.
    log:
        Log returns (the default) rather than simple returns. Log returns are
        additive across time, which is what makes them the right input to
        anything that sums or integrates increments -- including every model in
        :mod:`finax.models`.
    axis:
        Time axis.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> r = returns(jnp.array([100.0, 110.0, 121.0]), log=False)
    >>> [round(float(v), 4) for v in r]
    [0.1, 0.1]

    Log returns of a constant-growth series are exactly constant:

    >>> lr = returns(jnp.array([100.0, 110.0, 121.0]))
    >>> bool(jnp.allclose(lr[0], lr[1], atol=1e-6))
    True
    """
    prices = jnp.asarray(prices)
    if log:
        return jnp.diff(jnp.log(prices), axis=axis)
    return jnp.diff(prices, axis=axis) / jnp.take(
        prices, jnp.arange(prices.shape[axis] - 1), axis=axis
    )


def align_frames(*frames: Any, how: str = "inner"):
    """Align several DataFrames onto a common index.

    Merging price, volume and quote data from different sources is the usual
    first step of a microstructure study, and silently mismatched indices are
    the usual first bug.

    Parameters
    ----------
    *frames:
        DataFrames to align.
    how:
        Join type: ``"inner"`` (default) or ``"outer"``.

    Returns
    -------
    Tuple of reindexed DataFrames sharing one index.

    Examples
    --------
    >>> import pandas as pd
    >>> a = pd.DataFrame({"x": [1, 2, 3]}, index=[1, 2, 3])
    >>> b = pd.DataFrame({"y": [4, 5]}, index=[2, 3])
    >>> ra, rb = align_frames(a, b)
    >>> ra.index.tolist()
    [2, 3]
    """
    require("pandas", purpose="frame alignment")
    if not frames:
        raise DataValidationError("align_frames needs at least one frame.")
    if how not in ("inner", "outer"):
        raise DataValidationError(f"how must be 'inner' or 'outer', got {how!r}.")

    index = frames[0].index
    for frame in frames[1:]:
        index = (
            index.intersection(frame.index)
            if how == "inner"
            else index.union(frame.index)
        )
    index = index.sort_values()
    return tuple(frame.reindex(index) for frame in frames)
