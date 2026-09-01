"""Turning irregular, partially-observed time series into Diffrax control paths.

This is the piece of neural-CDE work that is fiddly to get right and that every
project ends up reimplementing.  The standard recipe (Kidger et al., *Neural
Controlled Differential Equations for Irregular Time Series*, NeurIPS 2020, and
Kidger, *On Neural Differential Equations*, 2021 ch. 3) is:

1. Append **time as a channel**, so the path is never constant in ``t`` and the
   CDE can tell how much wall-clock elapsed between observations.
2. Append **observational masks**, so "missing" is a signal the model can read
   rather than an artefact of imputation.  Kidger recommends cumulative-count
   masks, which additionally encode *how long* a channel has been missing.
3. **Fill forward** remaining NaNs, and back-fill any leading NaNs, so the
   interpolation is well-defined.
4. Interpolate with a scheme appropriate to the task.

On step 4 the choice matters more than people expect:

``"hermite"`` (default)
    Backward-differences Hermite cubic. Smooth (so adaptive solvers behave) and
    only looks one observation ahead, which keeps it usable in offline settings
    without the wild overshoot of natural cubic splines.
``"linear"``
    Continuous but non-differentiable at knots. Cheap; fine with fixed-step
    solvers. Also only looks one observation ahead.
``"rectilinear"``
    Steps in time then in value, so the path *never* depends on a future
    observation. This is the correct choice for genuinely online/streaming
    prediction. Doubles the number of knots.

Everything here is pure JAX and safe under ``jit``/``vmap``.
"""

from __future__ import annotations

from typing import Literal

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp

from .._typing import Array, Float
from ..errors import ShapeError

__all__ = [
    "InterpolationMethod",
    "ControlPath",
    "fill_forward",
    "pad_ragged",
    "prepare_channels",
    "build_control_path",
]

InterpolationMethod = Literal["hermite", "linear", "rectilinear"]


def fill_forward(ys: Float[Array, "time channel"]) -> Float[Array, "time channel"]:
    """Replace each NaN with the most recent non-NaN value in the same channel.

    Leading NaNs (those with no earlier observation) are left as NaN; use
    :func:`prepare_channels`, which additionally back-fills them.

    Implemented as a ``lax.scan`` so it is ``jit``/``vmap``-friendly and runs in
    O(time) rather than the O(time^2) of a broadcasted mask comparison.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> fill_forward(jnp.array([[1.0], [jnp.nan], [3.0]]))
    Array([[1.],
           [1.],
           [3.]], dtype=float32)
    """
    ys = jnp.asarray(ys)
    if ys.ndim != 2:
        raise ShapeError(f"fill_forward expects (time, channel), got shape {ys.shape}.")

    def step(carry: Array, y: Array) -> tuple[Array, Array]:
        filled = jnp.where(jnp.isnan(y), carry, y)
        return filled, filled

    _, out = jax.lax.scan(step, jnp.full(ys.shape[1:], jnp.nan, ys.dtype), ys)
    return out


def _fill_backward(ys: Float[Array, "time channel"]) -> Float[Array, "time channel"]:
    """Fill each leading NaN with the first later non-NaN value."""
    return fill_forward(ys[::-1])[::-1]


def pad_ragged(
    series: list[tuple[Float[Array, " time"], Float[Array, "time channel"]]],
    *,
    fill_value: float = jnp.nan,
) -> tuple[Float[Array, "batch time"], Float[Array, "batch time channel"], Array]:
    """Stack variable-length series into rectangular, ``vmap``-able arrays.

    Real panels are ragged: firms list at different dates, tick data arrives at
    different rates. JAX needs rectangular arrays, so shorter series are padded
    to the longest length.

    Time padding repeats the final timestamp rather than inserting ``fill_value``.
    Diffrax requires ``ts`` to be non-decreasing, and repeated final timestamps
    produce zero-length intervals that contribute nothing to a CDE integral --
    exactly the semantics you want for "this series already ended".

    Parameters
    ----------
    series:
        List of ``(ts, ys)`` pairs. ``ts`` has shape ``(time,)`` and must be
        increasing; ``ys`` has shape ``(time, channel)``. All series must share
        the same channel count.
    fill_value:
        Value written into the padded region of ``ys``. The default ``nan`` means
        :func:`prepare_channels` will mark the padding as unobserved.

    Returns
    -------
    ``(ts, ys, lengths)`` where ``ts`` is ``(batch, time)``, ``ys`` is
    ``(batch, time, channel)`` and ``lengths`` is ``(batch,)`` holding each
    series' true length.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> a = (jnp.array([0.0, 1.0, 2.0]), jnp.array([[1.0], [2.0], [3.0]]))
    >>> b = (jnp.array([0.0, 1.0]), jnp.array([[4.0], [5.0]]))
    >>> ts, ys, lengths = pad_ragged([a, b])
    >>> ts.shape, ys.shape, lengths.tolist()
    ((2, 3), (2, 3, 1), [3, 2])
    >>> ts[1].tolist()  # final timestamp repeated, not NaN
    [0.0, 1.0, 1.0]
    """
    if not series:
        raise ShapeError("pad_ragged requires at least one series.")

    arrays = [(jnp.asarray(t), jnp.asarray(y)) for t, y in series]
    for i, (t, y) in enumerate(arrays):
        if t.ndim != 1:
            raise ShapeError(f"series[{i}] ts must be 1-D, got shape {t.shape}.")
        if y.ndim != 2:
            raise ShapeError(f"series[{i}] ys must be 2-D, got shape {y.shape}.")
        if t.shape[0] != y.shape[0]:
            raise ShapeError(
                f"series[{i}] has {t.shape[0]} timestamps but {y.shape[0]} observations."
            )

    channels = {y.shape[1] for _, y in arrays}
    if len(channels) != 1:
        raise ShapeError(
            f"All series must have the same channel count, got {sorted(channels)}."
        )
    n_channels = channels.pop()
    max_len = max(t.shape[0] for t, _ in arrays)

    ts_out, ys_out, lengths = [], [], []
    for t, y in arrays:
        pad = max_len - t.shape[0]
        lengths.append(t.shape[0])
        if pad:
            t = jnp.concatenate([t, jnp.full((pad,), t[-1], t.dtype)])
            y = jnp.concatenate([y, jnp.full((pad, n_channels), fill_value, y.dtype)])
        ts_out.append(t)
        ys_out.append(y)

    return jnp.stack(ts_out), jnp.stack(ys_out), jnp.asarray(lengths)


def prepare_channels(
    ts: Float[Array, " time"],
    ys: Float[Array, "time channel"],
    *,
    append_time: bool = True,
    append_mask: bool = True,
    cumulative_mask: bool = True,
) -> Float[Array, "time out_channel"]:
    """Apply the standard neural-CDE channel augmentation to one series.

    Parameters
    ----------
    ts:
        Increasing timestamps, shape ``(time,)``.
    ys:
        Observations, shape ``(time, channel)``. NaN means "not observed".
    append_time:
        Prepend ``ts`` as channel 0. Strongly recommended: without it a CDE
        cannot distinguish a one-second gap from a one-year gap, and a path that
        is constant over an interval contributes nothing at all to the integral.
    append_mask:
        Append one channel per input channel recording observedness.
    cumulative_mask:
        With ``append_mask``, append the running *count* of observations rather
        than a 0/1 flag. The increment of a cumulative count is the 0/1 flag, and
        since a CDE integrates against path increments this gives the model the
        indicator for free while also encoding time-since-last-observation.
        This is the form recommended in Kidger (2021).

    Returns
    -------
    Augmented array with no NaNs, shape
    ``(time, append_time + channel + append_mask * channel)``.

    Notes
    -----
    Channels that are *never* observed are filled with zeros; there is no
    information to propagate and leaving NaNs would poison the solve.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> ts = jnp.array([0.0, 1.0, 2.0])
    >>> ys = jnp.array([[1.0], [jnp.nan], [3.0]])
    >>> prepare_channels(ts, ys)
    Array([[0., 1., 1.],
           [1., 1., 1.],
           [2., 3., 2.]], dtype=float32)
    """
    ts = jnp.asarray(ts)
    ys = jnp.asarray(ys)
    if ts.ndim != 1:
        raise ShapeError(f"ts must be 1-D, got shape {ts.shape}.")
    if ys.ndim != 2:
        raise ShapeError(f"ys must be (time, channel), got shape {ys.shape}.")
    if ts.shape[0] != ys.shape[0]:
        raise ShapeError(f"ts has {ts.shape[0]} entries but ys has {ys.shape[0]} rows.")

    observed = ~jnp.isnan(ys)

    filled = _fill_backward(fill_forward(ys))
    # Any channel that is never observed is still NaN; zero it out.
    filled = jnp.nan_to_num(filled, nan=0.0)

    parts = []
    if append_time:
        parts.append(ts[:, None].astype(filled.dtype))
    parts.append(filled)
    if append_mask:
        mask = observed.astype(filled.dtype)
        parts.append(jnp.cumsum(mask, axis=0) if cumulative_mask else mask)
    return jnp.concatenate(parts, axis=-1)


class ControlPath(eqx.Module):
    """An interpolated control path, ready to drive a :class:`~finax.models.NeuralCDE`.

    A thin, serialisable ``eqx.Module`` wrapper over a Diffrax interpolation. It
    is a PyTree, so it can be closed over inside ``jit``, batched with ``vmap``,
    and stored in a dataclass alongside model parameters.

    Attributes
    ----------
    interpolation:
        The underlying Diffrax path object.
    t0, t1:
        Integration limits, taken from the (possibly rectilinear-expanded)
        timestamps.
    n_channels:
        Width of the control signal, i.e. the ``input_size`` a
        :class:`~finax.models.NeuralCDE` must be built with.
    """

    interpolation: diffrax.AbstractPath
    t0: Float[Array, ""]
    t1: Float[Array, ""]
    n_channels: int = eqx.field(static=True)

    def evaluate(self, t0, t1=None, left: bool = True):
        """Evaluate the path at ``t0``, or its increment over ``[t0, t1]``."""
        return self.interpolation.evaluate(t0, t1, left=left)

    def derivative(self, t, left: bool = True):
        """Evaluate ``dX/dt`` at ``t``."""
        return self.interpolation.derivative(t, left=left)


def build_control_path(
    ts: Float[Array, " time"],
    ys: Float[Array, "time channel"],
    *,
    method: InterpolationMethod = "hermite",
    append_time: bool = True,
    append_mask: bool = True,
    cumulative_mask: bool = True,
    already_prepared: bool = False,
) -> ControlPath:
    """Build a :class:`ControlPath` from one irregular, partially-observed series.

    This is the one-call path from raw ``(ts, ys)`` to something a neural CDE can
    integrate against. For a batch, ``vmap`` it (see Examples).

    Parameters
    ----------
    ts:
        Increasing timestamps, shape ``(time,)``.
    ys:
        Observations, shape ``(time, channel)``; NaN means "not observed".
    method:
        ``"hermite"``, ``"linear"`` or ``"rectilinear"``. See the module
        docstring for how to choose.
    append_time, append_mask, cumulative_mask:
        Forwarded to :func:`prepare_channels`.
    already_prepared:
        Skip :func:`prepare_channels` because ``ys`` has been augmented already.
        Useful when the same augmentation is shared across many paths.

    Returns
    -------
    A :class:`ControlPath` whose ``n_channels`` tells you the ``input_size`` to
    give the model.

    Examples
    --------
    Single series with a gap:

    >>> import jax.numpy as jnp
    >>> ts = jnp.array([0.0, 1.0, 2.5, 4.0])
    >>> ys = jnp.array([[1.0], [jnp.nan], [3.0], [2.0]])
    >>> path = build_control_path(ts, ys)
    >>> path.n_channels  # time + value + cumulative mask
    3
    >>> bool(path.t0 == 0.0), bool(path.t1 == 4.0)
    (True, True)

    A whole batch at once:

    >>> import jax
    >>> batch_ts = jnp.stack([ts, ts])
    >>> batch_ys = jnp.stack([ys, ys + 1.0])
    >>> paths = jax.vmap(build_control_path)(batch_ts, batch_ys)
    >>> paths.t1.shape
    (2,)
    """
    ts = jnp.asarray(ts)
    ys = jnp.asarray(ys)

    if not already_prepared:
        ys = prepare_channels(
            ts,
            ys,
            append_time=append_time,
            append_mask=append_mask,
            cumulative_mask=cumulative_mask,
        )
    n_channels = ys.shape[-1]

    if method == "rectilinear":
        ts, ys = diffrax.rectilinear_interpolation(ts, ys)
        interpolation: diffrax.AbstractPath = diffrax.LinearInterpolation(ts, ys)
    elif method == "linear":
        interpolation = diffrax.LinearInterpolation(ts, ys)
    elif method == "hermite":
        coeffs = diffrax.backward_hermite_coefficients(ts, ys)
        interpolation = diffrax.CubicInterpolation(ts, coeffs)
    else:  # pragma: no cover - guarded by Literal
        raise ValueError(
            f"Unknown interpolation method {method!r}; "
            "expected 'hermite', 'linear' or 'rectilinear'."
        )

    return ControlPath(
        interpolation=interpolation,
        t0=ts[0],
        t1=ts[-1],
        n_channels=n_channels,
    )
