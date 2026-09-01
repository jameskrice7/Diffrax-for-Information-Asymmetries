"""A training loop that is actually fast.

The original implementation in this package ran ``jax.value_and_grad`` inside a
Python ``for`` loop with no ``jit``, which re-traces the model on every step and
throws away almost all of JAX's advantage. The loop here compiles the update
step once and reuses it, which is typically one to two orders of magnitude
faster.

It also provides the things a real training run needs and a bare loop does not:
minibatching, validation, early stopping, gradient clipping, and best-model
checkpointing in memory.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from .._typing import Array, Float, PRNGKeyArray, PyTree
from ..errors import DataValidationError

__all__ = ["TrainState", "TrainResult", "make_step", "fit", "dataloader"]


class TrainState(eqx.Module):
    """Model and optimiser state travelling together through the loop."""

    model: PyTree
    opt_state: PyTree


class TrainResult(eqx.Module):
    """Outcome of :func:`fit`.

    Attributes
    ----------
    model:
        The best model seen, by validation loss if a validation set was given,
        otherwise the final model.
    train_losses:
        Training loss at each step.
    val_losses:
        Validation loss at each evaluation, or an empty array.
    best_step:
        Step index of :attr:`model`.
    stopped_early:
        Whether early stopping triggered.
    """

    model: PyTree
    train_losses: Float[Array, " step"]
    val_losses: Float[Array, " eval"]
    best_step: int = eqx.field(static=True)
    stopped_early: bool = eqx.field(static=True)


def dataloader(
    arrays: tuple[Array, ...],
    *,
    batch_size: int,
    key: PRNGKeyArray,
    shuffle: bool = True,
) -> Iterator[tuple[Array, ...]]:
    """Yield shuffled minibatches from a tuple of equal-length arrays.

    Loops forever, reshuffling each epoch, so the caller controls the number of
    steps rather than the number of epochs.

    Examples
    --------
    >>> import jax.numpy as jnp, jax.random as jr
    >>> x = jnp.arange(10.0)[:, None]
    >>> loader = dataloader((x,), batch_size=4, key=jr.PRNGKey(0))
    >>> batch, = next(loader)
    >>> batch.shape
    (4, 1)
    """
    if not arrays:
        raise DataValidationError("dataloader needs at least one array.")
    n = arrays[0].shape[0]
    if any(a.shape[0] != n for a in arrays):
        raise DataValidationError(
            f"All arrays must share a leading dimension; got "
            f"{[int(a.shape[0]) for a in arrays]}."
        )
    if batch_size > n:
        raise DataValidationError(f"batch_size={batch_size} exceeds dataset size {n}.")

    indices = np.arange(n)
    while True:
        if shuffle:
            key, subkey = jax.random.split(key)
            perm = np.asarray(jax.random.permutation(subkey, n))
        else:
            perm = indices
        for start in range(0, n - batch_size + 1, batch_size):
            batch_idx = perm[start : start + batch_size]
            yield tuple(a[batch_idx] for a in arrays)


def make_step(
    loss_fn: Callable[..., Float[Array, ""]],
    optimiser: optax.GradientTransformation,
) -> Callable[[TrainState, tuple[Any, ...]], tuple[TrainState, Float[Array, ""]]]:
    """Build a compiled single training step.

    ``loss_fn(model, *batch) -> scalar``. Gradients are taken only with respect
    to inexact-array leaves via ``equinox.filter_grad``, so integer and static
    fields of the model are left alone automatically.

    Examples
    --------
    >>> import equinox as eqx, jax.numpy as jnp, jax.random as jr, optax
    >>> model = eqx.nn.Linear(2, 1, key=jr.PRNGKey(0))
    >>> opt = optax.adam(1e-2)
    >>> step = make_step(lambda m, x, y: jnp.mean((jax.vmap(m)(x) - y) ** 2), opt)
    >>> state = TrainState(model=model, opt_state=opt.init(
    ...     eqx.filter(model, eqx.is_inexact_array)))
    >>> x, y = jnp.ones((4, 2)), jnp.ones((4, 1))
    >>> state, loss = step(state, (x, y))
    >>> bool(jnp.isfinite(loss))
    True
    """

    @eqx.filter_jit
    def step(state: TrainState, batch: tuple[Any, ...]):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(state.model, *batch)
        updates, opt_state = optimiser.update(
            grads, state.opt_state, eqx.filter(state.model, eqx.is_inexact_array)
        )
        model = eqx.apply_updates(state.model, updates)
        return TrainState(model=model, opt_state=opt_state), loss

    return step


def fit(
    model: PyTree,
    loss_fn: Callable[..., Float[Array, ""]],
    data: tuple[Array, ...],
    *,
    steps: int = 1000,
    batch_size: int | None = None,
    learning_rate: float = 1e-3,
    optimiser: optax.GradientTransformation | None = None,
    key: PRNGKeyArray | None = None,
    validation_data: tuple[Array, ...] | None = None,
    validate_every: int = 50,
    patience: int | None = None,
    gradient_clip: float | None = 1.0,
    verbose: bool = False,
) -> TrainResult:
    """Train ``model`` by minimising ``loss_fn``.

    Parameters
    ----------
    model:
        Any ``equinox.Module`` -- every model in :mod:`finax.models` qualifies.
    loss_fn:
        ``loss_fn(model, *batch) -> scalar``.
    data:
        Tuple of arrays sharing a leading batch axis.
    steps:
        Number of gradient steps.
    batch_size:
        Minibatch size. ``None`` uses the full dataset every step.
    learning_rate:
        Used only when ``optimiser`` is not given.
    optimiser:
        An Optax transformation. Defaults to Adam, wrapped in gradient clipping.
    key:
        PRNG key for shuffling. Required when ``batch_size`` is given.
    validation_data:
        Held-out arrays, evaluated every ``validate_every`` steps.
    validate_every:
        Validation interval.
    patience:
        Stop after this many validations without improvement. Requires
        ``validation_data``.
    gradient_clip:
        Global gradient-norm clip. Defaults to 1.0, which matters more for
        differential-equation models than for ordinary networks: an untrained
        vector field can produce a very stiff system whose gradients are
        enormous, and one unclipped step can push parameters somewhere the
        solver cannot integrate at all.
    verbose:
        Print progress.

    Returns
    -------
    A :class:`TrainResult`.

    Examples
    --------
    Fitting a linear model to a known relationship:

    >>> import equinox as eqx, jax, jax.numpy as jnp, jax.random as jr
    >>> key = jr.PRNGKey(0)
    >>> x = jr.normal(key, (256, 2))
    >>> true_w = jnp.array([2.0, -3.0])
    >>> y = (x @ true_w)[:, None]
    >>> model = eqx.nn.Linear(2, 1, key=jr.PRNGKey(1))
    >>> loss = lambda m, xb, yb: jnp.mean((jax.vmap(m)(xb) - yb) ** 2)
    >>> res = fit(model, loss, (x, y), steps=2000, learning_rate=0.05)
    >>> bool(res.train_losses[-1] < 1e-3)
    True
    >>> bool(jnp.allclose(res.model.weight[0], true_w, atol=0.05))
    True

    Early stopping on a validation set:

    >>> res = fit(model, loss, (x[:200], y[:200]),
    ...           validation_data=(x[200:], y[200:]),
    ...           steps=5000, patience=3, validate_every=25,
    ...           learning_rate=0.05)
    >>> res.val_losses.shape[0] >= 1
    True
    """
    if patience is not None and validation_data is None:
        raise DataValidationError("patience requires validation_data.")
    if batch_size is not None and key is None:
        raise DataValidationError("batch_size requires a PRNG key for shuffling.")

    if optimiser is None:
        optimiser = optax.adam(learning_rate)
        if gradient_clip is not None:
            optimiser = optax.chain(optax.clip_by_global_norm(gradient_clip), optimiser)

    state = TrainState(
        model=model,
        opt_state=optimiser.init(eqx.filter(model, eqx.is_inexact_array)),
    )
    step_fn = make_step(loss_fn, optimiser)

    if batch_size is None:
        batches: Iterator[tuple[Array, ...]] = iter(lambda: data, None)
    else:
        batches = dataloader(data, batch_size=batch_size, key=key)

    evaluate = eqx.filter_jit(loss_fn)

    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val = jnp.inf
    best_model = model
    best_step = 0
    since_improvement = 0
    stopped_early = False

    for step_index in range(steps):
        state, loss = step_fn(state, next(batches))
        train_losses.append(float(loss))

        should_validate = validation_data is not None and (
            (step_index + 1) % validate_every == 0 or step_index == steps - 1
        )
        if should_validate:
            val = float(evaluate(state.model, *validation_data))
            val_losses.append(val)
            if val < best_val:
                best_val, best_model, best_step = val, state.model, step_index
                since_improvement = 0
            else:
                since_improvement += 1
            if verbose:
                print(f"step {step_index + 1}: train={float(loss):.6g} val={val:.6g}")
            if patience is not None and since_improvement >= patience:
                stopped_early = True
                break
        elif verbose and (step_index + 1) % validate_every == 0:
            print(f"step {step_index + 1}: train={float(loss):.6g}")

    if validation_data is None:
        best_model = state.model
        best_step = len(train_losses) - 1

    return TrainResult(
        model=best_model,
        train_losses=jnp.asarray(train_losses),
        val_losses=jnp.asarray(val_losses),
        best_step=best_step,
        stopped_early=stopped_early,
    )
