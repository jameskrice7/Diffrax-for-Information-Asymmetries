"""Training utilities for Finax models."""

from __future__ import annotations

from typing import Any, Callable


def train(model: Callable[..., Any], data: Any, *, steps: int = 100) -> None:
    """Placeholder training loop for models.

    Parameters
    ----------
    model:
        Callable with ``params`` and ``batch`` arguments.
    data:
        Training data or iterator.
    steps:
        Number of optimization steps.
    """
    raise NotImplementedError("Training routine not implemented.")
=======

"""Training utilities for Finax models.

This module exposes two utilities:

``train``
    A minimal gradient-based training loop using Optax optimisers.
``rolling_cv``
    A generator yielding rolling-window train/test splits, useful for
    time-series cross validation.
"""

from collections.abc import Iterable
from typing import Any, Callable, Generator, List, Optional, Tuple

import jax
import optax


def train(
    params: Any,
    loss_fn: Callable[[Any, Any], jax.Array],
    data: Iterable[Any],
    *,
    optimizer: optax.GradientTransformation | None = None,
    steps: int = 100,
    record_history: bool = False,
) -> Tuple[Any, Optional[List[float]]]:
    """Run a simple Optax-based training loop.

    Parameters
    ----------
    params:
        Initial model parameters.
    loss_fn:
        Callable ``(params, batch) -> loss`` returning a scalar.
    data:
        Iterable yielding batches used for optimisation.
    optimizer:
        Optax optimiser. Defaults to :func:`optax.adam(1e-3)`.
    steps:
        Number of optimisation steps.
    record_history:
        If ``True``, return list of losses for each step.

    Returns
    -------
    Tuple[Any, Optional[List[float]]]
        Updated parameters and optional loss history.
    """
    if optimizer is None:
        optimizer = optax.adam(1e-3)

    opt_state = optimizer.init(params)
    history: List[float] | None = [] if record_history else None
    iterator = iter(data)

    for _ in range(steps):
        try:
            batch = next(iterator)
        except StopIteration:  # pragma: no cover - defensive
            iterator = iter(data)
            batch = next(iterator)

        loss, grads = jax.value_and_grad(loss_fn)(params, batch)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        if history is not None:
            history.append(float(loss))

    return params, history


def rolling_cv(
    train_fn: Callable[[Iterable[Any]], Any],
    data: Iterable[Any],
    window: int,
    step: int,
) -> Generator[Tuple[Any, Iterable[Any]], None, None]:
    """Generate rolling-window train/test splits.

    Parameters
    ----------
    train_fn:
        Callable applied to each training split.
    data:
        Sequence of observations.
    window:
        Size of the training window.
    step:
        Step size between windows and the size of the test split.

    Yields
    ------
    Tuple[Any, Iterable[Any]]
        The result of ``train_fn`` on the current training split and the
        corresponding test split.
    """

    data = list(data)
    n = len(data)
    for start in range(0, n - window, step):
        train_split = data[start : start + window]
        test_split = data[start + window : start + window + step]
        yield train_fn(train_split), test_split

