
"""Training utilities for Finax models.

This module exposes two utilities:

``train``
    A minimal training loop with optional early-stopping and learning rate
    scheduling hooks. Users provide a ``step_fn`` which performs one update and
    returns a metric (typically the loss) for the current iteration::

        >>> data = [1, 2, 3]
        >>> def step_fn(batch, lr, step):
        ...     return batch * lr
        >>> train(step_fn, data, steps=3, lr_schedule=lambda s: 0.1)

``rolling_cv``
    A generator yielding rolling-window train/test splits, useful for
    time-series cross validation::

        >>> data = list(range(10))
        >>> for train_split, test_split in rolling_cv(lambda x: x, data, 4, 2):
        ...     pass

Both utilities are intentionally lightweight to accommodate a variety of model
types and optimisation strategies.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Callable, Generator, Tuple


def train(
    step_fn: Callable[[Any, float, int], float],
    data: Iterable[Any],
    *,
    steps: int = 100,
    early_stopping: Callable[[int, float], bool] | None = None,
    lr_schedule: Callable[[int], float] | None = None,
) -> Tuple[int, float]:
    """Run a simple training loop.

    Parameters
    ----------
    step_fn:
        Callable taking ``(batch, learning_rate, step)`` and returning a metric
        (e.g. loss) for the step.
    data:
        Iterable of batches supplied to ``step_fn``.
    steps:
        Maximum number of optimisation steps.
    early_stopping:
        Optional callable ``(step, metric) -> bool``. If it returns ``True`` the
        loop terminates early.
    lr_schedule:
        Optional callable ``step -> learning_rate`` used to adjust the learning
        rate per iteration.

    Returns
    -------
    Tuple[int, float]
        The last completed step index and its associated metric.
    """

    iterator = iter(data)
    metric = float("nan")
    for step in range(steps):
        try:
            batch = next(iterator)
        except StopIteration:  # pragma: no cover - defensive
            iterator = iter(data)
            batch = next(iterator)

        lr = lr_schedule(step) if lr_schedule is not None else 1.0
        metric = step_fn(batch, lr, step)

        if early_stopping is not None and early_stopping(step, metric):
            break

    return step, metric


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
