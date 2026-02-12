"""Training utilities for Finax models."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Callable, Generator

import jax


def train(
    params: Any,
    loss_fn: Callable[[Any, Any], jax.Array],
    data: Iterable[Any],
    *,
    optimizer: Any | None = None,
    steps: int = 100,
    record_history: bool = False,
):
    """Run a simple gradient-based training loop.

    Uses Optax when available, otherwise applies vanilla SGD updates.
    """
    history = [] if record_history else None
    iterator = iter(data)

    if optimizer is None:
        try:
            import optax  # type: ignore

            optimizer = optax.adam(1e-3)
            use_optax = True
        except Exception:  # pragma: no cover - optional dependency
            optimizer = 1e-3
            use_optax = False
    else:
        use_optax = hasattr(optimizer, "update") and hasattr(optimizer, "init")

    if use_optax:
        opt_state = optimizer.init(params)

    for _ in range(steps):
        try:
            batch = next(iterator)
        except StopIteration:  # pragma: no cover
            iterator = iter(data)
            batch = next(iterator)

        loss, grads = jax.value_and_grad(loss_fn)(params, batch)
        if use_optax:
            import optax  # type: ignore

            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
        else:
            lr = float(optimizer)
            params = jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)

        if history is not None:
            history.append(float(loss))

    return params, history


def rolling_cv(
    train_fn: Callable[[Iterable[Any]], Any],
    data: Iterable[Any],
    window: int,
    step: int,
) -> Generator[tuple[Any, Iterable[Any]], None, None]:
    """Generate rolling-window train/test splits."""
    data = list(data)
    n = len(data)
    for start in range(0, n - window, step):
        train_split = data[start : start + window]
        test_split = data[start + window : start + window + step]
        yield train_fn(train_split), test_split
