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
