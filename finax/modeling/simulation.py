"""Simulation utilities for Finax."""

from __future__ import annotations

from typing import Any, Iterable

import inspect
import time

try:  # pragma: no cover - optional at import time
    import jax.random as jr
except ImportError:  # pragma: no cover
    jr = None  # type: ignore


def _coerce_keys(key: Any | None, n_paths: int) -> Iterable[Any | None]:
    if key is None:
        if jr is None:
            return [None] * n_paths
        key = jr.PRNGKey(time.time_ns() % (2**32))
    if jr is None:
        return [key] * n_paths
    return jr.split(key, n_paths)


def simulate_paths(model: Any, *, n_paths: int, key: Any | None = None, **kwargs: Any) -> list[Any]:
    """Run a lightweight Monte Carlo simulation helper.

    The helper calls ``model.simulate`` when available (passing JAX keys if
    required), otherwise falls back to ``model.solve`` for deterministic models.
    """
    if n_paths <= 0:
        raise ValueError("n_paths must be a positive integer.")

    if hasattr(model, "simulate"):
        simulate = model.simulate
        params = inspect.signature(simulate).parameters
        needs_key = "key" in params
        if needs_key and jr is None and key is None:
            raise ImportError("JAX is required to provide PRNG keys for simulation.")
        outputs = []
        for path_key in _coerce_keys(key, n_paths):
            call_kwargs = dict(kwargs)
            if needs_key:
                call_kwargs["key"] = path_key
            outputs.append(simulate(**call_kwargs))
        return outputs

    if hasattr(model, "solve"):
        return [model.solve(**kwargs) for _ in range(n_paths)]

    raise AttributeError("Model must define a 'simulate' or 'solve' method.")
