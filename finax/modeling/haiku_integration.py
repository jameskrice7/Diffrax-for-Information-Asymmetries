"""Haiku integration utilities for Finax."""

from __future__ import annotations

from typing import Any, Callable

try:  # pragma: no cover - optional dependency
    import haiku as hk  # type: ignore
except Exception:  # pragma: no cover
    hk = None  # type: ignore


def haiku_module_to_jax(apply_fn: Callable[..., Any], params: Any, state: Any | None = None) -> Callable[[Any], Any]:
    """Wrap a Haiku module apply function as a JAX-callable function."""

    if hk is None:  # pragma: no cover - runtime check
        raise ImportError("Haiku is required for this utility.")

    def wrapped(x: Any) -> Any:
        return apply_fn(params, state, None, x) if state is not None else apply_fn(params, None, x)

    return wrapped
