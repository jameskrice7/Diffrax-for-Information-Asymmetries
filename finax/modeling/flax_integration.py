"""Flax integration utilities for Finax."""

from __future__ import annotations

from typing import Any, Callable

try:  # pragma: no cover - optional dependency
    import flax.linen as nn  # type: ignore
except Exception:  # pragma: no cover
    nn = None  # type: ignore


def flax_module_to_jax(module: "nn.Module", params: Any) -> Callable[[Any], Any]:
    """Wrap a Flax module with bound parameters as a JAX-callable function."""

    if nn is None:  # pragma: no cover - runtime check
        raise ImportError("Flax is required for this utility.")

    def wrapped(x: Any) -> Any:
        return module.apply(params, x)

    return wrapped
