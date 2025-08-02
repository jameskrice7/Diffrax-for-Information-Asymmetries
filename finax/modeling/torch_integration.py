"""PyTorch integration utilities for Finax."""

from __future__ import annotations

from typing import Any, Callable

try:  # pragma: no cover - optional dependency
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import jax.numpy as jnp  # type: ignore
    import numpy as np
except Exception:  # pragma: no cover
    jnp = None  # type: ignore
    np = None  # type: ignore


def torch_module_to_jax(module: "torch.nn.Module") -> Callable[[Any], Any]:
    """Wrap a PyTorch module as a JAX-callable function.

    Parameters
    ----------
    module:
        A ``torch.nn.Module`` set to evaluation mode. The wrapper converts input
        arrays to torch tensors and returns the output as ``jax.numpy`` arrays.
    """

    if torch is None or jnp is None or np is None:  # pragma: no cover - runtime check
        raise ImportError("PyTorch, NumPy, and JAX are required for this utility.")

    module.eval()

    def wrapped(x: Any) -> Any:
        with torch.no_grad():
            tensor = torch.as_tensor(np.asarray(x))
            result = module(tensor)
        return jnp.asarray(result.numpy())

    return wrapped
