"""Utilities for integrating Hugging Face Transformers with Finax."""

from __future__ import annotations

from typing import Any, Callable

import jax.numpy as jnp


def hf_model_to_jax(model_name: str, *, framework: str = "flax") -> Callable[[Any], Any]:
    """Load a Hugging Face model and expose it as a JAX-callable function.

    Parameters
    ----------
    model_name:
        Identifier of the pretrained model on the Hugging Face Hub.
    framework:
        Which backend to use. ``"flax"`` loads a Flax model, while any other value
        attempts to load a PyTorch model and converts outputs to ``jnp.ndarray``.
    """

    if framework == "flax":
        from transformers import FlaxAutoModel  # type: ignore

        model = FlaxAutoModel.from_pretrained(model_name)

        def apply_fn(inputs: Any) -> Any:
            outputs = model(inputs)
            return outputs.last_hidden_state

        return apply_fn

    from transformers import AutoModel  # type: ignore
    import numpy as np
    import torch

    model = AutoModel.from_pretrained(model_name)

    def apply_fn(inputs: Any) -> Any:  # pragma: no cover - conversion wrapper
        with torch.no_grad():
            tensor = torch.from_numpy(np.asarray(inputs))
            result = model(tensor).last_hidden_state
            return jnp.asarray(result.numpy())

    return apply_fn


__all__ = ["hf_model_to_jax"]
