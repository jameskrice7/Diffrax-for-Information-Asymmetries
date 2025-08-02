"""TensorFlow integration utilities for Finax."""

from __future__ import annotations

from typing import Any, Callable

try:  # pragma: no cover - optional dependency
    import tensorflow as tf  # type: ignore
except Exception:  # pragma: no cover - handled at runtime
    tf = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import jax.numpy as jnp  # type: ignore
    import numpy as np
except Exception:  # pragma: no cover - handled at runtime
    jnp = None  # type: ignore
    np = None  # type: ignore


def keras_to_jax(model: "tf.keras.Model") -> Callable[[Any], Any]:
    """Wrap a Keras model as a JAX-callable function.

    This helper runs the underlying TensorFlow model in inference mode and
    converts the output to ``jax.numpy`` arrays so it can be used inside JAX
    and Diffrax pipelines.
    """

    if tf is None or jnp is None or np is None:  # pragma: no cover - runtime check
        raise ImportError("TensorFlow, NumPy, and JAX are required for this utility.")

    model.trainable = False

    def wrapped(x: Any) -> Any:
        tensor = tf.convert_to_tensor(np.asarray(x))
        result = model(tensor)
        return jnp.asarray(result.numpy())

    return wrapped
