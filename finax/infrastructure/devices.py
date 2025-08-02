"""JAX device utilities to simplify GPU/TPU usage."""

from __future__ import annotations

from typing import Any, Optional

try:  # pragma: no cover - optional at import time
    import jax
except Exception:  # pragma: no cover - runtime check
    jax = None  # type: ignore


def available_devices() -> list[Any]:
    """Return all JAX devices available to the runtime."""

    if jax is None:  # pragma: no cover - runtime check
        raise ImportError("JAX is required for device inspection.")
    return list(jax.devices())


def default_device() -> Any:
    """Select a default device, preferring GPUs/TPUs when present."""

    devices = available_devices()
    for platform in ("gpu", "tpu"):
        for dev in devices:
            if dev.platform == platform:
                return dev
    return devices[0]


def to_device(x: Any, device: Optional[Any] = None) -> Any:
    """Move ``x`` to the specified JAX device."""

    if jax is None:  # pragma: no cover - runtime check
        raise ImportError("JAX is required for device placement.")
    return jax.device_put(x, device or default_device())
