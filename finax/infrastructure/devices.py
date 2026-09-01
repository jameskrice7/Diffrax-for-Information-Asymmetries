"""Device and precision management.

Two things that trip people up on JAX and are worth making explicit.

**float64 is off by default.** JAX silently uses float32 everywhere unless told
otherwise. For most deep learning that is fine; for likelihood optimisation and
for price differences it is not -- see the warning on
:func:`~finax.microstructure.liquidity.effective_spread`. :func:`enable_x64`
turns it on.

**Device placement must happen before the first array is created.** JAX caches
the backend, so configuration changes afterwards have no effect.
"""

from __future__ import annotations

from typing import Any, Literal

import jax

__all__ = [
    "available_devices",
    "default_device",
    "to_device",
    "device_summary",
    "enable_x64",
    "best_platform",
]

Platform = Literal["cpu", "gpu", "tpu"]


def available_devices(platform: Platform | None = None) -> list[Any]:
    """List JAX devices, optionally filtered to one platform.

    Examples
    --------
    >>> devices = available_devices()
    >>> len(devices) >= 1
    True
    """
    devices = list(jax.devices())
    if platform is not None:
        return [d for d in devices if d.platform == platform]
    return devices


def best_platform() -> Platform:
    """Return the fastest available platform: TPU, else GPU, else CPU.

    Examples
    --------
    >>> best_platform() in ("cpu", "gpu", "tpu")
    True
    """
    for platform in ("tpu", "gpu"):
        if available_devices(platform):  # type: ignore[arg-type]
            return platform  # type: ignore[return-value]
    return "cpu"


def default_device() -> Any:
    """Return the first device on the fastest available platform.

    Examples
    --------
    >>> default_device() in available_devices()
    True
    """
    return available_devices(best_platform())[0]


def to_device(x: Any, device: Any | None = None) -> Any:
    """Place a PyTree of arrays on a device.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> moved = to_device({"a": jnp.ones(3)})
    >>> moved["a"].shape
    (3,)
    """
    return jax.device_put(x, device if device is not None else default_device())


def device_summary() -> dict[str, Any]:
    """Describe the current JAX runtime.

    Handy at the top of a notebook to confirm you are on the accelerator you
    think you are, and at the precision you think you are.

    Examples
    --------
    >>> summary = device_summary()
    >>> sorted(summary)
    ['default_backend', 'device_count', 'devices', 'platform', 'x64_enabled']
    """
    devices = available_devices()
    return {
        "platform": best_platform(),
        "device_count": len(devices),
        "devices": [str(d) for d in devices],
        "default_backend": jax.default_backend(),
        "x64_enabled": bool(jax.config.jax_enable_x64),
    }


def enable_x64(enabled: bool = True) -> None:
    """Turn 64-bit precision on or off.

    Call this **before** creating any arrays. Recommended for maximum-likelihood
    work (:mod:`finax.microstructure.pin`, :mod:`finax.inference.calibrate`) and
    for anything involving differences of large, nearly equal prices.

    Examples
    --------
    >>> import jax, jax.numpy as jnp
    >>> original = bool(jax.config.jax_enable_x64)
    >>> enable_x64(True)
    >>> jnp.zeros(1).dtype
    dtype('float64')
    >>> enable_x64(original)
    """
    jax.config.update("jax_enable_x64", enabled)
