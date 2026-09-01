"""Experiment configuration and reproducibility."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from ..errors import DataValidationError, require

__all__ = ["load_config", "save_config", "seed_everything", "reproducibility_report"]


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a configuration file. Supports JSON, TOML and YAML by extension.

    Examples
    --------
    >>> import json, tempfile, os
    >>> p = os.path.join(tempfile.mkdtemp(), "c.json")
    >>> _ = open(p, "w").write(json.dumps({"lr": 0.01}))
    >>> load_config(p)
    {'lr': 0.01}
    """
    path = Path(path)
    if not path.exists():
        raise DataValidationError(f"Config file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if suffix == ".toml":
        import tomllib

        return tomllib.loads(path.read_text(encoding="utf-8"))
    if suffix in (".yaml", ".yml"):
        yaml = require("yaml", purpose="reading YAML config")
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    raise DataValidationError(
        f"Unsupported config format {suffix!r}; expected .json, .toml, .yaml or .yml."
    )


def save_config(config: dict[str, Any], path: str | Path) -> None:
    """Write a configuration dict to JSON.

    Examples
    --------
    >>> import tempfile, os
    >>> p = os.path.join(tempfile.mkdtemp(), "c.json")
    >>> save_config({"lr": 0.01}, p)
    >>> load_config(p)
    {'lr': 0.01}
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")


def seed_everything(seed: int):
    """Seed Python, NumPy and return a JAX PRNG key.

    JAX's randomness is explicit and does not depend on global state, so the
    returned key is the part that actually matters for JAX code. Python's and
    NumPy's global seeds are set too, since data shuffling and any NumPy-based
    preprocessing do use them.

    Examples
    --------
    >>> import jax.random as jr, jax.numpy as jnp
    >>> a = jr.normal(seed_everything(0), (3,))
    >>> b = jr.normal(seed_everything(0), (3,))
    >>> bool(jnp.array_equal(a, b))
    True
    """
    import random

    import jax
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    return jax.random.PRNGKey(seed)


def reproducibility_report() -> dict[str, Any]:
    """Capture versions and environment for the methods section of a paper.

    Examples
    --------
    >>> report = reproducibility_report()
    >>> "jax" in report["versions"] and "platform" in report
    True
    """
    import platform as platform_module

    import jax

    from .devices import best_platform

    versions: dict[str, str] = {}
    for name in ("jax", "jaxlib", "diffrax", "equinox", "optax", "numpy", "pandas"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except ImportError:
            versions[name] = "not installed"

    from .. import __version__

    return {
        "finax": __version__,
        "python": platform_module.python_version(),
        "os": f"{platform_module.system()} {platform_module.release()}",
        "platform": best_platform(),
        "x64_enabled": bool(jax.config.jax_enable_x64),
        "versions": versions,
        "env": {k: v for k, v in os.environ.items() if k.startswith(("JAX_", "XLA_"))},
    }
