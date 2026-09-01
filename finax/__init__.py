"""finax: neural differential equations and market microstructure on JAX.

Two things that do not otherwise exist together in one place:

**Differential-equation models that are proper JAX citizens.** Neural ODEs,
SDEs, CDEs, jump SDEs and latent SDEs, all built as ``equinox.Module`` PyTrees
on top of Diffrax, so ``jit``, ``vmap`` and ``grad`` apply to a model instance
directly.

**Differentiable market microstructure.** PIN, VPIN, Kyle's lambda and the
standard spread estimators, in pure JAX -- vectorised across a cross-section and
differentiable with respect to their inputs, so an estimated information-
asymmetry measure can sit inside a larger model instead of being frozen
preprocessing.

Quick start
-----------
>>> import jax.numpy as jnp, jax.random as jr
>>> import finax
>>> gbm = finax.processes.GeometricBrownianMotion(mu=0.05, sigma=0.2)
>>> paths = gbm.sample(jnp.array(100.0), ts=jnp.linspace(0, 1, 253),
...                    key=jr.PRNGKey(0), n_paths=1000)
>>> paths.shape
(1000, 253)

Submodules are imported lazily, so ``import finax`` stays cheap and an optional
dependency missing in one corner of the package never blocks the rest.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING

from .errors import (
    ConvergenceError,
    DataValidationError,
    FinaxError,
    MissingDependencyError,
    ShapeError,
)

__version__ = "0.2.0"

_SUBMODULES = frozenset(
    {
        "core",
        "data",
        "diagnostics",
        "evaluation",
        "inference",
        "infrastructure",
        "microstructure",
        "models",
        "processes",
        "utils",
        "visualization",
    }
)

__all__ = [
    *sorted(_SUBMODULES),
    "__version__",
    "FinaxError",
    "MissingDependencyError",
    "ConvergenceError",
    "DataValidationError",
    "ShapeError",
]


def __getattr__(name: str) -> ModuleType:
    if name in _SUBMODULES:
        module = import_module(f"finax.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module 'finax' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)


if TYPE_CHECKING:  # pragma: no cover
    from . import (
        core,
        data,
        diagnostics,
        evaluation,
        inference,
        infrastructure,
        microstructure,
        models,
        processes,
        utils,
        visualization,
    )
