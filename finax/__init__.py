
"""Finax: Financial modeling tools built on JAX and Diffrax.

This package provides utilities for loading and cleaning financial data, with
modeling capabilities powered by neural ordinary and stochastic differential
equations. It also offers research utilities for studying information
asymmetry in financial markets and infrastructure helpers to leverage JAX on
CPUs, GPUs, or TPUs.

The subpackages are exposed lazily to avoid importing optional heavy
dependencies during ``finax`` import. Users can access them via ``finax.data``,
``finax.modeling`` and so on without incurring the import cost until needed.
"""


__all__ = [
    "data",
    "modeling",
    "evaluation",
    "infrastructure",
    "utils",
    "research",
    "visualization",
]



def __getattr__(name: str) -> ModuleType:  # pragma: no cover - thin wrapper
    if name in __all__:
        module = import_module(f"finax.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module 'finax' has no attribute '{name}'")


if TYPE_CHECKING:  # pragma: no cover
    from . import data, modeling, evaluation, infrastructure, utils, research, visualization
