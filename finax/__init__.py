"""Finax: Financial modeling tools built on JAX and Diffrax."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING

__all__ = [
    "data",
    "modeling",
    "evaluation",
    "infrastructure",
    "utils",
    "research",
    "visualization",
    "app",
    "nlp",
]


def __getattr__(name: str) -> ModuleType:  # pragma: no cover - thin wrapper
    if name in __all__:
        module = import_module(f"finax.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module 'finax' has no attribute '{name}'")


if TYPE_CHECKING:  # pragma: no cover
    from . import app, data, evaluation, infrastructure, modeling, nlp, research, utils, visualization
