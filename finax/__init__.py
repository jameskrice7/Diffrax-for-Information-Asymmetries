"""Finax: Financial modeling tools built on JAX and Diffrax.

This package provides utilities for loading and cleaning financial data,
with modeling capabilities powered by neural ordinary and stochastic differential equations. It also offers research utilities for studying
information asymmetry in financial markets and infrastructure helpers to
leverage JAX on CPUs, GPUs, or TPUs.
"""

from . import data, modeling, evaluation, infrastructure, utils, research

__all__ = [
    "data",
    "modeling",
    "evaluation",
    "infrastructure",
    "utils",
]
