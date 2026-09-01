"""Shared type aliases.

These are deliberately thin wrappers over :mod:`jaxtyping` so that annotations
document array shapes without imposing a runtime checker on users.  Shape
variables used throughout the package:

``batch``
    Independent samples (paths, firms, stock-days).
``time``
    Points along a trajectory.
``state``
    Dimension of the SDE/ODE state.
``channel``
    Dimension of an observed/control signal.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Union

if TYPE_CHECKING:  # pragma: no cover - typing only
    from jaxtyping import Array, Float, Int, PRNGKeyArray, PyTree, Scalar
else:  # pragma: no cover - runtime fallbacks keep import cost near zero
    Array = Any
    Float = Any
    Int = Any
    PRNGKeyArray = Any
    PyTree = Any
    Scalar = Any

__all__ = [
    "Array",
    "Float",
    "Int",
    "PRNGKeyArray",
    "PyTree",
    "Scalar",
    "ArrayLike",
    "DTypeLike",
]

#: Anything that ``jnp.asarray`` accepts.
ArrayLike = Union[Any]

#: Anything that ``jnp.dtype`` accepts.
DTypeLike = Any
