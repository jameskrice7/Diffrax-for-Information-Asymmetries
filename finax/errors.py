"""Exception types and optional-dependency handling.

The package has a small mandatory core (``jax``, ``diffrax``, ``equinox``,
``optax``, ``numpy``) and a number of optional extras.  Rather than scattering
``try: import x except ImportError: x = None`` across every module -- which
silently turns a missing dependency into an ``AttributeError`` on ``None``
thousands of lines away -- optional imports go through :func:`require`.
"""

from __future__ import annotations

import importlib
from types import ModuleType

__all__ = [
    "FinaxError",
    "MissingDependencyError",
    "ConvergenceError",
    "DataValidationError",
    "ShapeError",
    "require",
]


class FinaxError(Exception):
    """Base class for every exception raised by finax."""


class MissingDependencyError(FinaxError, ImportError):
    """An optional dependency is needed but not installed."""


class ConvergenceError(FinaxError):
    """An iterative estimator failed to converge."""


class DataValidationError(FinaxError, ValueError):
    """Input data does not satisfy a documented precondition."""


class ShapeError(DataValidationError):
    """Array shapes are mutually inconsistent."""


# Maps an importable module name to the extra that provides it, so the error
# message can tell the user exactly what to type.
_EXTRAS: dict[str, str] = {
    "matplotlib": "viz",
    "seaborn": "viz",
    "statsmodels": "stats",
    "arch": "stats",
    "scipy": "stats",
    "pandas": "data",
    "pyarrow": "data",
    "openpyxl": "data",
    "tables": "data",
    "datasets": "hf",
    "transformers": "hf",
    "torch": "torch",
    "tensorflow": "tensorflow",
    "flax": "flax",
    "haiku": "haiku",
}


def require(module: str, *, purpose: str | None = None) -> ModuleType:
    """Import ``module``, or raise an error saying exactly how to install it.

    Parameters
    ----------
    module:
        Importable module name, e.g. ``"scipy"``.
    purpose:
        Short description of what the caller wanted it for. Included in the
        error message so the traceback is self-explanatory.

    Returns
    -------
    The imported module.

    Examples
    --------
    >>> np = require("numpy", purpose="array maths")
    >>> np.__name__
    'numpy'
    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:  # pragma: no cover - depends on the environment
        extra = _EXTRAS.get(module.split(".")[0])
        install = f"pip install 'finax[{extra}]'" if extra else f"pip install {module}"
        reason = f" is required for {purpose} but" if purpose else ""
        raise MissingDependencyError(
            f"'{module}'{reason} is not installed. Install it with: {install}"
        ) from exc
