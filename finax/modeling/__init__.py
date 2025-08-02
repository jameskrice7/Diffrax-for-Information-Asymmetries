"""Modeling utilities for Finax.

The modeling subpackage exposes neural differential equation wrappers and
training helpers. Optional integrations with frameworks such as Flax or PyTorch
are intentionally not imported here to keep dependency requirements minimal at
import time.
"""

from .neural_ode import NeuralODE
from .neural_sde import NeuralSDE
from .neural_cde import NeuralCDE
from .neural_jump_sde import NeuralJumpSDE
from .training import train, rolling_cv

__all__ = [
    "NeuralODE",
    "NeuralSDE",
    "NeuralCDE",
    "NeuralJumpSDE",
    "train",
    "rolling_cv",
]
