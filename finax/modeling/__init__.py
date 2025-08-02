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
from .finance import geometric_brownian_motion, vasicek_rate, logistic_growth
from .training import train
from .simulation import simulate_paths
from .tf_integration import keras_to_jax
from .torch_integration import torch_module_to_jax
from .flax_integration import flax_module_to_jax
from .haiku_integration import haiku_module_to_jax
from .hf_integration import hf_model_to_jax

try:  # Optional dependency
    from .flax_finance import FinancialRNN, LogReturn
except Exception:  # pragma: no cover - graceful fallback
    FinancialRNN = LogReturn = None
from .stochastic import brownian_motion, poisson_process


__all__ = [
    "NeuralODE",
    "NeuralSDE",
    "NeuralCDE",
    "NeuralJumpSDE",
    "train",
    "rolling_cv",
]
