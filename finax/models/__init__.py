"""Differentiable models, all of them ``equinox.Module`` PyTrees.

Every model here is a PyTree, which means ``jax.jit``, ``jax.vmap`` and
``equinox.filter_grad`` apply to model instances directly -- no separate
"params" dictionary to thread through your code, and no risk of parameters and
model drifting out of sync.
"""

from .cde import NeuralCDE
from .jump import NeuralJumpSDE
from .latent_sde import LatentSDE, LatentSDEOutput
from .mlp import LowRankTensorField, TensorFieldMLP, VectorFieldMLP
from .ode import NeuralODE
from .sde import NeuralSDE, NoiseType

__all__ = [
    "NeuralODE",
    "NeuralSDE",
    "NeuralCDE",
    "NeuralJumpSDE",
    "LatentSDE",
    "LatentSDEOutput",
    "NoiseType",
    "VectorFieldMLP",
    "TensorFieldMLP",
    "LowRankTensorField",
]
