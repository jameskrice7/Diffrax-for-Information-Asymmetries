"""Modeling utilities for Finax."""

from .finance import geometric_brownian_motion, logistic_growth, vasicek_rate
from .financial_processes import (
    JumpDiffusionConfig,
    RegimeSwitchingConfig,
    inject_discontinuities,
    simulate_jump_diffusion,
    simulate_regime_switching_process,
)
from .neural_cde import NeuralCDE
from .neural_jump_sde import NeuralJumpSDE
from .neural_ode import NeuralODE
from .neural_sde import NeuralSDE
from .nsde import MLP, NSDEConfig, NeuralFinancialSDE, estimate_parameter_count
from .sandbox import ProcessSandbox, SandboxScenario
from .simulation import simulate_paths
from .stochastic import brownian_motion, poisson_process
from .training import rolling_cv, train

__all__ = [
    "NeuralODE",
    "NeuralSDE",
    "NeuralCDE",
    "NeuralJumpSDE",
    "NeuralFinancialSDE",
    "NSDEConfig",
    "MLP",
    "estimate_parameter_count",
    "train",
    "rolling_cv",
    "simulate_paths",
    "geometric_brownian_motion",
    "vasicek_rate",
    "logistic_growth",
    "brownian_motion",
    "poisson_process",
    "JumpDiffusionConfig",
    "RegimeSwitchingConfig",
    "simulate_jump_diffusion",
    "simulate_regime_switching_process",
    "inject_discontinuities",
    "ProcessSandbox",
    "SandboxScenario",
]
