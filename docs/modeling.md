# Modeling

Finax provides Diffrax/JAX-first differential equation tooling for financial time series, including neural SDEs with jumps and sandbox experimentation APIs.

## Core differential models
- `finax.modeling.neural_ode.NeuralODE` for ODE dynamics.
- `finax.modeling.neural_sde.NeuralSDE` for drift/diffusion SDE simulation.
- `finax.modeling.neural_jump_sde.NeuralJumpSDE` for discontinuous jump-aware simulation.
- `finax.modeling.neural_cde.NeuralCDE` for controlled differential equations.

## nSDE extensions for finance
- `finax.modeling.nsde.NSDEConfig` defines state size and deep neural architecture.
- `finax.modeling.nsde.MLP` builds scalable dense networks (10,000+ parameters supported).
- `finax.modeling.nsde.NeuralFinancialSDE` runs Euler-style neural SDE simulation with optional jump intensity and scale controls.
- `finax.modeling.nsde.estimate_parameter_count` helps plan model capacity before training.

## Financial process families
- `simulate_jump_diffusion` for Merton-style jump diffusion prices.
- `simulate_regime_switching_process` for two-state volatility switching paths.
- `inject_discontinuities` for synthetic stress testing via abrupt shocks.

## Monte Carlo helpers
- `simulate_paths` repeatedly calls model `simulate`/`solve` routines to assemble Monte Carlo path ensembles.

## Sandbox exploration
- `ProcessSandbox` offers reproducible scenario runners in pure Python.
- `SandboxScenario` standardizes simulation horizon and naming for scripted experiments.

## App foundation
`finax.app` contains forward-compatible scaffolding:
- `AppBlueprint`
- `DataSourceConfig`
- `ModelServiceConfig`

These classes define the contract for a future interactive Finax app without locking implementation details yet.
