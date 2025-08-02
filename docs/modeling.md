# Modeling

Finax wraps Diffrax solvers to build neural ordinary and stochastic differential equation models.

## Neural ODE
- `finax.modeling.neural_ode.NeuralODE` solves systems governed by trainable vector fields via `diffrax.diffeqsolve`.

## Neural SDE
- `finax.modeling.neural_sde.NeuralSDE` simulates paths with learned drift and diffusion terms and supports stochastic integration using JAX PRNG keys.
- `finax.modeling.neural_jump_sde.NeuralJumpSDE` adds a jump component for discontinuous asset price dynamics.

## Neural CDE
- `finax.modeling.neural_cde.NeuralCDE` handles controlled differential equations where the derivative depends on an external control signal.

## Framework Adapters
Finax lets you author models in popular neural-network libraries and call them from JAX/Diffrax code:

- `finax.modeling.tf_integration.keras_to_jax` wraps a Keras model as a JAX function.
- `finax.modeling.torch_integration.torch_module_to_jax` converts a PyTorch `nn.Module` to JAX.
- `finax.modeling.flax_integration.flax_module_to_jax` exposes a Flax module with bound parameters.
- `finax.modeling.haiku_integration.haiku_module_to_jax` wraps a Haiku apply function.
- `finax.modeling.hf_integration.hf_model_to_jax` loads a Hugging Face Transformer model and presents it as a JAX callable.


```python
from finax.modeling.tf_integration import keras_to_jax
jax_fn = keras_to_jax(keras_model)
```

## Training and Simulation
- `finax.modeling.training.train` is a placeholder for future optimization loops.
- `finax.modeling.simulation.simulate_paths` will offer Monte Carlo path generation utilities.

## Predefined Financial Models
- `finax.modeling.finance.geometric_brownian_motion` builds a geometric Brownian motion for asset prices.
- `finax.modeling.finance.vasicek_rate` constructs a Vasicek interest rate model.
- `finax.modeling.finance.logistic_growth` provides a logistic growth ODE for macroeconomic output.
- `finax.modeling.stochastic.brownian_motion` and `poisson_process` generate basic stochastic processes.
- `finax.modeling.flax_finance.FinancialRNN` offers an LSTM block tailored for financial time-series data.

## Visualization
Solutions returned by `NeuralODE.solve` and `NeuralSDE.simulate` can be visualized via
`finax.visualization.plot_solution` or the models' `plot` methods.
- `finax.modeling.flax_finance.FinancialRNN` offers an LSTM block tailored for financial time-series data.

