# Modeling

Finax wraps Diffrax solvers to build neural ordinary and stochastic differential equation models.

## Neural ODE
- `finax.modeling.neural_ode.NeuralODE` solves systems governed by trainable vector fields via `diffrax.diffeqsolve`.

## Neural SDE
- `finax.modeling.neural_sde.NeuralSDE` simulates paths with learned drift and diffusion terms and supports stochastic integration using JAX PRNG keys.

## Framework Adapters
Finax lets you author models in popular neural-network libraries and call them from JAX/Diffrax code:

- `finax.modeling.tf_integration.keras_to_jax` wraps a Keras model as a JAX function.
- `finax.modeling.torch_integration.torch_module_to_jax` converts a PyTorch `nn.Module` to JAX.
- `finax.modeling.flax_integration.flax_module_to_jax` exposes a Flax module with bound parameters.
- `finax.modeling.haiku_integration.haiku_module_to_jax` wraps a Haiku apply function.

```python
from finax.modeling.tf_integration import keras_to_jax
jax_fn = keras_to_jax(keras_model)
```

## Training and Simulation
- `finax.modeling.training.train` is a placeholder for future optimization loops.
- `finax.modeling.simulation.simulate_paths` will offer Monte Carlo path generation utilities.


## Visualization
Solutions returned by `NeuralODE.solve` and `NeuralSDE.simulate` can be visualized via
`finax.visualization.plot_solution` or the models' `plot` methods.

