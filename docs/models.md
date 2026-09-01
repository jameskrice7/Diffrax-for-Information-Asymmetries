# Models

Every model in `finax.models` is an `equinox.Module`, which means it is a JAX
PyTree. Parameters live *in* the model, so there is no separate parameter
dictionary to thread through your code and no way for the two to drift apart.

```python
import equinox as eqx, jax

grads = eqx.filter_grad(loss)(model)                         # w.r.t. the model
paths = jax.vmap(lambda k: model(y0, 0., 1., key=k))(keys)   # batched sampling
fast = eqx.filter_jit(model)                                 # compiled
```

`eqx.filter_*` splits the PyTree into differentiable arrays and everything else,
so integer and static fields are handled automatically.

## Choosing a model

| Your data | Model |
| --- | --- |
| Regular, deterministic dynamics | `NeuralODE` |
| Regular, stochastic dynamics | `NeuralSDE` |
| **Irregular or partially observed** | `NeuralCDE` |
| Latent state to infer from observations | `LatentSDE` |
| Discontinuities / fat tails | `NeuralJumpSDE` |

## NeuralODE

```python
from finax.models import NeuralODE
from finax.core import SolveConfig

model = NeuralODE.from_hyperparameters(
    state_size=4, width=64, depth=2, key=key, config=SolveConfig(dt0=0.01))

y1 = model(y0, 0.0, 1.0)                    # terminal state
ys = model(y0, 0.0, 1.0, ts=ts)             # trajectory
sol = model.solve(y0, 0.0, 1.0)             # full diffrax.Solution
```

The default vector field uses `softplus` hidden activations and a `tanh` output.
Both choices matter. `softplus` is smooth, unlike ReLU whose kinks make adaptive
solvers reject steps and cost higher-order solvers their convergence order. The
`tanh` output bounds the drift, which is the single most effective guard against
the stiffness blow-up that makes an untrained neural ODE take enormous numbers of
solver steps.

## NeuralSDE

```python
from finax.models import NeuralSDE

model = NeuralSDE.from_hyperparameters(state_size=4, key=key)
paths = model.sample(y0, 0.0, 1.0, key=key, n_paths=1000)   # vmapped internally
```

### Noise structure

| `noise_type` | Diffusion returns | Meaning |
| --- | --- | --- |
| `"diagonal"` (default) | shape of `y` | Each state has its own Brownian driver |
| `"scalar"` | shape of `y` | One Brownian motion drives everything |
| `"general"` | `(state, noise)` matrix | Correlated noise across states |

Diagonal noise uses `lineax.DiagonalLinearOperator`, which is O(d) rather than the
O(d²) of a materialised diagonal matrix.

### Positive diffusion

`from_hyperparameters` passes the diffusion through `softplus` and adds
`diffusion_floor`. A diffusion that can reach zero makes the log-likelihood
singular, and that is the usual cause of NaN losses when training a neural SDE.

## NeuralCDE

The right model for irregular and partially observed data. It integrates against
the data path itself:

```
z_t = z_0 + ∫ f_θ(z_s) dX_s
```

so observation times enter natively rather than being imputed onto a grid.

```python
from finax.core import build_control_path
from finax.models import NeuralCDE

path = build_control_path(ts, ys)          # see docs/data.md
model = NeuralCDE.from_hyperparameters(
    input_size=path.n_channels,            # take it from the path
    hidden_size=64, output_size=1, rank=4, key=key)

prediction = model(path)                   # terminal readout
sequence = model(path, ts=query_times)     # prediction at each time
hidden = model.hidden_states(path, ts=query_times)
```

### The cubic parameter growth problem

A CDE vector field must output a `(hidden, control)` **matrix**, so a dense final
layer needs `width × hidden × control` parameters. This is the known limitation
that "makes deep or stacked Neural CDE architectures impractical".

`rank=r` switches to `LowRankTensorField`, which factorises `f_θ(z) = U(z) V(z)ᵀ`
and needs `width × r × (hidden + control)` instead. At `hidden=128`, `control=64`,
`width=128`, `r=4` that is 1,073,280 parameters down to 132,096 — an 8× reduction,
verified in the test suite.

Set `rank` whenever `hidden_size × input_size` exceeds a few thousand.

### Why condition `z_0` on the data

The CDE integral only sees *increments*, so a fixed `z_0` cannot see absolute
levels. `NeuralCDE` maps the first observation through a linear layer to get `z_0`,
which restores that information.

## LatentSDE

Variational inference for stochastic dynamics, after Li et al. (2020). A prior SDE
and a data-conditioned posterior SDE **share a diffusion term**, which makes the KL
between their path measures tractable by Girsanov:

```
KL(q‖p) = E_q ∫ ½‖u‖² ds,    u = (f_q - f_p) / g
```

The implementation integrates that KL as an extra coordinate of the state
alongside `z`, so it comes back exactly, not as an approximation.

```python
from finax.models import LatentSDE
from finax.inference import elbo, gaussian_nll

model = LatentSDE.from_hyperparameters(
    input_size=path.n_channels, latent_size=8, output_size=1,
    context_size=8, key=key)

out = model(path, ts=ts, key=key)
loss = elbo(gaussian_nll(out.outputs, targets), out.kl, beta=0.1)

# Unconditional generation from the trained prior:
samples = model.sample_prior(z0, 0.0, 1.0, key=key, ts=ts)
```

Start with `beta` well below 1. At `beta=1` a latent SDE tends to collapse to the
prior before the decoder has learned anything useful.

**Why this matters for information asymmetry.** The latent state is an unobserved
process inferred from observable data — structurally the same problem as
recovering a latent informed-trading intensity from prices and volumes. The
quantity of interest is never measured directly, only its noisy imprint. The
posterior drift is where that signal lives.

## NeuralJumpSDE

```
dz = f dt + g dW + h dN,    N ~ Poisson(λ)
```

```python
from finax.models import NeuralJumpSDE

model = NeuralJumpSDE.from_hyperparameters(
    state_size=1, key=key, intensity=2.0, compensate=True)
paths = model.sample(y0, n_steps=252, dt=1/252, key=key, n_paths=1000)
```

With `compensate=True` the drift is corrected by `λ·E[h]`, so jumps add tail risk
without shifting the expected return — the Merton convention.

**Gradient caveat.** Jump counts are discrete and not reparameterisable. Gradients
reach the drift, diffusion and jump-size networks, but not the intensity through
the counting process. Use `expected_jump_compensator`, which is differentiable in
the intensity, or a score-function estimator.

## Solve configuration

`SolveConfig` bundles solver, step size, adjoint, `saveat` and `max_steps` into one
PyTree:

```python
from finax.core import SolveConfig

cfg = (SolveConfig(dt0=0.001)
       .with_steps_for(0.0, 10.0)      # size max_steps for the horizon
       .saving_dense())                # continuously queryable output
model = NeuralSDE(drift, diffusion, config=cfg)
y = model(y0, 0., 1., key=key, config=other_cfg)   # override per call
```

### Itô vs Stratonovich

`solve_sde` interprets coefficients in the **Itô** sense — what finance means by
`dS = μS dt + σS dW` — and defaults to `diffrax.Euler`, correct for any noise
structure.

| Solver | Strong order | Requires |
| --- | --- | --- |
| `Euler` | 0.5 | nothing (default) |
| `ItoMilstein` | 1.0 | commutative noise |
| `ShARK` | 1.5 | additive noise (`for_additive_noise()`) |
| `Heun` | 1.0 | **Stratonovich** coefficients |

Applying a solver whose assumption is violated gives a silently wrong answer, not
an error. Use [`finax.diagnostics`](diagnostics.md) to verify empirically.

Lévy area is read from the solver's `minimal_levy_area` and configured
automatically, so mismatches cannot happen.

### Memory

`for_backprop_through_long_solve()` switches to `BacksolveAdjoint` for O(1)-memory
gradients. **It cannot differentiate values closed over by the vector field** — it
is a `custom_vjp`. Parameters must arrive via `args`, or as fields of an
`eqx.Module` vector field, which is how every model here is built.

## Training

```python
from finax.inference import fit, mse

result = fit(model, lambda m, x, y: mse(jax.vmap(m)(x), y), (x, y),
             steps=5000, batch_size=64, key=key,
             validation_data=(x_val, y_val), patience=10)
result.model            # best by validation loss, not the last iterate
result.train_losses
```

The update step is compiled once and reused. Gradient clipping defaults to a
global norm of 1.0, which matters more here than for ordinary networks: an
untrained vector field can produce a very stiff system whose gradients are
enormous, and one unclipped step can push parameters somewhere the solver cannot
integrate at all.
