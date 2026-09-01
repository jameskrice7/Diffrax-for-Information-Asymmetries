# finax

**Neural differential equations and differentiable market microstructure, on JAX.**

[![Tests](https://github.com/jameskrice7/Diffrax-for-Information-Asymmetries/actions/workflows/tests.yml/badge.svg)](https://github.com/jameskrice7/Diffrax-for-Information-Asymmetries/actions/workflows/tests.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)

finax does two things that do not otherwise exist together in one package.

**Differential-equation models that are proper JAX citizens.** Neural ODEs, SDEs,
CDEs, jump SDEs and latent SDEs, built as [Equinox](https://github.com/patrick-kidger/equinox)
modules on top of [Diffrax](https://github.com/patrick-kidger/diffrax). Every model
is a PyTree, so `jit`, `vmap` and `grad` apply to a model instance directly — no
separate parameter dictionary to keep in sync.

**Differentiable market microstructure.** PIN, VPIN, Kyle's lambda and the standard
spread estimators, in pure JAX. As far as we can tell these are the first JAX
implementations: they are vectorised across a cross-section and differentiable with
respect to their inputs, so an estimated information-asymmetry measure can sit
*inside* a larger model rather than being frozen preprocessing.

---

## Install

```bash
pip install finax                 # core: jax, diffrax, equinox, optax, numpy
pip install "finax[all]"          # + pandas, scipy, statsmodels, matplotlib
```

The core is deliberately small. If you only want the microstructure estimators you
do not pull in pandas, matplotlib and statsmodels. A missing optional dependency
raises a `MissingDependencyError` naming the exact `pip install` you need, rather
than an `AttributeError` on `None` three call frames later.

Extras: `data`, `stats`, `viz`, `hf`, `streaming`, `torch`, `tensorflow`, `dev`, `docs`.

## The 60-second tour

### Estimate PIN — correctly, and on a whole cross-section at once

```python
import jax.numpy as jnp
from finax.microstructure import estimate_pin, estimate_pin_panel

# Daily buyer- and seller-initiated trade counts.
result = estimate_pin(buys, sells)
print(result.pin, result.params.alpha, result.at_boundary)

# A whole cross-section as one compiled solve rather than a Python loop.
panel = estimate_pin_panel(buys_2d, sells_2d)   # panel.pin.shape == (2000,)
```

This is a genuine maximum-likelihood fit of the Easley–Kiefer–O'Hara–Paperman model
using the **Lin–Ke (2011) factorization**, so it stays finite at trade counts where
the textbook likelihood overflows to `inf`. It runs the full Yan–Zhang (2012)
5×5×5 grid of starting values in parallel under `vmap`, and it *tells you* when the
optimiser lands on a boundary solution instead of quietly returning a number you
should not trust.

### Neural CDE on irregular, partially observed data

```python
import jax.random as jr
from finax.core import build_control_path, SolveConfig
from finax.models import NeuralCDE

# ts is irregular; ys contains NaN where a channel was not observed.
path = build_control_path(ts, ys)          # time channel + masks + interpolation

model = NeuralCDE.from_hyperparameters(
    input_size=path.n_channels,            # take this from the path, don't count
    hidden_size=64, output_size=1,
    rank=4,                                # low-rank field: ~8x fewer parameters
    key=jr.PRNGKey(0), config=SolveConfig(dt0=0.01),
)
prediction = model(path)
```

`build_control_path` applies the full preprocessing recipe from Kidger et al. (2020)
in one call: time as a channel, cumulative observation masks, fill-forward, and a
choice of backward-Hermite, linear or rectilinear interpolation.

### Exact simulation of classical processes

```python
from finax.processes import Heston, CoxIngersollRoss

log_s, v = Heston(mu=0.03, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7).sample(
    jnp.log(100.0), 0.04, ts=ts, key=jr.PRNGKey(0), n_paths=10_000)
```

Where a transition density is known in closed form the sampler is **exact** — no
discretisation bias at any step size. CIR uses the non-central chi-squared law,
built from a Poisson-mixture-of-gammas so it stays correct even when the Feller
condition fails, the regime where a naive Euler scheme goes negative and returns NaN.

### Check that your solver is actually converging

```python
from finax.diagnostics import strong_order

report = strong_order(simulate, key=jr.PRNGKey(0))
print(report.estimated_order, report.r_squared)
```

A plausible-looking path tells you nothing. This measures the empirical convergence
order and reports an R² so you can tell a real measurement from a meaningless one.
On our test suite it recovers the textbook values: Euler–Maruyama 0.52 on
multiplicative noise, 1.11 on additive noise, Milstein 0.99.

## What's in it

| Module | What it gives you |
| --- | --- |
| `finax.models` | `NeuralODE`, `NeuralSDE`, `NeuralCDE`, `NeuralJumpSDE`, `LatentSDE` — all `eqx.Module` PyTrees |
| `finax.microstructure` | `estimate_pin`, `vpin`, `kyle_lambda`, `amihud_illiquidity`, `roll_spread`, `corwin_schultz_spread`, `lee_ready`, … |
| `finax.core` | `build_control_path`, `pad_ragged`, `SolveConfig` |
| `finax.processes` | GBM, Ornstein–Uhlenbeck, CIR, Heston, Merton — exact samplers where they exist |
| `finax.diagnostics` | `strong_order`, `weak_order`, `martingale_test`, `moment_report` |
| `finax.inference` | `fit` (jitted training loop), losses, closed-form calibration |
| `finax.data` | Loading, cleaning, features, and `to_arrays` / `panel_to_batch` to cross into JAX |
| `finax.evaluation` | Forecast and portfolio metrics, residual diagnostics |

## Design notes

**Everything is a PyTree.** Models are `equinox.Module`s, so this works:

```python
import equinox as eqx
grads = eqx.filter_grad(loss)(model)          # gradients w.r.t. the model itself
paths = jax.vmap(lambda k: model(y0, 0., 1., key=k))(keys)   # batched sampling
```

**Itô by default.** `solve_sde` interprets drift and diffusion in the Itô sense —
the convention finance uses — and defaults to `diffrax.Euler`, which is correct for
any noise structure. Faster solvers each carry an assumption (additive noise,
commutative noise, Stratonovich coefficients) and applying one whose assumption is
violated gives a *silently wrong* answer, not an error. `SolveConfig` documents the
trade-offs and `finax.diagnostics` lets you verify empirically.

**Lévy area is selected for you.** `SolveConfig` reads the solver's
`minimal_levy_area` and configures the Brownian motion to match, so mismatches
cannot happen.

**Failures are loud.** Boundary solutions in PIN are flagged. Convergence studies
report an R². Missing dependencies name their extra. A NaN target is masked rather
than propagated through the gradient.

## Precision

JAX defaults to float32. For maximum-likelihood work and for anything involving
differences of large, nearly equal prices (effective spreads at a $100 price level
lose about three significant digits), enable float64 **before creating any arrays**:

```python
from finax.infrastructure import enable_x64
enable_x64()
```

## Documentation

- [`docs/models.md`](docs/models.md) — neural ODE / SDE / CDE / latent SDE
- [`docs/microstructure.md`](docs/microstructure.md) — PIN, VPIN, liquidity measures
- [`docs/data.md`](docs/data.md) — ingestion, features, and the pandas↔JAX bridge
- [`docs/diagnostics.md`](docs/diagnostics.md) — verifying solver correctness
- [`docs/evaluation.md`](docs/evaluation.md) — metrics and residual tests
- [`docs/infrastructure.md`](docs/infrastructure.md) — devices, precision, reproducibility
- [`CHANGELOG.md`](CHANGELOG.md) — including a migration table from 0.1.x

Every public function carries a runnable doctest; the 101 doctests are executed in CI
alongside 233 unit tests.

## Upgrading from 0.1.x

0.2.0 is a ground-up rewrite and the API changed. See the migration table in the
[changelog](CHANGELOG.md). The most important note:

> The old `probability_of_informed_trading` computed a mean absolute order
> imbalance, not PIN, and the old `vpin` used neither volume buckets nor bulk
> volume classification. **Values from the 0.1.x functions are not comparable to
> the new ones.**

## References

Easley, Kiefer, O'Hara & Paperman (1996), *Liquidity, Information, and Infrequently
Traded Stocks*, Journal of Finance 51(4).
Lin & Ke (2011), *A computing bias in estimating the probability of informed
trading*, Journal of Financial Markets 14(4).
Yan & Zhang (2012), *An improved estimation method and empirical properties of the
probability of informed trading*, Journal of Banking & Finance 36(2).
Easley, López de Prado & O'Hara (2012), *Flow Toxicity and Liquidity in a High
Frequency World*, Review of Financial Studies 25(5).
Kidger, Morrill, Foster & Lyons (2020), *Neural Controlled Differential Equations
for Irregular Time Series*, NeurIPS.
Li, Wong, Chen & Duvenaud (2020), *Scalable Gradients for Stochastic Differential
Equations*, AISTATS.

## Contributing

Issues and pull requests welcome. `pip install -e ".[dev]"`, then `pytest tests`,
`pytest --doctest-modules finax`, and `ruff check finax tests`.

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
