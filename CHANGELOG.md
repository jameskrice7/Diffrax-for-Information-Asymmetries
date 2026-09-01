# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-09-01

A ground-up rewrite. The public API has changed substantially; see *Removed*
for the migration path.

### Added

#### Models (`finax.models`)
- Every model is now an `equinox.Module`, so `jax.jit`, `jax.vmap` and
  `equinox.filter_grad` apply to model instances directly.
- `LatentSDE` — variational latent SDE after Li et al. (2020), with the
  path-wise KL integrated alongside the state as an augmented coordinate.
- `LowRankTensorField` — factorised neural-CDE vector field that cuts final-layer
  parameters from `width * state * control` to `width * rank * (state + control)`,
  an 8x reduction at typical sizes. Addresses the cubic parameter growth that
  makes deep neural CDEs impractical.
- `NeuralSDE` supports diagonal, scalar and general noise, and uses
  `lineax.DiagonalLinearOperator` for diagonal noise (O(d) rather than O(d²)).
- `NeuralJumpSDE` with an optional martingale-preserving jump compensator.

#### Microstructure (`finax.microstructure`) — new
- `estimate_pin` — PIN by maximum likelihood using the **Lin & Ke (2011)
  factorization**, which is numerically stable at trade counts where the
  textbook likelihood overflows to `inf`.
- Yan & Zhang (2012) initial-value grid, with all 125 starts optimised in
  parallel under `vmap`.
- `estimate_pin_panel` — fits a whole cross-section as one batched solve.
- Boundary solutions are detected and reported via `PINResult.at_boundary`
  rather than silently returned.
- `vpin` — correct volume-bucketed VPIN with bulk volume classification.
- `kyle_lambda`, `amihud_illiquidity`, `roll_spread`,
  `corwin_schultz_spread`, `effective_spread`, `realized_spread`,
  `price_impact`.
- `tick_rule`, `quote_rule`, `lee_ready`, `bulk_volume_classification`.

#### Core (`finax.core`) — new
- `build_control_path` — the full neural-CDE preprocessing recipe (time channel,
  cumulative observation masks, fill-forward, choice of Hermite / linear /
  rectilinear interpolation) in one call.
- `pad_ragged` — stacks variable-length series into `vmap`-able arrays.
- `SolveConfig` — one PyTree bundling solver, adjoint, step-size controller,
  `saveat` and `max_steps`, with presets. Lévy area is selected automatically
  from the solver's `minimal_levy_area`.

#### Diagnostics (`finax.diagnostics`) — new
- `strong_order` / `weak_order` — empirically measure a solver's convergence
  order, with an R² so an untrustworthy fit is visible.
- `martingale_test`, `moment_report`.

#### Processes (`finax.processes`) — new
- `GeometricBrownianMotion`, `OrnsteinUhlenbeck`, `CoxIngersollRoss`, `Heston`,
  `MertonJumpDiffusion`, with **exact** samplers where the transition law is
  known. CIR uses the non-central chi-squared law, built from a
  Poisson-mixture-of-gammas so it stays valid when the Feller condition fails.

#### Inference (`finax.inference`) — new
- `fit` — a `jit`-compiled training loop with minibatching, validation, early
  stopping, gradient clipping and best-model checkpointing.
- Losses: `mse`, `mae`, `gaussian_nll`, `elbo`, `quantile_loss`, all NaN-masking.
- `fit_gbm` / `fit_ou` — closed-form MLE; `fit_mle` for the general case.

#### Other
- `finax.errors` — typed exception hierarchy and a `require()` helper that
  reports the exact `pip install` command for a missing optional dependency.
- `finax.data.frames` — `to_arrays` and `panel_to_batch` bridge pandas to JAX,
  preserving irregular timestamps and missingness.
- `enable_x64`, `device_summary`, `reproducibility_report`.
- `py.typed` marker; the package now ships type information.
- 100 doctests, all executed in CI.

### Fixed
- **`finax.data.ingestion` did not import at all**: `Optional`, `Any`,
  `Callable` and `Iterator` were used but never imported, so every
  `from finax.data import ...` raised `NameError`.
- `NeuralODE.solve` was defined twice; the first definition was dead code that
  also passed a raw function where Diffrax expects a term.
- `NeuralCDE.solve` passed a bare function to `diffeqsolve` instead of a term
  and specified no solver, so it could never have run.
- `ljung_box` used the `return_df=False` tuple API that modern statsmodels has
  removed.
- `load_sqlite` leaked its connection: `with sqlite3.connect(...)` manages the
  transaction, not the handle.
- `rsi` used a simple rolling mean instead of Wilder's smoothing.
- `detect_outliers` used a non-robust z-score, which lets a single large outlier
  inflate the standard deviation enough to mask itself. Now robust by default.
- `FinancialRNN` called `initialize_carry` with a signature removed in current
  Flax, and hard-coded `PRNGKey(0)`.
- `technical_indicator` was an exported function that only raised
  `NotImplementedError`.

### Changed
- License is now **Apache-2.0** (was unlicensed), matching JAX, Diffrax,
  Equinox and Optax.
- `requires-python` raised to `>=3.10`; the codebase already used `X | Y`
  annotations that do not work on 3.8.
- Dependencies are now version-bounded, and the core set is minimal with
  everything else behind extras.
- The default SDE solver is `diffrax.Euler`, which is Itô-correct for any noise
  structure. The previous `EulerHeun` default is a *Stratonovich* method and
  silently returned biased results for state-dependent diffusion.

### Removed
- `finax.app` — an HTML-string dashboard scaffold unrelated to the library's
  purpose.
- `finax.nlp` — a bag-of-words implementation better served by scikit-learn.
- `finax.data.eikon` — a wrapper around a discontinued proprietary terminal API.
- `finax.modeling.{tf,torch,haiku,flax}_integration` — one-line `.numpy()`
  wrappers that broke JAX tracing and could not appear inside a `jit`.
- `finax.modeling.sandbox`, `finax.modeling.highdim_simulation` — superseded by
  `finax.processes` and `finax.diagnostics`.
- `finax.research.asymmetry` — replaced by `finax.microstructure`. The old
  `probability_of_informed_trading` computed a mean absolute order imbalance,
  not PIN; `vpin` used neither volume buckets nor bulk volume classification.
  **Values produced by the old functions are not comparable to the new ones.**

### Migration

| 0.1.0 | 0.2.0 |
| --- | --- |
| `finax.modeling.NeuralODE(f)` | `finax.models.NeuralODE(f)` |
| `model.simulate(y0, t0, t1, key=k)` | `model(y0, t0, t1, key=k)` |
| `finax.research.probability_of_informed_trading` | `finax.microstructure.estimate_pin` |
| `finax.research.vpin(volume, price)` | `finax.microstructure.vpin(prices, volumes)` |
| `finax.modeling.train(params, loss, data)` | `finax.inference.fit(model, loss, data)` |
| `finax.data.daily_ohlcv(df)` | `finax.data.resample_ohlcv(df, "D")` |
| `finax.evaluation.tests` | `finax.evaluation.diagnostics` |

## [0.1.0]

Initial release.
