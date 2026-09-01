# Diagnostics

Getting a plausible-looking path out of an SDE solver tells you almost nothing.
The step size might be too coarse, or the solver's assumptions might be violated —
a Stratonovich solver applied to Itô coefficients, an additive-noise solver applied
to multiplicative noise — and in every one of those cases the output still looks
smooth and believable. It is just wrong.

`finax.diagnostics` makes that measurable.

## Convergence order

**Strong order** `p` means `E|Y_dt(T) - Y(T)| = O(dt^p)`: pathwise accuracy.
**Weak order** `q` means `|E[f(Y_dt)] - E[f(Y)]| = O(dt^q)`: accuracy of
expectations, which is what matters for pricing and moment matching.

```python
from finax.diagnostics import strong_order
import diffrax, jax.numpy as jnp, jax.random as jr

def simulate(dt, key):
    bm = diffrax.VirtualBrownianTree(0., 1., tol=1e-5, shape=(), key=key)
    terms = diffrax.MultiTerm(
        diffrax.ODETerm(lambda t, y, a: 0.05 * y),
        diffrax.ControlTerm(lambda t, y, a: 0.3 * y, bm))
    return diffrax.diffeqsolve(terms, diffrax.Euler(), 0., 1., dt,
                               jnp.array(1.0), max_steps=None).ys[-1]

report = strong_order(simulate, key=jr.PRNGKey(0))
print(report)
```

`simulate` **must** derive its Brownian path from `key` in a way that is consistent
across step sizes — a `VirtualBrownianTree` built from `key` does exactly this.
Without that consistency you are comparing different random paths and measuring
nothing.

Measured on the test suite:

| Solver | Noise | Measured | Theory |
| --- | --- | --- | --- |
| `Euler` | multiplicative | 0.52 | 0.5 |
| `ItoMilstein` | multiplicative | 0.99 | 1.0 |
| `Euler` | additive | 1.11 | 1.0 |

## Always read `r_squared`

The reference solve is itself numerical, so it sets an error floor. For a
high-order solver the tested step sizes may already be *at* that floor, in which
case the measured "error" is Brownian-tree tolerance and float32 round-off rather
than discretisation error, and the fitted order is meaningless.

A low `r_squared` is the signature. `ShARK` on additive noise reports order 0.31
with R² = 0.60 in exactly this situation — the tool telling you not to believe it,
rather than quietly returning a wrong number.

When it happens: use coarser `step_sizes`, tighten the Brownian tree `tol`, enable
float64, or compare against an exact sampler from `finax.processes`.

## Martingale tests

Many quantities are martingales by construction: a discounted price under the
risk-neutral measure, a compensated jump process, `exp(sigma*W_t - sigma^2*t/2)`.
If the simulation breaks that, something is wrong — usually a missing Itô
correction or a solver applied under the wrong stochastic calculus.

```python
from finax.diagnostics import martingale_test

martingale_test(paths)["passed"]
```

Reports a Bonferroni-corrected z-statistic across time points and the worst
violation. The test suite verifies it accepts `exp(sigma*W_t - sigma^2*t/2)` and
rejects `exp(sigma*W_t)` — precisely the missing-Itô-correction case.

## Moments

```python
from finax.diagnostics import moment_report

moment_report(samples, expected_mean=0.0, expected_variance=1.0,
              expected_kurtosis=3.0)
```

Each moment comes with a Monte Carlo standard error, so a discrepancy can be
judged against sampling noise rather than eyeballed. `|z| > 3` indicates a genuine
mismatch.

## Getting a ground truth

You cannot verify a solver without something to verify against. `finax.processes`
provides exact samplers — GBM, Ornstein–Uhlenbeck, CIR, Merton — with closed-form
moments, which is exactly what `weak_order` and `moment_report` need for
`exact_expectation`.
