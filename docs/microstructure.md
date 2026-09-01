# Market microstructure

Measures of information asymmetry and illiquidity, in pure JAX. Every function
here is `jit`-compilable, `vmap`-able across a cross-section, and differentiable
with respect to its inputs.

## Why differentiable?

The usual workflow computes PIN once as a fixed number and feeds it into a
regression. That treats an *estimate* as data, and it stops you doing anything
that requires gradients to flow back through the estimator.

Because `pin_log_likelihood` is a JAX function, `jax.grad` works through it. An
estimated PIN can therefore sit inside a larger model — as a feature in a neural
SDE, say — and the whole thing trains end to end. No other PIN implementation
supports this.

## The pipeline

```text
raw trades
    │
    │  classification.lee_ready / tick_rule
    ▼
signed trades ──► aggregate_daily_counts ──► pin.estimate_pin
    │
    │  vpin.volume_bars
    ▼
volume bars ──► vpin.vpin
```

## PIN

The Easley–Kiefer–O'Hara–Paperman (1996) model treats each day as a mixture:

| Branch | Probability | Buy rate | Sell rate |
| --- | --- | --- | --- |
| No event | `1 - α` | `ε_b` | `ε_s` |
| Bad news | `α δ` | `ε_b` | `ε_s + μ` |
| Good news | `α (1 - δ)` | `ε_b + μ` | `ε_s` |

and the headline statistic is the informed share of order flow:

```
PIN = α μ / (α μ + ε_b + ε_s)
```

```python
from finax.microstructure import estimate_pin

result = estimate_pin(buys, sells)
result.pin              # the statistic
result.params.alpha     # the structural parameters
result.log_likelihood
result.at_boundary      # read this
```

### Numerical stability

The naive likelihood contains `ε_b ** B` and `exp(-ε_b)` as separate factors. For
an actively traded stock `B` runs to five or six figures, both overflow, and the
estimate silently collapses. This is the best-documented failure mode in the PIN
literature.

finax uses the **Lin & Ke (2011) factorization**, which pulls out the common factor
and evaluates the remaining three-component mixture through `logsumexp`. Nothing is
exponentiated before it is safe to do so:

```python
params = PINParams(alpha=0.3, delta=0.5, mu=5000., eps_b=150_000., eps_s=150_000.)
pin_log_likelihood(params, jnp.array([200_000.]), jnp.array([190_000.]))
# finite — the textbook formula is inf here
```

Ersan & Alici (2016) show the earlier EHO factorization is systematically biased
downward where Lin–Ke is not.

### Starting values and local optima

The PIN likelihood is multimodal, so a single arbitrary start lands in a local
optimum often enough to matter. `estimate_pin` runs the **Yan & Zhang (2012)**
grid: it sweeps `(α, δ, γ)` and back-solves the arrival rates from the sample
means, so every start reproduces the observed average order flow.

All 125 starts are optimised **in parallel under `vmap`**. The cost is one batched
solve, not 125 sequential ones.

### Boundary solutions

Lin–Ke solves overflow but is known to produce boundary solutions (`α` or `δ`
driven to 0 or 1) more often. finax detects and reports these:

```python
if result.at_boundary:
    ...  # treat the PIN as unreliable
```

Reporting beats hiding: a boundary solution is a diagnostic, not a number.

### Panels

```python
panel = estimate_pin_panel(buys, sells)   # (n_series, n_days) in
panel.pin.shape                           # (n_series,)
```

The whole cross-section is one compiled computation rather than a Python loop.

How much that buys you depends on the hardware. On CPU the gain is modest — about
1.5x for 64 series in our measurements — because the inner 125-start `vmap`
already saturates the available cores. The batching matters most on GPU or TPU,
where there is parallelism left to exploit, and it always avoids paying Python
loop and dispatch overhead per series.

## VPIN

Easley, López de Prado & O'Hara (2012) replace maximum likelihood with a direct
order-imbalance measure in **volume time**. The algorithm has three steps and
skipping any of them gives something that is not VPIN:

1. **Volume bars** — equal-*volume* buckets, not equal-time.
2. **Bulk volume classification** — split each bucket's volume using a normal CDF
   of the standardised price change, rather than hard-labelling trades.
3. **Rolling imbalance** over a window of buckets.

```python
from finax.microstructure import volume_bars, vpin

prices, volumes = volume_bars(trade_prices, trade_sizes,
                              bucket_volume=total / (50 * n_days),
                              n_buckets=50 * n_days)
toxicity = vpin(prices, volumes, window=50)
```

Because BVC is a smooth CDF rather than a `sign`, VPIN here is differentiable —
unlike the tick and quote rules, whose gradient is zero almost everywhere.

## Trade classification

| Function | Needs | Notes |
| --- | --- | --- |
| `tick_rule` | prices | Carries zero ticks forward, which is the majority of observations in liquid names |
| `quote_rule` | prices, bid, ask | Returns 0 at the midpoint |
| `lee_ready` | prices, bid, ask | Quote rule, falling back to tick at the midpoint. The field standard |
| `bulk_volume_classification` | bar prices, volumes | Fractional, smooth, differentiable |

Classification is not innocuous: Boehmer, Grammig & Theissen (2007) show
misclassification biases PIN downward, so the choice changes the headline result.

## Liquidity and price impact

Ordered by data requirements, most to least demanding:

| Function | Needs | Measures |
| --- | --- | --- |
| `price_impact` | trades, quotes, future quotes | Adverse selection — the closest trade-level analogue of PIN |
| `effective_spread` | trades, quotes | What the trade actually cost |
| `realized_spread` | trades, future quotes | What the liquidity provider kept |
| `kyle_lambda` | signed order flow | Price impact per unit of flow |
| `roll_spread` | trade prices | Spread from bid-ask bounce autocovariance |
| `amihud_illiquidity` | daily returns, dollar volume | Return per dollar traded |
| `corwin_schultz_spread` | daily high, low | Spread from daily OHLC alone |

`effective_spread - realized_spread = price_impact` is the standard decomposition
separating what market makers earn from what informed traders take.

### Two things worth knowing

**Roll's estimator needs random trade signs.** Identification relies on
independent bid-ask bounce. A deterministic alternation doubles the autocovariance
and hence the estimate. Where the sample autocovariance comes out positive the
model is rejected by the data, and `roll_spread` returns `0.0` rather than `nan`,
following the empirical convention.

**Spreads need float64.** `P_t - M_t` is a difference of nearly equal large
numbers. At a $100 price level float32 retains about three significant digits of
the difference. Call `finax.infrastructure.enable_x64()` before creating arrays.

## Vectorising across a cross-section

Everything batches:

```python
import jax
illiq = jax.vmap(amihud_illiquidity)(returns_2d, dollar_volume_2d)   # (n_firms,)
```
