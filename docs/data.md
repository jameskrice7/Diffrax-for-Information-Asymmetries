# Data

Pandas-facing utilities, plus the bridge into JAX arrays.

## Loading

```python
from finax.data import load_csv, load_parquet, load_sqlite

df = load_csv("prices.csv", parse_dates=["date"], index_col="date")
```

Also `load_parquet`, `load_json`, `load_excel`, `load_hdf5`, `load_sqlite`,
`load_remote_csv`, `fetch_url_csv`, `load_hf_dataset`, `stream_quotes`.

All of these need the `data` extra. A missing dependency raises
`MissingDependencyError` naming the exact install command.

## Crossing into JAX

This is the boundary where most of the friction lives: irregular timestamps,
missingness that should be preserved rather than imputed, and ragged panels that
have to become rectangular arrays.

### One series

```python
from finax.data import to_arrays

ts, ys = to_arrays(df, time_column="date", time_unit="D")
```

Two details that matter:

**Time is measured from the first observation**, not from the epoch. An
epoch-based float64 timestamp in seconds is around 1.7e9; subtracting two of those
in float32 loses all intraday resolution.

**NaNs are preserved.** Missingness is information, and `prepare_channels` encodes
it as mask channels rather than imputing it away.

### A ragged panel

```python
from finax.data import panel_to_batch

ts, ys, tickers = panel_to_batch(df, entity_column="ticker", time_column="date")
# ts:  (n_entities, max_len)
# ys:  (n_entities, max_len, n_channels)
```

Shorter series are padded. Note that **time padding repeats the final timestamp**
rather than inserting NaN: Diffrax requires non-decreasing `ts`, and repeated
timestamps produce zero-length intervals that contribute nothing to a CDE
integral — exactly the semantics you want for "this series already ended".

### Building a control path

```python
from finax.core import build_control_path
import jax

paths = jax.vmap(build_control_path)(ts, ys)
```

`build_control_path` applies the standard neural-CDE recipe (Kidger et al. 2020;
Kidger 2021 ch. 3):

1. **Time as a channel.** Without it a CDE cannot distinguish a one-second gap
   from a one-year gap, and a path constant over an interval contributes nothing
   at all to the integral.
2. **Cumulative observation masks.** The *increment* of a running count is the 0/1
   observed flag, so a CDE gets the indicator for free while also encoding
   time-since-last-observation.
3. **Fill forward**, then back-fill any leading NaNs. Channels never observed
   become zero.
4. **Interpolate.**

| `method` | Smooth? | Looks ahead? | Use when |
| --- | --- | --- | --- |
| `"hermite"` (default) | yes | one observation | General offline use; adaptive solvers behave |
| `"linear"` | no | one observation | Cheap, with fixed-step solvers |
| `"rectilinear"` | no | **never** | Genuine online/streaming prediction |

Read `path.n_channels` for the model's `input_size` rather than counting by hand —
the augmentation adds channels.

## Cleaning

```python
from finax.data import fill_missing, detect_outliers, winsorize

df = fill_missing(df, method="ffill", limit=3)
mask = detect_outliers(df, threshold=3.0)      # robust by default
df = winsorize(df, lower=0.01, upper=0.99)
```

**Set `limit`.** Filling forward without one propagates a stale price across an
entire gap, which later shows up as spurious zero-volatility periods.

**`detect_outliers` is robust by default**, using the median and MAD rather than
the mean and standard deviation. The classical z-score is computed *from* the data
it is testing, so a single extreme outlier inflates the standard deviation enough
to mask itself:

```python
df = pd.DataFrame({"x": [1.0, 1.1, 0.9, 1.05, 0.95, 100.0]})
detect_outliers(df)["x"].iloc[-1]                 # True
detect_outliers(df, robust=False)["x"].iloc[-1]   # False - masked
```

For returns, prefer `winsorize` to dropping: extreme returns are real events, and
winsorizing limits their leverage without discarding the observation.

## Features

```python
from finax.data import rsi, macd, bollinger_bands, realized_volatility

df["rsi"] = rsi(df["close"])
df["rv"] = realized_volatility(df["close"], 21, annualize=252)
```

`rsi` uses **Wilder's smoothing** (`alpha = 1/window`), not a simple rolling mean.
The two give visibly different values and charting packages universally implement
Wilder's.

`realized_volatility` does not subtract a mean, matching the quadratic-variation
definition used in high-frequency econometrics — over short horizons the drift is
negligible and estimating it only adds noise. `rolling_volatility` is the centred
alternative.

`event_flags` annotates a date-indexed frame with a binary column per event type,
which is the usual setup for studying behaviour around scheduled disclosures.

## OHLCV

```python
from finax.data import resample_ohlcv, compute_bid_ask_spread

bars = resample_ohlcv(ticks, "5min", price_column="price")   # OHLC from trades
daily = resample_ohlcv(bars, "D")                            # aggregate OHLC
spread = compute_bid_ask_spread(quotes, relative=True)
```

Use `relative=True` for cross-sectional comparisons: one cent means very different
things at $5 and at $500.
