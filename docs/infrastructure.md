# Infrastructure

## Precision

JAX defaults to float32. That is fine for most deep learning and not fine for
likelihood optimisation or for differences of large, nearly equal prices.

```python
from finax.infrastructure import enable_x64
enable_x64()      # BEFORE creating any arrays
```

JAX caches its backend, so this has no effect once arrays exist.

Enable it for: `finax.microstructure.pin`, `finax.inference.calibrate`, and
trade-level spread work (`effective_spread` at a $100 price level retains about
three significant digits in float32).

## Devices

```python
from finax.infrastructure import device_summary, best_platform, to_device

device_summary()
# {'platform': 'cpu', 'device_count': 1, 'devices': [...],
#  'default_backend': 'cpu', 'x64_enabled': False}
```

`best_platform()` returns TPU, else GPU, else CPU. `to_device` places a whole
PyTree.

Worth calling `device_summary()` at the top of a notebook to confirm you are on
the accelerator you think you are, at the precision you think you are.

## Reproducibility

```python
from finax.infrastructure import seed_everything, reproducibility_report

key = seed_everything(42)
report = reproducibility_report()
```

JAX randomness is explicit and does not depend on global state, so the returned
key is the part that matters. Python's and NumPy's global seeds are set too, since
data shuffling and NumPy preprocessing do use them.

`reproducibility_report()` captures library versions, platform, precision and
`JAX_*` / `XLA_*` environment variables — the methods-section material for a paper.

## Configuration

```python
from finax.infrastructure import load_config, save_config

config = load_config("experiment.toml")   # .json, .toml, .yaml all supported
save_config(config, "run/config.json")
```

## Logging

```python
from finax.utils import get_logger, set_level

set_level("INFO")
log = get_logger("models")   # -> "finax.models"
```

Importing finax attaches only a `NullHandler`, so it never configures logging for
the application that imports it. Call `set_level` to opt in.
