# Finax

Finax is a Python library built on JAX and Diffrax for financial data processing and modeling with neural ordinary and stochastic differential equations. It targets researchers studying information asymmetry and provides adapters for popular deep learning frameworks alongside GPU/TPU execution, making it suitable for use in Google Colab or other accelerated environments.

## Features

### Data Handling
- Load CSV, Parquet, JSON, Excel, HDF5, and SQLite datasets using pandas
- Import remote data via URLs or Hugging Face datasets; fetch market data through connectors such as Refinitiv Eikon
- Aggregate intraday quotes into daily or monthly OHLCV bars and compute bid-ask spreads


### Modeling
- Build neural ODE, SDE, CDE, and jump-diffusion models on top of Diffrax
- Predefined constructs for geometric Brownian motion, Vasicek interest rates, and logistic growth
- Simulate standalone Brownian motion and Poisson processes
- Integrate networks authored in TensorFlow, PyTorch, Flax, Haiku, or Hugging Face Transformers
- Flax modules tailored to financial time-series modeling

### Research Utilities
- Compute publication-grade metrics including probability of informed trading (PIN), volume‑synchronized PIN (VPIN), and PIN derived from daily prices

### Evaluation
- Fit AR, MA, ARMA, ARIMA, or GARCH models to simulated time series for post-hoc analysis
- 
### Infrastructure
- Device helpers automatically select CPU, GPU, or TPU and move arrays to accelerators
- Configuration loading for reproducible experiments

### Visualization
- Plot time series, training curves, and model solutions via Matplotlib and Seaborn

## Installation

Finax depends on JAX, Diffrax, NumPy and Pandas. Optional extras enable framework or data connectors:

```bash

pip install finax[tensorflow,torch,eikon,flax,haiku,visualization,huggingface]

```

Each extra can also be installed individually (e.g., `pip install finax[eikon]`).

## Quick Start

```python
from finax.data.eikon import fetch_eikon
from finax.infrastructure.devices import to_device
import jax.numpy as jnp

df = fetch_eikon("AAPL.O", fields=["CLOSE"], start_date="2020-01-01", end_date="2020-06-01")
x = to_device(jnp.asarray(df["CLOSE"].values))
```

## Documentation

Additional guides are available in the `docs/` directory:

- `docs/data.md` – ingestion, cleaning, and feature engineering
- `docs/modeling.md` – neural ODE/SDE wrappers and framework adapters
- `docs/evaluation.md` – metrics and classical time-series models
- `docs/research.md` – information asymmetry metrics
- `docs/infrastructure.md` – device management and configuration utilities
- `docs/visualization.md` – plotting helpers for time series and model outputs

The project will expand with additional connectors, models, and training routines as development progresses.

Finax is a Python library built on JAX and Diffrax for financial data
processing and modeling with neural ordinary and stochastic differential
 equations. It targets researchers studying information asymmetry and
provides adapters for popular deep learning frameworks alongside GPU/TPU
execution, making it suitable for use in Google Colab or other
accelerated environments.


## Package Structure

- `finax/data` – loading, cleaning, feature engineering, and API connectors
  such as Refinitiv Eikon.
- `finax/modeling` – neural ODE/SDE abstractions, training helpers, and
  adapters for TensorFlow and PyTorch models.
- `finax/evaluation` – performance metrics.
- `finax/infrastructure` – configuration helpers.
- `finax/utils` – shared utilities such as logging.

The project will expand with additional connectors, models, and training
routines as development progresses.


