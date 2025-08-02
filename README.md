# Finax

Finax is a Python library built on JAX and Diffrax for financial data processing and modeling with neural ordinary and stochastic differential equations. It targets researchers studying information asymmetry and provides adapters for popular deep learning frameworks alongside GPU/TPU execution, making it suitable for use in Google Colab or other accelerated environments.

## Features

### Data Handling
- Load CSV, Parquet, JSON, Excel, HDF5, and SQLite datasets
- Clean and engineer features; fetch market data through connectors such as Refinitiv Eikon

### Modeling
- Build neural ODE, SDE, CDE, and jump-diffusion models on top of Diffrax
- Predefined constructs for geometric Brownian motion, Vasicek interest rates, and logistic growth
- Integrate networks authored in TensorFlow, PyTorch, Flax, or Haiku

### Research Utilities
- Compute publication-grade metrics including probability of informed trading (PIN) and volume‑synchronized PIN (VPIN)

### Infrastructure
- Device helpers automatically select CPU, GPU, or TPU and move arrays to accelerators
- Configuration loading for reproducible experiments

### Visualization
- Plot time series, training curves, and model solutions via Matplotlib and Seaborn

## Installation

Finax depends on JAX, Diffrax, NumPy and Pandas. Optional extras enable framework or data connectors:

```bash
pip install finax[tensorflow,torch,eikon,flax,haiku,visualization]
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
- `docs/research.md` – information asymmetry metrics
- `docs/infrastructure.md` – device management and configuration utilities
- `docs/visualization.md` – plotting helpers for time series and model outputs

The project will expand with additional connectors, models, and training routines as development progresses.
