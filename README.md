# Finax

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
  adapters for TensorFlow, PyTorch, Flax, and Haiku models.
- `finax/research` – information asymmetry metrics like PIN and VPIN.
- `finax/evaluation` – performance metrics.
- `finax/infrastructure` – configuration helpers and device utilities to
  select CPU, GPU, or TPU.
- `finax/utils` – shared utilities such as logging.

## Hardware acceleration

Finax automatically detects available JAX devices. When running in Google
Colab, choose a GPU or TPU runtime and use
`finax.infrastructure.to_device` to place arrays on the selected
accelerator.

The project will expand with additional connectors, models, and training
routines as development progresses.
