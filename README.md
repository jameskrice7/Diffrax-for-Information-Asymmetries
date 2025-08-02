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
  adapters for TensorFlow and PyTorch models.
- `finax/evaluation` – performance metrics.
- `finax/infrastructure` – configuration helpers.
- `finax/utils` – shared utilities such as logging.

The project will expand with additional connectors, models, and training
routines as development progresses.
