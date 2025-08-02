# Data Handling

Finax provides utilities for ingesting, cleaning, and engineering features from financial datasets.

## Ingestion
- `finax.data.ingestion.load_csv`, `load_parquet`, and `load_json` load common file formats into `pandas.DataFrame` objects.
- `finax.data.ingestion.fetch_yahoo` is a placeholder for a Yahoo Finance connector.
- `finax.data.eikon.fetch_eikon` retrieves time-series data from Refinitiv Eikon when the `eikon` package is installed.

```python
from finax.data.eikon import fetch_eikon
quotes = fetch_eikon(
    "AAPL.O",
    fields=["CLOSE"],
    start_date="2020-01-01",
    end_date="2020-06-01",
    api_key="YOUR_APP_KEY",
)
```

## Cleaning
- `finax.data.cleaning.fill_missing` forward-fills missing values.
- `finax.data.cleaning.detect_outliers` flags outliers using a z-score threshold.

## Feature Engineering
- `finax.data.features.rolling_mean` computes rolling averages.
- Additional indicators such as RSI can be implemented via `finax.data.features.technical_indicator`.
