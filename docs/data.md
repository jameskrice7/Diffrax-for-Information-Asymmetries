# Data Handling

Finax provides utilities for ingesting, cleaning, and engineering features from financial datasets.

## Ingestion
- `finax.data.ingestion.load_csv`, `load_parquet`, and `load_json` load common file formats into `pandas.DataFrame` objects.
- `finax.data.ingestion.load_excel` reads workbooks in XLSX/XLS format.
- `finax.data.ingestion.load_hdf5` loads HDF5 stores, while `load_sqlite` issues SQL queries against SQLite databases.
- `finax.data.ingestion.load_remote_csv` reads CSV files directly from URLs for quick experimentation.
- `finax.data.ingestion.load_hf_dataset` fetches datasets from the Hugging Face Hub and converts them to DataFrames.
- `finax.data.ingestion.fetch_yahoo` downloads historical quotes from Yahoo Finance via its REST API.
- `finax.data.ingestion.fetch_quandl` retrieves datasets from Quandl using an API key.
- `finax.data.ingestion.stream_quotes` streams real-time quotes from WebSocket or Kafka feeds.
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

### API keys and dependencies

- Yahoo Finance endpoints used here do not require an API key.
- Quandl requests need a `QUANDL_API_KEY` provided via argument or environment variable.
- HTTP helpers rely on the `requests` package.
- Streaming utilities depend on `websocket-client` for WebSocket connections and `kafka-python` for Kafka consumers. Install them via optional extras: `finax[yahoo]`, `finax[quandl]`, or `finax[streaming]`.

## Cleaning
- `finax.data.cleaning.fill_missing` forward-fills missing values.
- `finax.data.cleaning.detect_outliers` flags outliers using a z-score threshold.

## Feature Engineering
- `finax.data.features.rolling_mean` computes rolling averages.
- Additional indicators such as RSI can be implemented via `finax.data.features.technical_indicator`.
- `finax.data.ohlc.daily_ohlcv` and `monthly_ohlcv` aggregate intraday trades into OHLCV bars and compute bid-ask spreads when available.
