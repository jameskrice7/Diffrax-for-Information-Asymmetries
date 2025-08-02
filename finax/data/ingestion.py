"""Data ingestion utilities for Finax."""

from __future__ import annotations

from typing import Any, Callable, Iterator, Optional

import io
import json
import pandas as pd
import requests


def load_csv(path: str, *, parse_dates: Optional[list[str]] = None) -> pd.DataFrame:
    """Load CSV financial data into a DataFrame.

    Parameters
    ----------
    path:
        Local file path to a CSV file.
    parse_dates:
        Optional list of column names to parse as dates.
    """

    return pd.read_csv(path, parse_dates=parse_dates)


def load_parquet(path: str) -> pd.DataFrame:
    """Load Parquet financial data into a DataFrame."""
    return pd.read_parquet(path)


def load_json(path: str) -> pd.DataFrame:
    """Load JSON financial data into a DataFrame."""
    return pd.read_json(path)


def load_excel(path: str, *, sheet_name: str | int | None = 0) -> pd.DataFrame:
    """Load Excel financial data into a DataFrame.

    Parameters
    ----------
    path:
        Location of the Excel file.
    sheet_name:
        Sheet within the workbook to read. Defaults to the first sheet.
    """

    return pd.read_excel(path, sheet_name=sheet_name)


def load_hdf5(path: str, key: str = "data") -> pd.DataFrame:
    """Load HDF5 financial data into a DataFrame."""

    return pd.read_hdf(path, key=key)


def load_sqlite(path: str, query: str) -> pd.DataFrame:
    """Load data from a SQLite database using a SQL query."""
    import sqlite3

    with sqlite3.connect(path) as conn:
        return pd.read_sql_query(query, conn)


def load_remote_csv(url: str, *, parse_dates: Optional[list[str]] = None) -> pd.DataFrame:
    """Load a remote CSV file directly into a DataFrame using pandas."""

    return pd.read_csv(url, parse_dates=parse_dates)


def load_hf_dataset(name: str, *, split: str = "train", **kwargs) -> pd.DataFrame:
    """Load a dataset hosted on the Hugging Face Hub into a DataFrame.

    Parameters
    ----------
    name:
        Dataset identifier on the Hub.
    split:
        Which split to load (e.g. ``"train"`` or ``"test"``).
    **kwargs:
        Additional keyword arguments forwarded to ``datasets.load_dataset``.
    """

    from datasets import load_dataset  # type: ignore

    ds = load_dataset(name, split=split, **kwargs)
    return ds.to_pandas()


def fetch_yahoo(
    symbol: str,
    *,
    start: str | None = None,
    end: str | None = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """Fetch historical data from Yahoo Finance.

    Parameters
    ----------
    symbol:
        Ticker symbol to download, e.g. ``"AAPL"``.
    start, end:
        Optional date range in ``YYYY-MM-DD`` format.
    interval:
        Data frequency such as ``"1d"`` or ``"1h"``.
    """

    params: dict[str, Any] = {"interval": interval, "events": "history"}
    if start:
        params["period1"] = int(pd.Timestamp(start, tz="UTC").timestamp())
    else:
        params["period1"] = 0
    if end:
        params["period2"] = int(pd.Timestamp(end, tz="UTC").timestamp())
    else:
        params["period2"] = int(pd.Timestamp.utcnow().timestamp())

    url = f"https://query1.finance.yahoo.com/v7/finance/download/{symbol}"
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    return pd.read_csv(io.StringIO(resp.text))


def fetch_quandl(
    dataset: str,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    api_key: str | None = None,
) -> pd.DataFrame:
    """Retrieve a dataset from Quandl.

    Parameters
    ----------
    dataset:
        Quandl dataset identifier such as ``"WIKI/AAPL"``.
    start_date, end_date:
        Optional date filters in ``YYYY-MM-DD`` format.
    api_key:
        Quandl API key. If ``None``, the ``QUANDL_API_KEY`` environment
        variable is used when available.
    """

    params = {}
    if start_date:
        params["start_date"] = start_date
    if end_date:
        params["end_date"] = end_date
    if api_key is None:
        from os import getenv

        api_key = getenv("QUANDL_API_KEY")
    if api_key:
        params["api_key"] = api_key

    url = f"https://www.quandl.com/api/v3/datasets/{dataset}.json"
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    payload = resp.json()["dataset"]
    return pd.DataFrame(payload["data"], columns=payload["column_names"])


def stream_quotes(
    *,
    ws_url: str | None = None,
    kafka_servers: list[str] | None = None,
    kafka_topic: str | None = None,
    parser: Callable[[str], Any] = json.loads,
) -> Iterator[Any]:
    """Stream quote messages from WebSocket or Kafka sources.

    Parameters
    ----------
    ws_url:
        WebSocket endpoint providing quote data.
    kafka_servers:
        List of Kafka bootstrap servers.
    kafka_topic:
        Topic to subscribe to when using Kafka.
    parser:
        Function used to parse each incoming message. Defaults to ``json.loads``.

    Yields
    ------
    Parsed messages from the stream.
    """

    if ws_url:
        import websocket  # type: ignore

        ws = websocket.create_connection(ws_url)
        try:
            while True:
                msg = ws.recv()
                yield parser(msg)
        finally:
            ws.close()
    elif kafka_servers and kafka_topic:
        from kafka import KafkaConsumer  # type: ignore

        consumer = KafkaConsumer(kafka_topic, bootstrap_servers=kafka_servers)
        for msg in consumer:
            if isinstance(msg.value, bytes):
                yield parser(msg.value.decode("utf-8"))
            else:
                yield parser(msg.value)
    else:
        raise ValueError(
            "Provide either a WebSocket URL or Kafka connection parameters."
        )

