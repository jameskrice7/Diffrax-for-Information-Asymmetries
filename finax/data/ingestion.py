"""Loading tabular data from files, databases and remote sources.

Thin wrappers over pandas that add consistent error messages and lazy optional
imports. Nothing here is JAX-specific; use :func:`~finax.data.frames.to_arrays`
to cross into JAX.
"""

from __future__ import annotations

import io
import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from ..errors import DataValidationError, require

__all__ = [
    "load_csv",
    "load_parquet",
    "load_json",
    "load_excel",
    "load_hdf5",
    "load_sqlite",
    "load_remote_csv",
    "load_hf_dataset",
    "fetch_url_csv",
    "stream_quotes",
]


def _pandas():
    return require("pandas", purpose="tabular data loading")


def load_csv(
    path: str | Path,
    *,
    parse_dates: list[str] | None = None,
    index_col: str | None = None,
    **kwargs: Any,
):
    """Load a CSV file into a DataFrame.

    Parameters
    ----------
    path:
        Local file path.
    parse_dates:
        Columns to parse as datetimes.
    index_col:
        Column to use as the index.
    **kwargs:
        Forwarded to ``pandas.read_csv``.
    """
    pd = _pandas()
    return pd.read_csv(path, parse_dates=parse_dates, index_col=index_col, **kwargs)


def load_parquet(path: str | Path, **kwargs: Any):
    """Load a Parquet file into a DataFrame."""
    pd = _pandas()
    require("pyarrow", purpose="reading Parquet")
    return pd.read_parquet(path, **kwargs)


def load_json(path: str | Path, **kwargs: Any):
    """Load a JSON file into a DataFrame."""
    pd = _pandas()
    return pd.read_json(path, **kwargs)


def load_excel(path: str | Path, *, sheet_name: str | int | None = 0, **kwargs: Any):
    """Load a sheet of an Excel workbook into a DataFrame."""
    pd = _pandas()
    require("openpyxl", purpose="reading Excel workbooks")
    return pd.read_excel(path, sheet_name=sheet_name, **kwargs)


def load_hdf5(path: str | Path, key: str = "data", **kwargs: Any):
    """Load a dataset from an HDF5 store."""
    pd = _pandas()
    require("tables", purpose="reading HDF5")
    return pd.read_hdf(path, key=key, **kwargs)


def load_sqlite(path: str | Path, query: str, **kwargs: Any):
    """Run a SQL query against a SQLite file and return the result.

    Examples
    --------
    >>> import sqlite3, tempfile, os
    >>> tmp = os.path.join(tempfile.mkdtemp(), "t.db")
    >>> con = sqlite3.connect(tmp)
    >>> _ = con.execute("CREATE TABLE p (d TEXT, close REAL)")
    >>> _ = con.execute("INSERT INTO p VALUES ('2024-01-01', 10.5)")
    >>> con.commit(); con.close()
    >>> df = load_sqlite(tmp, "SELECT * FROM p")
    >>> df["close"].tolist()
    [10.5]
    """
    import sqlite3

    pd = _pandas()
    # `with sqlite3.connect(...)` manages the *transaction*, not the
    # connection -- it does not close it. Use closing() so the handle is
    # actually released.
    from contextlib import closing

    with closing(sqlite3.connect(str(path))) as conn:
        return pd.read_sql_query(query, conn, **kwargs)


def load_remote_csv(url: str, *, parse_dates: list[str] | None = None, **kwargs: Any):
    """Load a CSV directly from a URL."""
    pd = _pandas()
    return pd.read_csv(url, parse_dates=parse_dates, **kwargs)


def fetch_url_csv(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 30.0,
):
    """Fetch a CSV over HTTP with explicit parameters, headers and timeout.

    Use this rather than :func:`load_remote_csv` when the endpoint needs query
    parameters or authentication headers.
    """
    requests = require("requests", purpose="HTTP requests")
    pd = _pandas()
    response = requests.get(url, params=params, headers=headers, timeout=timeout)
    response.raise_for_status()
    return pd.read_csv(io.StringIO(response.text))


def load_hf_dataset(name: str, *, split: str = "train", **kwargs: Any):
    """Load a Hugging Face Hub dataset into a DataFrame."""
    datasets = require("datasets", purpose="loading Hugging Face datasets")
    return datasets.load_dataset(name, split=split, **kwargs).to_pandas()


def stream_quotes(
    *,
    ws_url: str | None = None,
    kafka_servers: list[str] | None = None,
    kafka_topic: str | None = None,
    parser: Callable[[str], Any] = json.loads,
) -> Iterator[Any]:
    """Yield parsed messages from a WebSocket or Kafka quote stream.

    Parameters
    ----------
    ws_url:
        WebSocket endpoint.
    kafka_servers, kafka_topic:
        Kafka bootstrap servers and topic.
    parser:
        Applied to each raw message. Defaults to ``json.loads``.

    Yields
    ------
    Parsed messages, indefinitely. Wrap in ``itertools.islice`` to bound it.
    """
    if ws_url:
        websocket = require("websocket", purpose="WebSocket streaming")
        connection = websocket.create_connection(ws_url)
        try:
            while True:
                yield parser(connection.recv())
        finally:
            connection.close()
    elif kafka_servers and kafka_topic:
        kafka = require("kafka", purpose="Kafka streaming")
        consumer = kafka.KafkaConsumer(kafka_topic, bootstrap_servers=kafka_servers)
        try:
            for message in consumer:
                value = message.value
                if isinstance(value, bytes):
                    value = value.decode("utf-8")
                yield parser(value)
        finally:
            consumer.close()
    else:
        raise DataValidationError(
            "stream_quotes needs either ws_url, or both kafka_servers and kafka_topic."
        )
