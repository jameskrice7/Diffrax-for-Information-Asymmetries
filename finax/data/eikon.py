"""Refinitiv Eikon data connector."""

from __future__ import annotations

from typing import Optional

import pandas as pd

try:  # pragma: no cover - optional dependency
    import eikon  # type: ignore
except Exception:  # pragma: no cover
    eikon = None  # type: ignore


def fetch_eikon(
    symbol: str,
    *,
    fields: Optional[list[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch time series data from Refinitiv Eikon.

    Parameters
    ----------
    symbol:
        Instrument identifier (RIC).
    fields:
        Optional list of fields to request.
    start_date, end_date:
        Date range for the request.
    api_key:
        Application key for authenticating with the Eikon API. If omitted, the
        function relies on previously configured environment variables.
    """

    if eikon is None:  # pragma: no cover - runtime check
        raise ImportError("The 'eikon' package is required for Refinitiv access.")

    if api_key is not None:
        eikon.set_app_key(api_key)

    data = eikon.get_timeseries(
        symbols=symbol,
        fields=fields,
        start_date=start_date,
        end_date=end_date,
    )
    return data
