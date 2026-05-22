"""Cryptocurrency data connector (CoinGecko free API).

Fetches OHLCV and return data without requiring an API key.

References
----------
CoinGecko API v3 — https://www.coingecko.com/en/api/documentation
"""

from __future__ import annotations

import time

import pandas as pd

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None  # type: ignore[assignment]

_BASE_URL = "https://api.coingecko.com/api/v3"
_RATE_LIMIT_S = 1.2  # CoinGecko free tier: ~50 req/min


def fetch_crypto_prices(
    coins: list[str],
    vs_currency: str = "usd",
    days: int = 90,
) -> pd.DataFrame:
    """Fetch daily close prices from CoinGecko.

    Parameters
    ----------
    coins : list[str]
        CoinGecko coin IDs, e.g. ``["bitcoin", "ethereum"]``.
    vs_currency : str
        Quote currency (default ``"usd"``).
    days : int
        Look-back window in days.

    Returns
    -------
    pd.DataFrame
        Columns are coin IDs, index is datetime.
    """
    if requests is None:  # pragma: no cover
        raise ImportError(
            "requests is required for crypto data: pip install requests"
        )

    frames: dict[str, pd.Series] = {}
    for i, coin in enumerate(coins):
        if i > 0:
            time.sleep(_RATE_LIMIT_S)
        url = f"{_BASE_URL}/coins/{coin}/market_chart"
        params = {"vs_currency": vs_currency, "days": days, "interval": "daily"}
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        prices = data["prices"]  # list of [timestamp_ms, price]
        ts = pd.to_datetime([p[0] for p in prices], unit="ms", utc=True)
        frames[coin] = pd.Series([p[1] for p in prices], index=ts, name=coin)

    df = pd.DataFrame(frames)
    df.index.name = "date"
    return df.dropna()


def fetch_crypto_returns(
    coins: list[str],
    vs_currency: str = "usd",
    days: int = 90,
) -> pd.DataFrame:
    """Fetch daily simple returns from CoinGecko.

    Parameters
    ----------
    coins : list[str]
        CoinGecko coin IDs.
    vs_currency : str
        Quote currency.
    days : int
        Look-back window in days.

    Returns
    -------
    pd.DataFrame
        Daily simple returns (pct_change).
    """
    prices = fetch_crypto_prices(coins, vs_currency=vs_currency, days=days)
    return prices.pct_change().dropna()
