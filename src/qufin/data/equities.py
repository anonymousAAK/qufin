"""Equity data provider using yfinance."""

from __future__ import annotations

from typing import Literal

import pandas as pd


class YahooEquityProvider:
    """Fetch adjusted equity returns from Yahoo Finance via yfinance.

    Parameters
    ----------
    cache : bool
        Whether to cache downloaded data to parquet.
    """

    def __init__(self, cache: bool = True) -> None:
        self._cache = cache

    def get_prices(
        self,
        tickers: list[str],
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """Fetch adjusted close prices.

        Returns
        -------
        pd.DataFrame
            Columns are tickers, index is datetime.
        """
        import yfinance as yf

        data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            prices = data["Close"]
        else:
            prices = data[["Close"]]
            prices.columns = tickers
        return prices.dropna()

    def get_returns(
        self,
        tickers: list[str],
        start: str,
        end: str,
        frequency: Literal["D", "W", "M"] = "D",
    ) -> pd.DataFrame:
        """Fetch log returns at the given frequency.

        Returns
        -------
        pd.DataFrame
            Log returns, columns are tickers.
        """
        import numpy as np

        prices = self.get_prices(tickers, start, end)

        freq_map = {"D": "D", "W": "W-FRI", "M": "ME"}
        if frequency != "D":
            prices = prices.resample(freq_map[frequency]).last().dropna()

        returns = np.log(prices / prices.shift(1)).dropna()
        return returns
