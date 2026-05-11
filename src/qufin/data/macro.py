"""Macro data provider using FRED API.

Fetches risk-free rates, yield curves, and macro indicators from the
Federal Reserve Economic Data (FRED) API via the fredapi package.

Requires: pip install fredapi
Requires FRED_API_KEY environment variable or passed directly.

References
----------
FRED: https://fred.stlouisfed.org/
"""

from __future__ import annotations

from typing import Literal

import pandas as pd


class FREDProvider:
    """Fetch macroeconomic data from FRED.

    Parameters
    ----------
    api_key : str | None
        FRED API key. If None, reads from FRED_API_KEY env variable.
    """

    # Common FRED series IDs
    SERIES = {
        "tbill_3m": "DTB3",          # 3-Month Treasury Bill
        "tbill_6m": "DTB6",          # 6-Month Treasury Bill
        "yield_1y": "DGS1",          # 1-Year Treasury
        "yield_2y": "DGS2",          # 2-Year Treasury
        "yield_5y": "DGS5",          # 5-Year Treasury
        "yield_10y": "DGS10",        # 10-Year Treasury
        "yield_30y": "DGS30",        # 30-Year Treasury
        "fed_funds": "DFF",          # Federal Funds Rate
        "cpi": "CPIAUCSL",           # CPI (monthly)
        "unemployment": "UNRATE",    # Unemployment Rate
        "gdp": "GDP",                # GDP (quarterly)
        "vix": "VIXCLS",             # VIX Close
        "sp500": "SP500",            # S&P 500 Index
    }

    def __init__(self, api_key: str | None = None) -> None:
        try:
            from fredapi import Fred
        except ImportError as e:
            raise ImportError(
                "fredapi is required for FRED data. "
                "Install with: pip install fredapi"
            ) from e

        self._fred = Fred(api_key=api_key)

    def get_series(
        self,
        series_id: str,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.Series:
        """Fetch a single FRED series.

        Parameters
        ----------
        series_id : str
            FRED series ID (e.g., "DGS10") or a shorthand from SERIES dict.
        start, end : str | None
            Date strings (e.g., "2020-01-01").

        Returns
        -------
        pd.Series with datetime index.
        """
        sid = self.SERIES.get(series_id, series_id)
        data = self._fred.get_series(sid, observation_start=start, observation_end=end)
        return data.dropna()

    def get_risk_free_rate(
        self,
        maturity: Literal["3m", "6m", "1y", "2y", "5y", "10y"] = "3m",
        start: str | None = None,
        end: str | None = None,
    ) -> pd.Series:
        """Fetch Treasury yield as a risk-free rate proxy.

        Returns annualized rate as a decimal (e.g., 0.05 for 5%).
        """
        maturity_map = {
            "3m": "DTB3",
            "6m": "DTB6",
            "1y": "DGS1",
            "2y": "DGS2",
            "5y": "DGS5",
            "10y": "DGS10",
        }
        sid = maturity_map.get(maturity, "DTB3")
        data = self._fred.get_series(sid, observation_start=start, observation_end=end)
        return data.dropna() / 100.0  # Convert from percent to decimal

    def get_yield_curve(self, date: str) -> pd.Series:
        """Fetch the Treasury yield curve for a specific date.

        Returns a Series indexed by maturity label with annualized yields.
        """
        maturities = ["3m", "6m", "1y", "2y", "5y", "10y", "30y"]
        series_ids = ["DTB3", "DTB6", "DGS1", "DGS2", "DGS5", "DGS10", "DGS30"]

        yields = {}
        for mat, sid in zip(maturities, series_ids, strict=False):
            data = self._fred.get_series(sid, observation_start=date, observation_end=date)
            data = data.dropna()
            if len(data) > 0:
                yields[mat] = float(data.iloc[-1]) / 100.0

        return pd.Series(yields, name=f"yield_curve_{date}")

    def get_vix(
        self,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.Series:
        """Fetch VIX (implied volatility index)."""
        return self.get_series("VIXCLS", start=start, end=end)
