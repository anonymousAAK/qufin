"""Refinitiv/LSEG data provider using the Eikon Data API.

Fetches equities, fixed income, and derivatives data from the
Refinitiv Eikon / LSEG Workspace platform.  An alternative to
Bloomberg for firms in the LSEG ecosystem.

Requires: pip install eikon   (or refinitiv-data)
Requires an active Eikon / Workspace session or an API proxy.

References
----------
Eikon Data API: https://developers.lseg.com/en/api-catalog/eikon/eikon-data-api
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Optional dependency guard
# ---------------------------------------------------------------------------
try:
    import eikon as ek  # type: ignore[import-untyped]

    _HAS_EIKON = True
except ImportError:
    ek = None  # type: ignore[assignment]
    _HAS_EIKON = False


def _require_eikon() -> None:
    """Raise a helpful error when the eikon package is missing."""
    if not _HAS_EIKON:
        raise ImportError(
            "The eikon package is required for Refinitiv data. "
            "Install with: pip install eikon"
        )


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class RefinitivConfig:
    """Configuration for the Refinitiv Eikon connection.

    Parameters
    ----------
    app_key : str
        Eikon Data API application key (aka app-key / api-key).
    timeout : int
        Request timeout in seconds.  Default 30.
    cache : bool
        Whether to cache downloaded data to parquet via qufin caching.
    """

    app_key: str = ""
    timeout: int = 30
    cache: bool = True


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TimeSeriesResult:
    """Container for time-series query results.

    Attributes
    ----------
    data : pd.DataFrame
        The returned price / field data.
    rics : list[str]
        RIC codes that were queried.
    fields : list[str]
        Eikon field names returned.
    metadata : dict[str, Any]
        Extra metadata from the response.
    """

    data: pd.DataFrame
    rics: list[str]
    fields: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SnapshotResult:
    """Container for a point-in-time data snapshot.

    Attributes
    ----------
    data : pd.DataFrame
        Snapshot data with RICs as rows.
    rics : list[str]
        RIC codes that were queried.
    fields : list[str]
        Eikon field names returned.
    """

    data: pd.DataFrame
    rics: list[str]
    fields: list[str]


# ---------------------------------------------------------------------------
# Main provider
# ---------------------------------------------------------------------------


class RefinitivDataSource:
    """Fetch market data from Refinitiv Eikon / LSEG Workspace.

    Parameters
    ----------
    config : RefinitivConfig | None
        Connection configuration.  When *None*, a default config is used
        and the ``app_key`` must be supplied separately via :meth:`set_app_key`.
    """

    # Common Eikon fields
    EQUITY_FIELDS = [
        "TR.PriceClose",
        "TR.Volume",
        "TR.PriceHigh",
        "TR.PriceLow",
        "TR.PriceOpen",
    ]
    FIXED_INCOME_FIELDS = [
        "TR.BIDPRICE",
        "TR.ASKPRICE",
        "TR.BIDYIELD",
        "TR.ASKYIELD",
        "TR.MIDYIELD",
    ]
    DERIVATIVES_FIELDS = [
        "TR.SETTLEMENTPRICE",
        "TR.OPENINTEREST",
        "TR.VOLUME",
        "TR.STRIKEPRICE",
        "TR.EXPIRATIONDATE",
    ]

    def __init__(self, config: RefinitivConfig | None = None) -> None:
        _require_eikon()
        self._config = config or RefinitivConfig()
        if self._config.app_key:
            ek.set_app_key(self._config.app_key)  # type: ignore[union-attr]
        self._timeout = self._config.timeout

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def set_app_key(self, app_key: str) -> None:
        """Set (or reset) the Eikon application key at runtime."""
        self._config.app_key = app_key
        ek.set_app_key(app_key)  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # Equities
    # ------------------------------------------------------------------

    def get_equity_prices(
        self,
        rics: list[str],
        start: str,
        end: str,
        interval: Literal["daily", "weekly", "monthly"] = "daily",
    ) -> TimeSeriesResult:
        """Fetch historical equity close prices.

        Parameters
        ----------
        rics : list[str]
            Reuters Instrument Codes, e.g. ``["AAPL.O", "MSFT.O"]``.
        start, end : str
            ISO date strings (``"2023-01-01"``).
        interval : {"daily", "weekly", "monthly"}
            Bar frequency.

        Returns
        -------
        TimeSeriesResult
        """
        data = ek.get_timeseries(  # type: ignore[union-attr]
            rics,
            fields=["CLOSE"],
            start_date=start,
            end_date=end,
            interval=interval,
            timeout=self._timeout,
        )
        if isinstance(data, pd.DataFrame):
            df = data
        else:
            df = pd.DataFrame(data)

        return TimeSeriesResult(
            data=df,
            rics=rics,
            fields=["CLOSE"],
            metadata={"interval": interval},
        )

    def get_equity_ohlcv(
        self,
        rics: list[str],
        start: str,
        end: str,
        interval: Literal["daily", "weekly", "monthly"] = "daily",
    ) -> TimeSeriesResult:
        """Fetch historical OHLCV bars for equities.

        Parameters
        ----------
        rics : list[str]
            Reuters Instrument Codes.
        start, end : str
            ISO date strings.
        interval : {"daily", "weekly", "monthly"}
            Bar frequency.

        Returns
        -------
        TimeSeriesResult
        """
        ts_fields = ["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]
        data = ek.get_timeseries(  # type: ignore[union-attr]
            rics,
            fields=ts_fields,
            start_date=start,
            end_date=end,
            interval=interval,
            timeout=self._timeout,
        )
        if isinstance(data, pd.DataFrame):
            df = data
        else:
            df = pd.DataFrame(data)

        return TimeSeriesResult(
            data=df,
            rics=rics,
            fields=ts_fields,
            metadata={"interval": interval},
        )

    def get_equity_returns(
        self,
        rics: list[str],
        start: str,
        end: str,
        frequency: Literal["D", "W", "M"] = "D",
    ) -> pd.DataFrame:
        """Compute log returns from equity close prices.

        Parameters
        ----------
        rics : list[str]
            Reuters Instrument Codes.
        start, end : str
            ISO date strings.
        frequency : {"D", "W", "M"}
            Return frequency.

        Returns
        -------
        pd.DataFrame
            Log returns with datetime index, one column per RIC.
        """
        interval_map: dict[str, Literal["daily", "weekly", "monthly"]] = {
            "D": "daily",
            "W": "weekly",
            "M": "monthly",
        }
        result = self.get_equity_prices(
            rics, start, end, interval=interval_map[frequency]
        )
        prices = result.data
        if prices.empty:
            return prices
        returns: pd.DataFrame = np.log(prices / prices.shift(1)).dropna()
        return returns

    # ------------------------------------------------------------------
    # Fixed income
    # ------------------------------------------------------------------

    def get_bond_data(
        self,
        rics: list[str],
        fields: list[str] | None = None,
    ) -> SnapshotResult:
        """Fetch current bond / fixed-income snapshot data.

        Parameters
        ----------
        rics : list[str]
            Bond RICs, e.g. ``["US10YT=RR", "DE10YT=RR"]``.
        fields : list[str] | None
            Eikon field names.  Defaults to :attr:`FIXED_INCOME_FIELDS`.

        Returns
        -------
        SnapshotResult
        """
        fields = fields or list(self.FIXED_INCOME_FIELDS)
        df, err = ek.get_data(  # type: ignore[union-attr]
            rics,
            fields,
            timeout=self._timeout,
        )
        if err is not None:
            raise RuntimeError(f"Eikon get_data error: {err}")
        return SnapshotResult(data=df, rics=rics, fields=fields)

    def get_yield_curve(
        self,
        currency: str = "USD",
        rics: list[str] | None = None,
    ) -> pd.DataFrame:
        """Fetch government bond yields across maturities.

        Parameters
        ----------
        currency : str
            ``"USD"``, ``"EUR"``, ``"GBP"``, etc.  Used to pick default RICs.
        rics : list[str] | None
            Explicit RIC list.  Overrides *currency* defaults.

        Returns
        -------
        pd.DataFrame
            Columns ``maturity``, ``bid_yield``, ``ask_yield``, ``mid_yield``.
        """
        default_rics: dict[str, list[str]] = {
            "USD": [
                "US3MT=RR",
                "US6MT=RR",
                "US1YT=RR",
                "US2YT=RR",
                "US5YT=RR",
                "US10YT=RR",
                "US30YT=RR",
            ],
            "EUR": [
                "EU3MT=RR",
                "EU6MT=RR",
                "EU1YT=RR",
                "EU2YT=RR",
                "EU5YT=RR",
                "EU10YT=RR",
                "EU30YT=RR",
            ],
            "GBP": [
                "GB3MT=RR",
                "GB6MT=RR",
                "GB1YT=RR",
                "GB2YT=RR",
                "GB5YT=RR",
                "GB10YT=RR",
                "GB30YT=RR",
            ],
        }
        if rics is None:
            rics = default_rics.get(currency.upper())
            if rics is None:
                raise ValueError(
                    f"No default RICs for currency {currency!r}. "
                    f"Supported: {list(default_rics.keys())}. "
                    "Supply rics explicitly."
                )

        yield_fields = ["TR.BIDYIELD", "TR.ASKYIELD", "TR.MIDYIELD"]
        df, err = ek.get_data(rics, yield_fields, timeout=self._timeout)  # type: ignore[union-attr]
        if err is not None:
            raise RuntimeError(f"Eikon get_data error: {err}")

        maturities = ["3M", "6M", "1Y", "2Y", "5Y", "10Y", "30Y"]
        if len(df) == len(maturities):
            df.insert(0, "maturity", maturities)
        return df

    # ------------------------------------------------------------------
    # Derivatives
    # ------------------------------------------------------------------

    def get_option_chain(
        self,
        underlying_ric: str,
        fields: list[str] | None = None,
    ) -> SnapshotResult:
        """Fetch option chain data for a given underlying.

        Parameters
        ----------
        underlying_ric : str
            Underlying RIC, e.g. ``"AAPL.O"``.
        fields : list[str] | None
            Eikon field names.  Defaults to :attr:`DERIVATIVES_FIELDS`.

        Returns
        -------
        SnapshotResult
        """
        fields = fields or list(self.DERIVATIVES_FIELDS)
        # Use ek.get_data with a chain RIC to pull the option chain.
        chain_ric = f"0#{underlying_ric}*.O"  # Eikon chain syntax
        df, err = ek.get_data(  # type: ignore[union-attr]
            chain_ric,
            fields,
            timeout=self._timeout,
        )
        if err is not None:
            raise RuntimeError(f"Eikon get_data error: {err}")
        return SnapshotResult(data=df, rics=[chain_ric], fields=fields)

    def get_futures_data(
        self,
        rics: list[str],
        fields: list[str] | None = None,
    ) -> SnapshotResult:
        """Fetch futures snapshot data.

        Parameters
        ----------
        rics : list[str]
            Futures RICs, e.g. ``["CLc1", "ESc1"]``.
        fields : list[str] | None
            Eikon field names.  Defaults to settlement, OI, volume.

        Returns
        -------
        SnapshotResult
        """
        fields = fields or ["TR.SETTLEMENTPRICE", "TR.OPENINTEREST", "TR.VOLUME"]
        df, err = ek.get_data(  # type: ignore[union-attr]
            rics,
            fields,
            timeout=self._timeout,
        )
        if err is not None:
            raise RuntimeError(f"Eikon get_data error: {err}")
        return SnapshotResult(data=df, rics=rics, fields=fields)

    # ------------------------------------------------------------------
    # Generic helpers
    # ------------------------------------------------------------------

    def get_data(
        self,
        rics: list[str],
        fields: list[str],
    ) -> SnapshotResult:
        """Generic wrapper around ``ek.get_data``.

        Parameters
        ----------
        rics : list[str]
            Instrument RICs.
        fields : list[str]
            Eikon TR field names.

        Returns
        -------
        SnapshotResult
        """
        df, err = ek.get_data(  # type: ignore[union-attr]
            rics,
            fields,
            timeout=self._timeout,
        )
        if err is not None:
            raise RuntimeError(f"Eikon get_data error: {err}")
        return SnapshotResult(data=df, rics=rics, fields=fields)

    def get_timeseries(
        self,
        rics: list[str],
        fields: list[str],
        start: str,
        end: str,
        interval: Literal["daily", "weekly", "monthly"] = "daily",
    ) -> TimeSeriesResult:
        """Generic wrapper around ``ek.get_timeseries``.

        Parameters
        ----------
        rics : list[str]
            Instrument RICs.
        fields : list[str]
            Timeseries field names (e.g. ``["CLOSE", "VOLUME"]``).
        start, end : str
            ISO date strings.
        interval : {"daily", "weekly", "monthly"}
            Bar frequency.

        Returns
        -------
        TimeSeriesResult
        """
        data = ek.get_timeseries(  # type: ignore[union-attr]
            rics,
            fields=fields,
            start_date=start,
            end_date=end,
            interval=interval,
            timeout=self._timeout,
        )
        if isinstance(data, pd.DataFrame):
            df = data
        else:
            df = pd.DataFrame(data)

        return TimeSeriesResult(
            data=df,
            rics=rics,
            fields=fields,
            metadata={"interval": interval},
        )

    def search(
        self,
        query: str,
        select: str | None = None,
        top: int = 10,
    ) -> pd.DataFrame:
        """Search for instruments using Eikon search.

        Parameters
        ----------
        query : str
            Free-text search query.
        select : str | None
            Comma-separated Eikon properties to return.
        top : int
            Maximum number of results.

        Returns
        -------
        pd.DataFrame
        """
        kwargs: dict[str, Any] = {"top": top}
        if select is not None:
            kwargs["select"] = select
        results = ek.get_symbology(  # type: ignore[union-attr]
            query,
            timeout=self._timeout,
            **kwargs,
        )
        if isinstance(results, pd.DataFrame):
            return results
        return pd.DataFrame(results)
