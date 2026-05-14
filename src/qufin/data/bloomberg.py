"""Bloomberg data provider using the blpapi SDK.

Provides access to Bloomberg Terminal data including:
- Historical end-of-day data for equities, fixed income, and FX
- Real-time streaming for live portfolio monitoring
- Corporate actions: dividends, splits, mergers

Requires: Bloomberg Terminal license + ``pip install blpapi``

References
----------
Bloomberg Open API: https://www.bloomberg.com/professional/support/api-library/
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import blpapi  # type: ignore[import-untyped]

    _HAS_BLPAPI = True
except ImportError:
    blpapi = None  # type: ignore[assignment]
    _HAS_BLPAPI = False


def _require_blpapi() -> None:
    """Raise ImportError if blpapi is not installed."""
    if not _HAS_BLPAPI:
        raise ImportError(
            "blpapi is required for Bloomberg data. "
            "Install with: pip install blpapi  "
            "(requires Bloomberg Terminal license and C++ SDK)"
        )


# ---------------------------------------------------------------------------
# Field mapping: Bloomberg field -> qufin internal name
# ---------------------------------------------------------------------------

#: Default Bloomberg -> qufin field mapping for historical data.
FIELD_MAP: dict[str, str] = {
    "PX_LAST": "close",
    "PX_OPEN": "open",
    "PX_HIGH": "high",
    "PX_LOW": "low",
    "PX_VOLUME": "volume",
    "OPEN_INT": "open_interest",
    "EQY_WEIGHTED_AVG_PX": "vwap",
    "CUR_MKT_CAP": "market_cap",
    "TOT_RETURN_INDEX_GROSS_DVDS": "total_return",
    "YLD_YTM_MID": "yield_to_maturity",
    "DUR_ADJ_MID": "modified_duration",
    "CONVEXITY": "convexity",
}

#: Bloomberg -> qufin field mapping for corporate actions.
CORP_ACTION_MAP: dict[str, str] = {
    "DVD_EX_DT": "ex_date",
    "DVD_RECORD_DT": "record_date",
    "DVD_PAY_DT": "pay_date",
    "DVD_CRNCY": "currency",
    "IS_DVD_STOCK_TYPE": "is_stock_dividend",
    "DVD_SH_LAST": "dividend_amount",
    "SPLIT_RATIO": "split_ratio",
    "SPLIT_ADJ_FACTOR": "split_adj_factor",
}


class AssetClass(Enum):
    """Supported Bloomberg asset classes."""

    EQUITY = "equity"
    FIXED_INCOME = "fixed_income"
    FX = "fx"
    COMMODITY = "commodity"
    INDEX = "index"


@dataclass
class BloombergConfig:
    """Configuration for Bloomberg session.

    Parameters
    ----------
    host : str
        Bloomberg server host.
    port : int
        Bloomberg server port.
    timeout_ms : int
        Session start timeout in milliseconds.
    max_pending : int
        Maximum pending requests.
    """

    host: str = "localhost"
    port: int = 8194
    timeout_ms: int = 10_000
    max_pending: int = 1024


@dataclass
class CorporateAction:
    """A single corporate action event.

    Parameters
    ----------
    ticker : str
        Bloomberg ticker.
    action_type : str
        One of "dividend", "split", "merger".
    ex_date : str
        Ex-date as YYYY-MM-DD string.
    details : dict[str, Any]
        Additional action-specific fields.
    """

    ticker: str
    action_type: str
    ex_date: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamTick:
    """A single real-time market data tick.

    Parameters
    ----------
    ticker : str
        Bloomberg ticker.
    field : str
        qufin-normalised field name.
    value : float
        Tick value.
    timestamp : datetime
        Tick timestamp.
    """

    ticker: str
    field: str
    value: float
    timestamp: datetime


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def normalize_fields(
    raw: dict[str, Any],
    field_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Translate Bloomberg field names to qufin internal names.

    Parameters
    ----------
    raw : dict
        Raw Bloomberg key-value pairs.
    field_map : dict | None
        Custom mapping; defaults to ``FIELD_MAP``.

    Returns
    -------
    dict with translated keys. Unmapped keys are passed through lower-cased.
    """
    fmap = field_map or FIELD_MAP
    out: dict[str, Any] = {}
    for k, v in raw.items():
        mapped = fmap.get(k)
        if mapped is not None:
            out[mapped] = v
        else:
            out[k.lower()] = v
    return out


def normalize_dataframe(
    df: pd.DataFrame,
    field_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Rename DataFrame columns from Bloomberg to qufin names.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with Bloomberg column names.
    field_map : dict | None
        Custom mapping; defaults to ``FIELD_MAP``.

    Returns
    -------
    pd.DataFrame with renamed columns.
    """
    fmap = field_map or FIELD_MAP
    rename = {}
    for col in df.columns:
        if col in fmap:
            rename[col] = fmap[col]
        else:
            rename[col] = col.lower()
    return df.rename(columns=rename)


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------

class BloombergSession:
    """Manages a blpapi session lifecycle.

    Parameters
    ----------
    config : BloombergConfig | None
        Session configuration. Uses defaults if None.
    """

    def __init__(self, config: BloombergConfig | None = None) -> None:
        _require_blpapi()
        self._config = config or BloombergConfig()
        self._session: Any | None = None
        self._ref_service: Any | None = None

    @property
    def connected(self) -> bool:
        """Whether the session is currently active."""
        return self._session is not None

    def start(self) -> None:
        """Open a Bloomberg session and open the reference data service.

        Raises
        ------
        ConnectionError
            If the session cannot be started or the service cannot be opened.
        """
        _require_blpapi()

        opts = blpapi.SessionOptions()
        opts.setServerHost(self._config.host)
        opts.setServerPort(self._config.port)
        opts.setMaxPendingRequests(self._config.max_pending)

        session = blpapi.Session(opts)
        if not session.start():
            raise ConnectionError(
                f"Failed to start Bloomberg session at "
                f"{self._config.host}:{self._config.port}"
            )

        if not session.openService("//blp/refdata"):
            session.stop()
            raise ConnectionError("Failed to open //blp/refdata service")

        self._session = session
        self._ref_service = session.getService("//blp/refdata")
        logger.info("Bloomberg session started on %s:%d", self._config.host, self._config.port)

    def stop(self) -> None:
        """Close the Bloomberg session."""
        if self._session is not None:
            self._session.stop()
            self._session = None
            self._ref_service = None
            logger.info("Bloomberg session stopped")

    def _ensure_started(self) -> None:
        """Start the session if not already connected."""
        if not self.connected:
            self.start()

    def __enter__(self) -> BloombergSession:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()


# ---------------------------------------------------------------------------
# BloombergDataSource — main public API
# ---------------------------------------------------------------------------

class BloombergDataSource:
    """Fetch market data from Bloomberg Terminal via blpapi.

    Parameters
    ----------
    config : BloombergConfig | None
        Session parameters. Defaults to localhost:8194.
    field_map : dict[str, str] | None
        Bloomberg -> qufin field mapping override.
    cache : bool
        Whether to cache downloaded data to parquet via qufin cache layer.

    Examples
    --------
    >>> from qufin.data.bloomberg import BloombergDataSource
    >>> bbg = BloombergDataSource()
    >>> prices = bbg.get_historical(
    ...     ["AAPL US Equity", "MSFT US Equity"],
    ...     start="2024-01-01", end="2024-06-30",
    ... )
    """

    def __init__(
        self,
        config: BloombergConfig | None = None,
        field_map: dict[str, str] | None = None,
        cache: bool = True,
    ) -> None:
        _require_blpapi()
        self._session = BloombergSession(config)
        self._field_map = field_map or FIELD_MAP
        self._cache = cache
        self._subscribers: dict[str, Any] = {}

    # ---- context manager ------------------------------------------------

    def __enter__(self) -> BloombergDataSource:
        self._session.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        """Stop the session and clean up subscriptions."""
        self._subscribers.clear()
        self._session.stop()

    # ---- helpers --------------------------------------------------------

    def _get_session(self) -> Any:
        """Return the underlying blpapi session, starting if needed."""
        self._session._ensure_started()
        return self._session._session

    def _get_refdata_service(self) -> Any:
        """Return the //blp/refdata service handle."""
        self._session._ensure_started()
        return self._session._ref_service

    # ---- cache layer ----------------------------------------------------

    def _try_cache_get(self, prefix: str, *args: object) -> pd.DataFrame | None:
        """Attempt to load data from the parquet cache."""
        if not self._cache:
            return None
        try:
            from qufin.data.cache import get_cached
            return get_cached(prefix, *args)
        except Exception:
            return None

    def _try_cache_put(self, df: pd.DataFrame, prefix: str, *args: object) -> None:
        """Attempt to store data in the parquet cache."""
        if not self._cache:
            return
        try:
            from qufin.data.cache import put_cache
            put_cache(df, prefix, *args)
        except Exception:
            logger.debug("Cache write failed for %s", prefix, exc_info=True)

    # ---- historical data ------------------------------------------------

    def get_historical(
        self,
        tickers: list[str],
        start: str,
        end: str,
        fields: list[str] | None = None,
        frequency: Literal["DAILY", "WEEKLY", "MONTHLY"] = "DAILY",
        currency: str | None = None,
    ) -> pd.DataFrame:
        """Fetch historical end-of-day data.

        Parameters
        ----------
        tickers : list[str]
            Bloomberg tickers (e.g. ``["AAPL US Equity"]``).
        start, end : str
            Date strings ``YYYY-MM-DD``.
        fields : list[str] | None
            Bloomberg fields to request. Defaults to ``["PX_LAST"]``.
        frequency : str
            ``DAILY``, ``WEEKLY``, or ``MONTHLY``.
        currency : str | None
            Override currency for cross-asset comparison (e.g. ``"USD"``).

        Returns
        -------
        pd.DataFrame
            Multi-indexed by (date, ticker) or pivoted with tickers as columns
            when a single field is requested.

        Raises
        ------
        ValueError
            If tickers list is empty.
        ConnectionError
            If the Bloomberg session cannot be established.
        """
        if not tickers:
            raise ValueError("tickers list must not be empty")

        fields = fields or ["PX_LAST"]

        # Check cache
        cache_key_args = (tuple(tickers), start, end, tuple(fields), frequency)
        cached = self._try_cache_get("bbg_hist", *cache_key_args)
        if cached is not None:
            return cached

        session = self._get_session()
        service = self._get_refdata_service()

        request = service.createRequest("HistoricalDataRequest")

        for t in tickers:
            request.getElement("securities").appendValue(t)
        for f in fields:
            request.getElement("fields").appendValue(f)

        request.set("startDate", start.replace("-", ""))
        request.set("endDate", end.replace("-", ""))
        request.set("periodicitySelection", frequency)
        if currency:
            request.set("currency", currency)

        session.sendRequest(request)

        frames: list[pd.DataFrame] = []
        done = False
        while not done:
            event = session.nextEvent(self._session._config.timeout_ms)
            for msg in event:
                if msg.hasElement("securityData"):
                    sec_data = msg.getElement("securityData")
                    ticker = str(sec_data.getElementAsString("security"))
                    field_data = sec_data.getElement("fieldData")

                    rows: list[dict[str, Any]] = []
                    for i in range(field_data.numValues()):
                        row_elem = field_data.getValueAsElement(i)
                        row: dict[str, Any] = {"ticker": ticker}
                        row["date"] = pd.Timestamp(
                            row_elem.getElementAsString("date")
                        )
                        for f in fields:
                            if row_elem.hasElement(f):
                                row[f] = row_elem.getElementAsFloat(f)
                            else:
                                row[f] = np.nan
                        rows.append(row)

                    if rows:
                        frames.append(pd.DataFrame(rows))

            if event.eventType() == blpapi.Event.RESPONSE:
                done = True

        if not frames:
            result = pd.DataFrame(columns=["date", "ticker", *fields])
        else:
            result = pd.concat(frames, ignore_index=True)

        # Normalise field names
        result = normalize_dataframe(result, self._field_map)

        # Pivot for single-field convenience
        if len(fields) == 1:
            mapped_field = self._field_map.get(fields[0], fields[0].lower())
            result = result.pivot(
                index="date", columns="ticker", values=mapped_field
            )
            result.index.name = None

        self._try_cache_put(result, "bbg_hist", *cache_key_args)
        return result

    def get_prices(
        self,
        tickers: list[str],
        start: str,
        end: str,
        frequency: Literal["DAILY", "WEEKLY", "MONTHLY"] = "DAILY",
    ) -> pd.DataFrame:
        """Fetch closing prices — convenience wrapper around get_historical.

        Returns
        -------
        pd.DataFrame
            Columns are tickers, index is datetime.
        """
        return self.get_historical(
            tickers, start, end, fields=["PX_LAST"], frequency=frequency
        )

    def get_returns(
        self,
        tickers: list[str],
        start: str,
        end: str,
        frequency: Literal["DAILY", "WEEKLY", "MONTHLY"] = "DAILY",
    ) -> pd.DataFrame:
        """Fetch log returns.

        Returns
        -------
        pd.DataFrame
            Log returns, columns are tickers.
        """
        prices = self.get_prices(tickers, start, end, frequency=frequency)
        returns: pd.DataFrame = np.log(prices / prices.shift(1)).dropna()
        return returns

    # ---- reference / snapshot data --------------------------------------

    def get_reference(
        self,
        tickers: list[str],
        fields: list[str],
    ) -> pd.DataFrame:
        """Fetch current reference (snapshot) data.

        Parameters
        ----------
        tickers : list[str]
            Bloomberg tickers.
        fields : list[str]
            Bloomberg fields (e.g. ``["PX_LAST", "CUR_MKT_CAP"]``).

        Returns
        -------
        pd.DataFrame
            One row per ticker, columns are normalised field names.
        """
        if not tickers:
            raise ValueError("tickers list must not be empty")

        session = self._get_session()
        service = self._get_refdata_service()

        request = service.createRequest("ReferenceDataRequest")
        for t in tickers:
            request.getElement("securities").appendValue(t)
        for f in fields:
            request.getElement("fields").appendValue(f)

        session.sendRequest(request)

        rows: list[dict[str, Any]] = []
        done = False
        while not done:
            event = session.nextEvent(self._session._config.timeout_ms)
            for msg in event:
                if msg.hasElement("securityData"):
                    sec_arr = msg.getElement("securityData")
                    for i in range(sec_arr.numValues()):
                        sec = sec_arr.getValueAsElement(i)
                        ticker = str(sec.getElementAsString("security"))
                        fd = sec.getElement("fieldData")
                        row: dict[str, Any] = {"ticker": ticker}
                        for f in fields:
                            if fd.hasElement(f):
                                row[f] = fd.getElementAsString(f)
                            else:
                                row[f] = np.nan
                        rows.append(row)

            if event.eventType() == blpapi.Event.RESPONSE:
                done = True

        df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["ticker", *fields])
        return normalize_dataframe(df, self._field_map)

    # ---- corporate actions ----------------------------------------------

    def get_dividends(
        self,
        tickers: list[str],
        start: str,
        end: str,
    ) -> list[CorporateAction]:
        """Fetch dividend history for the given tickers.

        Parameters
        ----------
        tickers : list[str]
            Bloomberg tickers.
        start, end : str
            Date range ``YYYY-MM-DD``.

        Returns
        -------
        list[CorporateAction]
        """
        if not tickers:
            raise ValueError("tickers list must not be empty")

        session = self._get_session()
        service = self._get_refdata_service()

        request = service.createRequest("ReferenceDataRequest")
        for t in tickers:
            request.getElement("securities").appendValue(t)

        request.getElement("fields").appendValue("DVD_HIST_ALL")

        overrides = request.getElement("overrides")
        ovrd_start = overrides.appendElement()
        ovrd_start.setElement("fieldId", "DVD_START_DT")
        ovrd_start.setElement("value", start.replace("-", ""))
        ovrd_end = overrides.appendElement()
        ovrd_end.setElement("fieldId", "DVD_END_DT")
        ovrd_end.setElement("value", end.replace("-", ""))

        session.sendRequest(request)

        actions: list[CorporateAction] = []
        done = False
        while not done:
            event = session.nextEvent(self._session._config.timeout_ms)
            for msg in event:
                if msg.hasElement("securityData"):
                    sec_arr = msg.getElement("securityData")
                    for i in range(sec_arr.numValues()):
                        sec = sec_arr.getValueAsElement(i)
                        ticker = str(sec.getElementAsString("security"))
                        fd = sec.getElement("fieldData")
                        if fd.hasElement("DVD_HIST_ALL"):
                            dvd_arr = fd.getElement("DVD_HIST_ALL")
                            for j in range(dvd_arr.numValues()):
                                dvd = dvd_arr.getValueAsElement(j)
                                details: dict[str, Any] = {}
                                ex_date = ""
                                for bfield, qfield in CORP_ACTION_MAP.items():
                                    if dvd.hasElement(bfield):
                                        val = str(dvd.getElementAsString(bfield))
                                        details[qfield] = val
                                        if qfield == "ex_date":
                                            ex_date = val
                                actions.append(
                                    CorporateAction(
                                        ticker=ticker,
                                        action_type="dividend",
                                        ex_date=ex_date,
                                        details=details,
                                    )
                                )

            if event.eventType() == blpapi.Event.RESPONSE:
                done = True

        return actions

    def get_splits(
        self,
        tickers: list[str],
        start: str,
        end: str,
    ) -> list[CorporateAction]:
        """Fetch stock split history.

        Parameters
        ----------
        tickers : list[str]
            Bloomberg tickers.
        start, end : str
            Date range ``YYYY-MM-DD``.

        Returns
        -------
        list[CorporateAction]
        """
        if not tickers:
            raise ValueError("tickers list must not be empty")

        session = self._get_session()
        service = self._get_refdata_service()

        request = service.createRequest("ReferenceDataRequest")
        for t in tickers:
            request.getElement("securities").appendValue(t)

        request.getElement("fields").appendValue("SPLIT_HIST")

        overrides = request.getElement("overrides")
        ovrd_start = overrides.appendElement()
        ovrd_start.setElement("fieldId", "START_DT")
        ovrd_start.setElement("value", start.replace("-", ""))
        ovrd_end = overrides.appendElement()
        ovrd_end.setElement("fieldId", "END_DT")
        ovrd_end.setElement("value", end.replace("-", ""))

        session.sendRequest(request)

        actions: list[CorporateAction] = []
        done = False
        while not done:
            event = session.nextEvent(self._session._config.timeout_ms)
            for msg in event:
                if msg.hasElement("securityData"):
                    sec_arr = msg.getElement("securityData")
                    for i in range(sec_arr.numValues()):
                        sec = sec_arr.getValueAsElement(i)
                        ticker = str(sec.getElementAsString("security"))
                        fd = sec.getElement("fieldData")
                        if fd.hasElement("SPLIT_HIST"):
                            split_arr = fd.getElement("SPLIT_HIST")
                            for j in range(split_arr.numValues()):
                                sp = split_arr.getValueAsElement(j)
                                details: dict[str, Any] = {}
                                ex_date = ""
                                if sp.hasElement("Split Date"):
                                    ex_date = str(sp.getElementAsString("Split Date"))
                                if sp.hasElement("Split Ratio"):
                                    details["split_ratio"] = str(
                                        sp.getElementAsString("Split Ratio")
                                    )
                                actions.append(
                                    CorporateAction(
                                        ticker=ticker,
                                        action_type="split",
                                        ex_date=ex_date,
                                        details=details,
                                    )
                                )

            if event.eventType() == blpapi.Event.RESPONSE:
                done = True

        return actions

    # ---- real-time streaming --------------------------------------------

    def subscribe(
        self,
        tickers: list[str],
        fields: list[str] | None = None,
        callback: Callable[[StreamTick], None] | None = None,
        interval: float = 0.0,
    ) -> None:
        """Subscribe to real-time market data.

        Parameters
        ----------
        tickers : list[str]
            Bloomberg tickers to subscribe to.
        fields : list[str] | None
            Fields to stream. Defaults to ``["LAST_PRICE", "BID", "ASK"]``.
        callback : callable | None
            Called with a ``StreamTick`` for each incoming update. If None,
            ticks are logged at DEBUG level.
        interval : float
            Minimum interval in seconds between updates (0 = every tick).

        Raises
        ------
        ValueError
            If tickers list is empty.
        """
        if not tickers:
            raise ValueError("tickers list must not be empty")

        fields = fields or ["LAST_PRICE", "BID", "ASK"]

        session = self._get_session()

        sub_list = blpapi.SubscriptionList()
        for t in tickers:
            options_str = f"interval={interval}" if interval > 0 else ""
            sub_list.add(
                t,
                ",".join(fields),
                options_str,
                blpapi.CorrelationId(t),
            )

        session.subscribe(sub_list)
        self._subscribers[id(sub_list)] = {
            "sub_list": sub_list,
            "callback": callback,
            "fields": fields,
        }
        logger.info("Subscribed to %d tickers", len(tickers))

    def unsubscribe_all(self) -> None:
        """Cancel all active real-time subscriptions."""
        if not self._subscribers:
            return
        session = self._get_session()
        for info in self._subscribers.values():
            session.unsubscribe(info["sub_list"])
        self._subscribers.clear()
        logger.info("All subscriptions cancelled")

    def poll_events(self, timeout_ms: int = 500, max_events: int = 100) -> list[StreamTick]:
        """Poll for streaming events and return collected ticks.

        Parameters
        ----------
        timeout_ms : int
            Timeout per event poll.
        max_events : int
            Maximum number of events to process in this call.

        Returns
        -------
        list[StreamTick]
        """
        session = self._get_session()
        ticks: list[StreamTick] = []

        for _ in range(max_events):
            event = session.nextEvent(timeout_ms)
            if event.eventType() in (
                blpapi.Event.SUBSCRIPTION_DATA,
                blpapi.Event.SUBSCRIPTION_STATUS,
            ):
                for msg in event:
                    cid = msg.correlationIds()
                    ticker = str(cid[0].value()) if cid else "UNKNOWN"
                    now = datetime.now()

                    for info in self._subscribers.values():
                        for f in info["fields"]:
                            if msg.hasElement(f):
                                mapped = self._field_map.get(f, f.lower())
                                val = msg.getElementAsFloat(f)
                                tick = StreamTick(
                                    ticker=ticker,
                                    field=mapped,
                                    value=val,
                                    timestamp=now,
                                )
                                ticks.append(tick)
                                if info["callback"] is not None:
                                    info["callback"](tick)
                                else:
                                    logger.debug("Tick: %s", tick)
            if event.eventType() == blpapi.Event.TIMEOUT:
                break

        return ticks
