"""Tests for Bloomberg data provider.

blpapi is not installed in CI — every test mocks it completely.
"""
# ruff: noqa: N802 — camelCase method names match blpapi's Java-style API

from __future__ import annotations

import sys
from datetime import datetime
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Build a fake blpapi module so imports succeed
# ---------------------------------------------------------------------------

def _make_fake_blpapi() -> ModuleType:
    """Create a minimal mock blpapi module."""
    mod = ModuleType("blpapi")

    # Event type constants
    class _Event:
        RESPONSE = 1
        PARTIAL_RESPONSE = 2
        SUBSCRIPTION_DATA = 3
        SUBSCRIPTION_STATUS = 4
        TIMEOUT = 5

    mod.Event = _Event  # type: ignore[attr-defined]
    mod.SessionOptions = MagicMock  # type: ignore[attr-defined]
    mod.Session = MagicMock  # type: ignore[attr-defined]
    mod.SubscriptionList = MagicMock  # type: ignore[attr-defined]
    mod.CorrelationId = MagicMock  # type: ignore[attr-defined]
    return mod


@pytest.fixture(autouse=True)
def _inject_fake_blpapi(monkeypatch: pytest.MonkeyPatch):
    """Inject fake blpapi into sys.modules before each test."""
    fake = _make_fake_blpapi()
    monkeypatch.setitem(sys.modules, "blpapi", fake)

    # Patch the module-level state in bloomberg.py
    import qufin.data.bloomberg as bbg_mod
    monkeypatch.setattr(bbg_mod, "blpapi", fake)
    monkeypatch.setattr(bbg_mod, "_HAS_BLPAPI", True)

    yield fake


# ---------------------------------------------------------------------------
# Helpers: mock session / service / messages
# ---------------------------------------------------------------------------

class MockElement:
    """Simulates a blpapi Element for tests."""

    def __init__(self, data: dict[str, Any] | None = None, values: list | None = None):
        self._data = data or {}
        self._values = values or []

    def hasElement(self, name: str) -> bool:
        return name in self._data

    def getElementAsString(self, name: str) -> str:
        return str(self._data[name])

    def getElementAsFloat(self, name: str) -> float:
        return float(self._data[name])

    def getElement(self, name: str) -> MockElement:
        val = self._data.get(name)
        if isinstance(val, MockElement):
            return val
        return MockElement(data=val if isinstance(val, dict) else {})

    def numValues(self) -> int:
        return len(self._values)

    def getValueAsElement(self, idx: int) -> MockElement:
        v = self._values[idx]
        if isinstance(v, MockElement):
            return v
        return MockElement(data=v)

    def appendValue(self, value: Any) -> None:
        self._values.append(value)

    def appendElement(self) -> MockElement:
        elem = MockElement()
        self._values.append(elem)
        return elem

    def setElement(self, name: str, value: Any) -> None:
        self._data[name] = value

    def set(self, name: str, value: Any) -> None:
        self._data[name] = value


class MockMessage:
    def __init__(self, data: dict[str, Any]):
        self._elem = MockElement(data)

    def hasElement(self, name: str) -> bool:
        return self._elem.hasElement(name)

    def getElement(self, name: str) -> MockElement:
        return self._elem.getElement(name)

    def getElementAsString(self, name: str) -> str:
        return self._elem.getElementAsString(name)

    def getElementAsFloat(self, name: str) -> float:
        return self._elem.getElementAsFloat(name)

    def correlationIds(self) -> list:
        cids = self._elem._data.get("_cids", [])
        return cids


class MockEvent:
    def __init__(self, messages: list[MockMessage], event_type: int = 1):
        self._messages = messages
        self._event_type = event_type

    def eventType(self) -> int:
        return self._event_type

    def __iter__(self):
        return iter(self._messages)


def _build_session_mock(events: list[MockEvent]) -> MagicMock:
    """Build a mock blpapi.Session that yields the given events."""
    session = MagicMock()
    session.start.return_value = True
    session.openService.return_value = True

    service = MagicMock()
    service.createRequest.return_value = MockElement()
    session.getService.return_value = service

    event_iter = iter(events)
    session.nextEvent.side_effect = lambda timeout=0: next(event_iter)

    return session


# ---------------------------------------------------------------------------
# Tests: module-level helpers
# ---------------------------------------------------------------------------

class TestRequireBlpapi:
    def test_import_error_when_missing(self, monkeypatch: pytest.MonkeyPatch):
        import qufin.data.bloomberg as bbg_mod
        monkeypatch.setattr(bbg_mod, "_HAS_BLPAPI", False)
        with pytest.raises(ImportError, match="blpapi is required"):
            bbg_mod._require_blpapi()

    def test_no_error_when_present(self):
        from qufin.data.bloomberg import _require_blpapi
        _require_blpapi()  # should not raise


class TestNormalizeFields:
    def test_default_mapping(self):
        from qufin.data.bloomberg import normalize_fields
        raw = {"PX_LAST": 150.0, "PX_VOLUME": 1_000_000}
        result = normalize_fields(raw)
        assert result == {"close": 150.0, "volume": 1_000_000}

    def test_custom_mapping(self):
        from qufin.data.bloomberg import normalize_fields
        raw = {"CUSTOM_FIELD": 42}
        result = normalize_fields(raw, field_map={"CUSTOM_FIELD": "my_field"})
        assert result == {"my_field": 42}

    def test_unmapped_lowercased(self):
        from qufin.data.bloomberg import normalize_fields
        raw = {"UNKNOWN_FIELD": 99}
        result = normalize_fields(raw)
        assert result == {"unknown_field": 99}


class TestNormalizeDataFrame:
    def test_renames_columns(self):
        from qufin.data.bloomberg import normalize_dataframe
        df = pd.DataFrame({"PX_LAST": [1, 2], "PX_VOLUME": [100, 200]})
        result = normalize_dataframe(df)
        assert list(result.columns) == ["close", "volume"]

    def test_custom_map(self):
        from qufin.data.bloomberg import normalize_dataframe
        df = pd.DataFrame({"A": [1], "B": [2]})
        result = normalize_dataframe(df, field_map={"A": "alpha"})
        assert "alpha" in result.columns
        assert "b" in result.columns  # unmapped lowercased


# ---------------------------------------------------------------------------
# Tests: dataclasses
# ---------------------------------------------------------------------------

class TestDataclasses:
    def test_bloomberg_config_defaults(self):
        from qufin.data.bloomberg import BloombergConfig
        cfg = BloombergConfig()
        assert cfg.host == "localhost"
        assert cfg.port == 8194
        assert cfg.timeout_ms == 10_000

    def test_corporate_action(self):
        from qufin.data.bloomberg import CorporateAction
        ca = CorporateAction(ticker="AAPL", action_type="dividend", ex_date="2024-01-15")
        assert ca.ticker == "AAPL"
        assert ca.details == {}

    def test_stream_tick(self):
        from qufin.data.bloomberg import StreamTick
        now = datetime.now()
        tick = StreamTick(ticker="AAPL", field="close", value=150.0, timestamp=now)
        assert tick.value == 150.0

    def test_asset_class_enum(self):
        from qufin.data.bloomberg import AssetClass
        assert AssetClass.EQUITY.value == "equity"
        assert AssetClass.FX.value == "fx"


# ---------------------------------------------------------------------------
# Tests: BloombergSession
# ---------------------------------------------------------------------------

class TestBloombergSession:
    def test_start_success(self, _inject_fake_blpapi):
        fake = _inject_fake_blpapi
        from qufin.data.bloomberg import BloombergSession

        mock_sess = MagicMock()
        mock_sess.start.return_value = True
        mock_sess.openService.return_value = True
        mock_sess.getService.return_value = MagicMock()
        fake.Session = MagicMock(return_value=mock_sess)

        bs = BloombergSession()
        bs.start()
        assert bs.connected
        bs.stop()
        assert not bs.connected

    def test_start_failure(self, _inject_fake_blpapi):
        fake = _inject_fake_blpapi
        from qufin.data.bloomberg import BloombergSession

        mock_sess = MagicMock()
        mock_sess.start.return_value = False
        fake.Session = MagicMock(return_value=mock_sess)

        bs = BloombergSession()
        with pytest.raises(ConnectionError, match="Failed to start"):
            bs.start()

    def test_service_open_failure(self, _inject_fake_blpapi):
        fake = _inject_fake_blpapi
        from qufin.data.bloomberg import BloombergSession

        mock_sess = MagicMock()
        mock_sess.start.return_value = True
        mock_sess.openService.return_value = False
        fake.Session = MagicMock(return_value=mock_sess)

        bs = BloombergSession()
        with pytest.raises(ConnectionError, match="Failed to open"):
            bs.start()

    def test_context_manager(self, _inject_fake_blpapi):
        fake = _inject_fake_blpapi
        from qufin.data.bloomberg import BloombergSession

        mock_sess = MagicMock()
        mock_sess.start.return_value = True
        mock_sess.openService.return_value = True
        mock_sess.getService.return_value = MagicMock()
        fake.Session = MagicMock(return_value=mock_sess)

        with BloombergSession() as bs:
            assert bs.connected
        assert not bs.connected


# ---------------------------------------------------------------------------
# Tests: BloombergDataSource
# ---------------------------------------------------------------------------

def _make_data_source(events: list[MockEvent], fake_blpapi):
    """Helper: build a BloombergDataSource with mocked session."""
    from qufin.data.bloomberg import BloombergDataSource

    mock_sess = _build_session_mock(events)
    fake_blpapi.Session = MagicMock(return_value=mock_sess)

    ds = BloombergDataSource(cache=False)
    return ds, mock_sess


class TestGetHistorical:
    def test_single_field_pivoted(self, _inject_fake_blpapi):
        fake = _inject_fake_blpapi

        field_data_vals = [
            MockElement({"date": "2024-01-02", "PX_LAST": 150.0}),
            MockElement({"date": "2024-01-03", "PX_LAST": 152.0}),
        ]
        sec_data = MockElement({
            "security": "AAPL US Equity",
            "fieldData": MockElement(values=field_data_vals),
        })
        msg = MockMessage({"securityData": sec_data})
        event = MockEvent([msg], event_type=fake.Event.RESPONSE)

        ds, _ = _make_data_source([event], fake)
        result = ds.get_historical(
            ["AAPL US Equity"], start="2024-01-02", end="2024-01-03"
        )
        assert isinstance(result, pd.DataFrame)
        assert "AAPL US Equity" in result.columns
        assert len(result) == 2

    def test_empty_tickers_raises(self, _inject_fake_blpapi):
        fake = _inject_fake_blpapi
        ds, _ = _make_data_source([], fake)
        with pytest.raises(ValueError, match="tickers list must not be empty"):
            ds.get_historical([], start="2024-01-01", end="2024-01-31")

    def test_empty_response(self, _inject_fake_blpapi):
        fake = _inject_fake_blpapi
        msg = MockMessage({})  # no securityData
        event = MockEvent([msg], event_type=fake.Event.RESPONSE)
        ds, _ = _make_data_source([event], fake)
        result = ds.get_historical(
            ["AAPL US Equity"], start="2024-01-01", end="2024-01-31"
        )
        assert isinstance(result, pd.DataFrame)

    def test_multiple_fields_not_pivoted(self, _inject_fake_blpapi):
        fake = _inject_fake_blpapi
        field_data_vals = [
            MockElement({"date": "2024-01-02", "PX_LAST": 150.0, "PX_VOLUME": 1e6}),
        ]
        sec_data = MockElement({
            "security": "AAPL US Equity",
            "fieldData": MockElement(values=field_data_vals),
        })
        msg = MockMessage({"securityData": sec_data})
        event = MockEvent([msg], event_type=fake.Event.RESPONSE)

        ds, _ = _make_data_source([event], fake)
        result = ds.get_historical(
            ["AAPL US Equity"],
            start="2024-01-02",
            end="2024-01-02",
            fields=["PX_LAST", "PX_VOLUME"],
        )
        assert "close" in result.columns
        assert "volume" in result.columns

    def test_missing_field_returns_nan(self, _inject_fake_blpapi):
        fake = _inject_fake_blpapi
        field_data_vals = [
            MockElement({"date": "2024-01-02", "PX_LAST": 150.0}),
        ]
        sec_data = MockElement({
            "security": "AAPL US Equity",
            "fieldData": MockElement(values=field_data_vals),
        })
        msg = MockMessage({"securityData": sec_data})
        event = MockEvent([msg], event_type=fake.Event.RESPONSE)

        ds, _ = _make_data_source([event], fake)
        result = ds.get_historical(
            ["AAPL US Equity"],
            start="2024-01-02",
            end="2024-01-02",
            fields=["PX_LAST", "PX_VOLUME"],
        )
        assert np.isnan(result["volume"].iloc[0])


class TestGetPrices:
    def test_returns_dataframe(self, _inject_fake_blpapi):
        fake = _inject_fake_blpapi
        field_data_vals = [
            MockElement({"date": "2024-01-02", "PX_LAST": 150.0}),
        ]
        sec_data = MockElement({
            "security": "MSFT US Equity",
            "fieldData": MockElement(values=field_data_vals),
        })
        msg = MockMessage({"securityData": sec_data})
        event = MockEvent([msg], event_type=fake.Event.RESPONSE)

        ds, _ = _make_data_source([event], fake)
        result = ds.get_prices(["MSFT US Equity"], start="2024-01-02", end="2024-01-02")
        assert isinstance(result, pd.DataFrame)


class TestGetReturns:
    def test_log_returns(self, _inject_fake_blpapi):
        fake = _inject_fake_blpapi
        field_data_vals = [
            MockElement({"date": "2024-01-02", "PX_LAST": 100.0}),
            MockElement({"date": "2024-01-03", "PX_LAST": 105.0}),
            MockElement({"date": "2024-01-04", "PX_LAST": 103.0}),
        ]
        sec_data = MockElement({
            "security": "AAPL US Equity",
            "fieldData": MockElement(values=field_data_vals),
        })
        msg = MockMessage({"securityData": sec_data})
        event = MockEvent([msg], event_type=fake.Event.RESPONSE)

        ds, _ = _make_data_source([event], fake)
        result = ds.get_returns(["AAPL US Equity"], start="2024-01-02", end="2024-01-04")
        # log(105/100) ≈ 0.04879, log(103/105) ≈ -0.01923
        assert len(result) == 2
        assert result.iloc[0]["AAPL US Equity"] == pytest.approx(np.log(105.0 / 100.0))


class TestGetReference:
    def test_snapshot_data(self, _inject_fake_blpapi):
        fake = _inject_fake_blpapi

        sec_elem = MockElement({
            "security": "AAPL US Equity",
            "fieldData": MockElement({"PX_LAST": "150.0", "CUR_MKT_CAP": "2500000"}),
        })
        sec_arr = MockElement(values=[sec_elem])
        msg = MockMessage({"securityData": sec_arr})
        event = MockEvent([msg], event_type=fake.Event.RESPONSE)

        ds, _ = _make_data_source([event], fake)
        result = ds.get_reference(
            ["AAPL US Equity"], fields=["PX_LAST", "CUR_MKT_CAP"]
        )
        assert "close" in result.columns
        assert "market_cap" in result.columns

    def test_empty_tickers_raises(self, _inject_fake_blpapi):
        fake = _inject_fake_blpapi
        ds, _ = _make_data_source([], fake)
        with pytest.raises(ValueError):
            ds.get_reference([], fields=["PX_LAST"])


class TestGetDividends:
    def test_returns_corporate_actions(self, _inject_fake_blpapi):
        fake = _inject_fake_blpapi
        from qufin.data.bloomberg import CorporateAction

        dvd_elem = MockElement({
            "DVD_EX_DT": "2024-02-15",
            "DVD_SH_LAST": "0.96",
            "DVD_CRNCY": "USD",
        })
        fd = MockElement({
            "DVD_HIST_ALL": MockElement(values=[dvd_elem]),
        })
        sec_elem = MockElement({"security": "AAPL US Equity", "fieldData": fd})
        sec_arr = MockElement(values=[sec_elem])
        msg = MockMessage({"securityData": sec_arr})
        event = MockEvent([msg], event_type=fake.Event.RESPONSE)

        ds, _ = _make_data_source([event], fake)
        actions = ds.get_dividends(
            ["AAPL US Equity"], start="2024-01-01", end="2024-12-31"
        )
        assert len(actions) == 1
        assert isinstance(actions[0], CorporateAction)
        assert actions[0].action_type == "dividend"
        assert actions[0].details["ex_date"] == "2024-02-15"

    def test_empty_tickers_raises(self, _inject_fake_blpapi):
        fake = _inject_fake_blpapi
        ds, _ = _make_data_source([], fake)
        with pytest.raises(ValueError):
            ds.get_dividends([], start="2024-01-01", end="2024-12-31")


class TestGetSplits:
    def test_returns_split_actions(self, _inject_fake_blpapi):
        fake = _inject_fake_blpapi

        split_elem = MockElement({
            "Split Date": "2024-06-10",
            "Split Ratio": "4:1",
        })
        fd = MockElement({
            "SPLIT_HIST": MockElement(values=[split_elem]),
        })
        sec_elem = MockElement({"security": "NVDA US Equity", "fieldData": fd})
        sec_arr = MockElement(values=[sec_elem])
        msg = MockMessage({"securityData": sec_arr})
        event = MockEvent([msg], event_type=fake.Event.RESPONSE)

        ds, _ = _make_data_source([event], fake)
        actions = ds.get_splits(
            ["NVDA US Equity"], start="2024-01-01", end="2024-12-31"
        )
        assert len(actions) == 1
        assert actions[0].action_type == "split"
        assert actions[0].details["split_ratio"] == "4:1"

    def test_empty_tickers_raises(self, _inject_fake_blpapi):
        fake = _inject_fake_blpapi
        ds, _ = _make_data_source([], fake)
        with pytest.raises(ValueError):
            ds.get_splits([], start="2024-01-01", end="2024-12-31")


class TestSubscribe:
    def test_subscribe_creates_subscription(self, _inject_fake_blpapi):
        fake = _inject_fake_blpapi

        mock_sess = MagicMock()
        mock_sess.start.return_value = True
        mock_sess.openService.return_value = True
        mock_sess.getService.return_value = MagicMock()
        fake.Session = MagicMock(return_value=mock_sess)

        from qufin.data.bloomberg import BloombergDataSource
        ds = BloombergDataSource(cache=False)
        ds.subscribe(["AAPL US Equity"])
        assert len(ds._subscribers) == 1
        mock_sess.subscribe.assert_called_once()

    def test_subscribe_empty_tickers_raises(self, _inject_fake_blpapi):
        fake = _inject_fake_blpapi
        ds, _ = _make_data_source([], fake)
        with pytest.raises(ValueError):
            ds.subscribe([])


class TestUnsubscribeAll:
    def test_clears_subscribers(self, _inject_fake_blpapi):
        fake = _inject_fake_blpapi

        mock_sess = MagicMock()
        mock_sess.start.return_value = True
        mock_sess.openService.return_value = True
        mock_sess.getService.return_value = MagicMock()
        fake.Session = MagicMock(return_value=mock_sess)

        from qufin.data.bloomberg import BloombergDataSource
        ds = BloombergDataSource(cache=False)
        ds.subscribe(["AAPL US Equity"])
        assert len(ds._subscribers) == 1
        ds.unsubscribe_all()
        assert len(ds._subscribers) == 0


class TestPollEvents:
    def test_collects_ticks(self, _inject_fake_blpapi):
        fake = _inject_fake_blpapi

        mock_sess = MagicMock()
        mock_sess.start.return_value = True
        mock_sess.openService.return_value = True
        mock_sess.getService.return_value = MagicMock()
        fake.Session = MagicMock(return_value=mock_sess)

        cid_mock = MagicMock()
        cid_mock.value.return_value = "AAPL US Equity"

        tick_msg = MockMessage({"LAST_PRICE": 155.0, "_cids": [cid_mock]})
        data_event = MockEvent([tick_msg], event_type=fake.Event.SUBSCRIPTION_DATA)
        timeout_event = MockEvent([], event_type=fake.Event.TIMEOUT)

        mock_sess.nextEvent.side_effect = [data_event, timeout_event]

        from qufin.data.bloomberg import BloombergDataSource
        ds = BloombergDataSource(cache=False)
        ds.subscribe(["AAPL US Equity"], fields=["LAST_PRICE"])

        ticks = ds.poll_events(timeout_ms=100, max_events=2)
        assert len(ticks) == 1
        assert ticks[0].ticker == "AAPL US Equity"
        assert ticks[0].value == 155.0


class TestContextManager:
    def test_enter_exit(self, _inject_fake_blpapi):
        fake = _inject_fake_blpapi

        mock_sess = MagicMock()
        mock_sess.start.return_value = True
        mock_sess.openService.return_value = True
        mock_sess.getService.return_value = MagicMock()
        fake.Session = MagicMock(return_value=mock_sess)

        from qufin.data.bloomberg import BloombergDataSource
        with BloombergDataSource(cache=False) as ds:
            assert ds._session.connected
        mock_sess.stop.assert_called()


class TestCacheIntegration:
    def test_cache_disabled_skips(self, _inject_fake_blpapi):
        fake = _inject_fake_blpapi
        from qufin.data.bloomberg import BloombergDataSource

        mock_sess = MagicMock()
        mock_sess.start.return_value = True
        mock_sess.openService.return_value = True
        mock_sess.getService.return_value = MagicMock()
        fake.Session = MagicMock(return_value=mock_sess)

        ds = BloombergDataSource(cache=False)
        assert ds._try_cache_get("test", "a", "b") is None
        # Should not raise
        ds._try_cache_put(pd.DataFrame(), "test", "a", "b")


class TestImportGuard:
    def test_datasource_raises_without_blpapi(self, monkeypatch: pytest.MonkeyPatch):
        import qufin.data.bloomberg as bbg_mod
        monkeypatch.setattr(bbg_mod, "_HAS_BLPAPI", False)
        with pytest.raises(ImportError, match="blpapi is required"):
            bbg_mod.BloombergDataSource()

    def test_session_raises_without_blpapi(self, monkeypatch: pytest.MonkeyPatch):
        import qufin.data.bloomberg as bbg_mod
        monkeypatch.setattr(bbg_mod, "_HAS_BLPAPI", False)
        with pytest.raises(ImportError, match="blpapi is required"):
            bbg_mod.BloombergSession()
