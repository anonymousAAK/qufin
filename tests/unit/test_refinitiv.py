"""Tests for Refinitiv/LSEG data provider.

All tests mock the eikon SDK so they pass WITHOUT eikon installed.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Build a fake ``eikon`` module injected into sys.modules so the import
# guard in refinitiv.py resolves without the real package.
# ---------------------------------------------------------------------------


def _make_mock_eikon() -> types.ModuleType:
    """Create a minimal mock of the eikon module."""
    mod = types.ModuleType("eikon")
    mod.set_app_key = MagicMock()  # type: ignore[attr-defined]
    mod.get_timeseries = MagicMock()  # type: ignore[attr-defined]
    mod.get_data = MagicMock()  # type: ignore[attr-defined]
    mod.get_symbology = MagicMock()  # type: ignore[attr-defined]
    return mod


_mock_eikon = _make_mock_eikon()


@pytest.fixture(autouse=True)
def _inject_eikon(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject the mock eikon module before every test."""
    monkeypatch.setitem(sys.modules, "eikon", _mock_eikon)
    # Reset mocks between tests
    _mock_eikon.set_app_key.reset_mock()  # type: ignore[attr-defined]
    _mock_eikon.get_timeseries.reset_mock()  # type: ignore[attr-defined]
    _mock_eikon.get_data.reset_mock()  # type: ignore[attr-defined]
    _mock_eikon.get_symbology.reset_mock()  # type: ignore[attr-defined]


# Force re-import so refinitiv.py picks up the mock
@pytest.fixture(autouse=True)
def _reimport_refinitiv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure refinitiv module is re-imported with mock eikon."""
    # Remove cached module so it re-imports with our mock eikon
    for key in list(sys.modules):
        if "qufin.data.refinitiv" in key:
            del sys.modules[key]


def _import_module():
    """Import (or re-import) refinitiv after mock injection."""
    import importlib

    if "qufin.data.refinitiv" in sys.modules:
        return importlib.reload(sys.modules["qufin.data.refinitiv"])
    import qufin.data.refinitiv as mod
    return mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _price_df(rics: list[str], n: int = 5) -> pd.DataFrame:
    """Create a simple price DataFrame for testing."""
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    data = {ric: 100.0 + np.arange(n, dtype=float) for ric in rics}
    return pd.DataFrame(data, index=idx)


def _snapshot_df(rics: list[str], fields: list[str]) -> pd.DataFrame:
    """Create a snapshot DataFrame for testing."""
    data = {"Instrument": rics}
    for f in fields:
        data[f] = [1.0] * len(rics)
    return pd.DataFrame(data)


# ===================================================================
# Tests
# ===================================================================


class TestRefinitivConfig:
    """Tests for RefinitivConfig dataclass."""

    def test_defaults(self) -> None:
        mod = _import_module()
        cfg = mod.RefinitivConfig()
        assert cfg.app_key == ""
        assert cfg.timeout == 30
        assert cfg.cache is True

    def test_custom_values(self) -> None:
        mod = _import_module()
        cfg = mod.RefinitivConfig(app_key="abc123", timeout=60, cache=False)
        assert cfg.app_key == "abc123"
        assert cfg.timeout == 60
        assert cfg.cache is False


class TestResultDataclasses:
    """Tests for TimeSeriesResult and SnapshotResult."""

    def test_timeseries_result(self) -> None:
        mod = _import_module()
        df = pd.DataFrame({"A": [1, 2, 3]})
        r = mod.TimeSeriesResult(data=df, rics=["A"], fields=["CLOSE"])
        assert r.rics == ["A"]
        assert r.fields == ["CLOSE"]
        assert r.metadata == {}
        pd.testing.assert_frame_equal(r.data, df)

    def test_snapshot_result(self) -> None:
        mod = _import_module()
        df = pd.DataFrame({"X": [10]})
        r = mod.SnapshotResult(data=df, rics=["X"], fields=["F1"])
        assert r.rics == ["X"]
        assert r.fields == ["F1"]


class TestImportGuard:
    """Test that missing eikon raises ImportError."""

    def test_missing_eikon_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Remove mock eikon temporarily
        monkeypatch.delitem(sys.modules, "eikon", raising=False)
        # Remove cached refinitiv module
        for key in list(sys.modules):
            if "qufin.data.refinitiv" in key:
                del sys.modules[key]
        # Patch the built-in import to make eikon fail
        import builtins
        real_import = builtins.__import__

        def _fake_import(name, *args, **kwargs):
            if name == "eikon":
                raise ImportError("No module named 'eikon'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fake_import)
        # Re-import
        for key in list(sys.modules):
            if "qufin.data.refinitiv" in key:
                del sys.modules[key]
        import importlib
        mod = importlib.import_module("qufin.data.refinitiv")
        with pytest.raises(ImportError, match="eikon"):
            mod.RefinitivDataSource()


class TestRefinitivDataSourceInit:
    """Tests for RefinitivDataSource construction."""

    def test_default_config(self) -> None:
        mod = _import_module()
        src = mod.RefinitivDataSource()
        assert src._timeout == 30

    def test_custom_config_sets_app_key(self) -> None:
        mod = _import_module()
        cfg = mod.RefinitivConfig(app_key="mykey", timeout=45)
        src = mod.RefinitivDataSource(config=cfg)
        _mock_eikon.set_app_key.assert_called_with("mykey")  # type: ignore[attr-defined]
        assert src._timeout == 45

    def test_set_app_key_runtime(self) -> None:
        mod = _import_module()
        src = mod.RefinitivDataSource()
        src.set_app_key("newkey")
        _mock_eikon.set_app_key.assert_called_with("newkey")  # type: ignore[attr-defined]
        assert src._config.app_key == "newkey"


class TestEquityMethods:
    """Tests for equity data methods."""

    def test_get_equity_prices(self) -> None:
        mod = _import_module()
        rics = ["AAPL.O", "MSFT.O"]
        df = _price_df(rics)
        _mock_eikon.get_timeseries.return_value = df  # type: ignore[attr-defined]

        src = mod.RefinitivDataSource()
        result = src.get_equity_prices(rics, "2024-01-01", "2024-01-07")

        assert isinstance(result, mod.TimeSeriesResult)
        assert result.rics == rics
        assert result.fields == ["CLOSE"]
        assert result.metadata["interval"] == "daily"
        _mock_eikon.get_timeseries.assert_called_once()  # type: ignore[attr-defined]

    def test_get_equity_ohlcv(self) -> None:
        mod = _import_module()
        rics = ["AAPL.O"]
        idx = pd.date_range("2024-01-01", periods=3, freq="B")
        df = pd.DataFrame(
            {
                "OPEN": [100, 101, 102],
                "HIGH": [105, 106, 107],
                "LOW": [99, 100, 101],
                "CLOSE": [104, 105, 106],
                "VOLUME": [1e6, 1.1e6, 1.2e6],
            },
            index=idx,
        )
        _mock_eikon.get_timeseries.return_value = df  # type: ignore[attr-defined]

        src = mod.RefinitivDataSource()
        result = src.get_equity_ohlcv(rics, "2024-01-01", "2024-01-05")

        assert result.fields == ["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]
        assert len(result.data) == 3

    def test_get_equity_returns(self) -> None:
        mod = _import_module()
        rics = ["AAPL.O"]
        df = _price_df(rics, n=5)
        _mock_eikon.get_timeseries.return_value = df  # type: ignore[attr-defined]

        src = mod.RefinitivDataSource()
        returns = src.get_equity_returns(rics, "2024-01-01", "2024-01-07")

        assert isinstance(returns, pd.DataFrame)
        # 5 prices -> 4 returns
        assert len(returns) == 4
        # All values should be log returns
        expected = np.log(df / df.shift(1)).dropna()
        pd.testing.assert_frame_equal(returns, expected)

    def test_get_equity_returns_empty(self) -> None:
        mod = _import_module()
        _mock_eikon.get_timeseries.return_value = pd.DataFrame()  # type: ignore[attr-defined]
        src = mod.RefinitivDataSource()
        returns = src.get_equity_returns(["X"], "2024-01-01", "2024-01-07")
        assert returns.empty

    def test_get_equity_prices_weekly(self) -> None:
        mod = _import_module()
        rics = ["AAPL.O"]
        df = _price_df(rics, n=3)
        _mock_eikon.get_timeseries.return_value = df  # type: ignore[attr-defined]

        src = mod.RefinitivDataSource()
        src.get_equity_prices(rics, "2024-01-01", "2024-02-01", interval="weekly")

        call_kwargs = _mock_eikon.get_timeseries.call_args  # type: ignore[attr-defined]
        assert call_kwargs[1]["interval"] == "weekly" or call_kwargs.kwargs["interval"] == "weekly"


class TestFixedIncomeMethods:
    """Tests for fixed income data methods."""

    def test_get_bond_data_default_fields(self) -> None:
        mod = _import_module()
        rics = ["US10YT=RR"]
        df = _snapshot_df(rics, mod.RefinitivDataSource.FIXED_INCOME_FIELDS)
        _mock_eikon.get_data.return_value = (df, None)  # type: ignore[attr-defined]

        src = mod.RefinitivDataSource()
        result = src.get_bond_data(rics)

        assert isinstance(result, mod.SnapshotResult)
        assert result.rics == rics

    def test_get_bond_data_custom_fields(self) -> None:
        mod = _import_module()
        rics = ["US10YT=RR"]
        fields = ["TR.MIDYIELD"]
        df = _snapshot_df(rics, fields)
        _mock_eikon.get_data.return_value = (df, None)  # type: ignore[attr-defined]

        src = mod.RefinitivDataSource()
        result = src.get_bond_data(rics, fields=fields)
        assert result.fields == fields

    def test_get_bond_data_error(self) -> None:
        mod = _import_module()
        _mock_eikon.get_data.return_value = (pd.DataFrame(), "timeout")  # type: ignore[attr-defined]

        src = mod.RefinitivDataSource()
        with pytest.raises(RuntimeError, match="Eikon get_data error"):
            src.get_bond_data(["US10YT=RR"])

    def test_get_yield_curve_usd(self) -> None:
        mod = _import_module()
        rics_usd = [
            "US3MT=RR", "US6MT=RR", "US1YT=RR", "US2YT=RR",
            "US5YT=RR", "US10YT=RR", "US30YT=RR",
        ]
        df = pd.DataFrame({
            "Instrument": rics_usd,
            "TR.BIDYIELD": [4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6],
            "TR.ASKYIELD": [4.01, 4.11, 4.21, 4.31, 4.41, 4.51, 4.61],
            "TR.MIDYIELD": [4.005, 4.105, 4.205, 4.305, 4.405, 4.505, 4.605],
        })
        _mock_eikon.get_data.return_value = (df, None)  # type: ignore[attr-defined]

        src = mod.RefinitivDataSource()
        result = src.get_yield_curve()

        assert "maturity" in result.columns
        assert list(result["maturity"]) == ["3M", "6M", "1Y", "2Y", "5Y", "10Y", "30Y"]

    def test_get_yield_curve_unsupported_currency(self) -> None:
        mod = _import_module()
        src = mod.RefinitivDataSource()
        with pytest.raises(ValueError, match="No default RICs"):
            src.get_yield_curve(currency="JPY")


class TestDerivativesMethods:
    """Tests for derivatives data methods."""

    def test_get_option_chain(self) -> None:
        mod = _import_module()
        df = pd.DataFrame({
            "Instrument": ["AAPL250117C00100000.O"],
            "TR.SETTLEMENTPRICE": [5.5],
            "TR.OPENINTEREST": [1000],
            "TR.VOLUME": [500],
            "TR.STRIKEPRICE": [100.0],
            "TR.EXPIRATIONDATE": ["2025-01-17"],
        })
        _mock_eikon.get_data.return_value = (df, None)  # type: ignore[attr-defined]

        src = mod.RefinitivDataSource()
        result = src.get_option_chain("AAPL.O")

        assert isinstance(result, mod.SnapshotResult)
        assert len(result.data) == 1

    def test_get_option_chain_error(self) -> None:
        mod = _import_module()
        _mock_eikon.get_data.return_value = (pd.DataFrame(), "API error")  # type: ignore[attr-defined]

        src = mod.RefinitivDataSource()
        with pytest.raises(RuntimeError, match="Eikon get_data error"):
            src.get_option_chain("AAPL.O")

    def test_get_futures_data(self) -> None:
        mod = _import_module()
        rics = ["CLc1", "ESc1"]
        df = _snapshot_df(rics, ["TR.SETTLEMENTPRICE", "TR.OPENINTEREST", "TR.VOLUME"])
        _mock_eikon.get_data.return_value = (df, None)  # type: ignore[attr-defined]

        src = mod.RefinitivDataSource()
        result = src.get_futures_data(rics)

        assert result.rics == rics
        assert len(result.data) == 2

    def test_get_futures_data_error(self) -> None:
        mod = _import_module()
        _mock_eikon.get_data.return_value = (pd.DataFrame(), "network error")  # type: ignore[attr-defined]

        src = mod.RefinitivDataSource()
        with pytest.raises(RuntimeError, match="Eikon get_data error"):
            src.get_futures_data(["CLc1"])


class TestGenericHelpers:
    """Tests for generic get_data, get_timeseries, and search."""

    def test_get_data(self) -> None:
        mod = _import_module()
        rics = ["AAPL.O"]
        fields = ["TR.PriceClose"]
        df = _snapshot_df(rics, fields)
        _mock_eikon.get_data.return_value = (df, None)  # type: ignore[attr-defined]

        src = mod.RefinitivDataSource()
        result = src.get_data(rics, fields)
        assert result.fields == fields

    def test_get_data_error(self) -> None:
        mod = _import_module()
        _mock_eikon.get_data.return_value = (pd.DataFrame(), "err")  # type: ignore[attr-defined]
        src = mod.RefinitivDataSource()
        with pytest.raises(RuntimeError):
            src.get_data(["X"], ["F"])

    def test_get_timeseries(self) -> None:
        mod = _import_module()
        rics = ["AAPL.O"]
        df = _price_df(rics)
        _mock_eikon.get_timeseries.return_value = df  # type: ignore[attr-defined]

        src = mod.RefinitivDataSource()
        result = src.get_timeseries(rics, ["CLOSE"], "2024-01-01", "2024-01-07")

        assert result.metadata["interval"] == "daily"
        assert result.rics == rics

    def test_search(self) -> None:
        mod = _import_module()
        result_df = pd.DataFrame({"RIC": ["AAPL.O"], "Name": ["Apple Inc"]})
        _mock_eikon.get_symbology.return_value = result_df  # type: ignore[attr-defined]

        src = mod.RefinitivDataSource()
        result = src.search("Apple")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1


class TestClassAttributes:
    """Tests for class-level field lists."""

    def test_equity_fields_present(self) -> None:
        mod = _import_module()
        assert len(mod.RefinitivDataSource.EQUITY_FIELDS) == 5

    def test_fixed_income_fields_present(self) -> None:
        mod = _import_module()
        assert len(mod.RefinitivDataSource.FIXED_INCOME_FIELDS) == 5

    def test_derivatives_fields_present(self) -> None:
        mod = _import_module()
        assert len(mod.RefinitivDataSource.DERIVATIVES_FIELDS) == 5
