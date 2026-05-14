"""Tests for the qufin plugin system."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from qufin.plugins import (
    DataSourcePlugin,
    PluginError,
    PluginInfo,
    _validate_backend,
    clear_data_sources,
    clear_strategies,
    discover_all,
    discover_backends,
    get_data_source,
    get_strategy,
    list_data_sources,
    list_strategies,
    load_backend,
    register_data_source,
    register_strategy,
    unregister_data_source,
    unregister_strategy,
)

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


class _DummyDataSource(DataSourcePlugin):
    """Concrete data source for testing."""

    name = "dummy"

    def fetch(self, tickers, start, end, **kwargs):
        return {"tickers": tickers, "start": start, "end": end}

    def validate(self):
        return True


class _NoNameDataSource(DataSourcePlugin):
    """Data source without a name — should be rejected."""

    def fetch(self, tickers, start, end, **kwargs):
        return {}

    def validate(self):
        return True


class _FakeBackendCls:
    """Minimal class that satisfies backend interface checks."""

    backend_id = "fake"

    def run(self, circuit, shots=1024):
        return {}


class _BadBackendCls:
    """Class missing required backend attributes."""

    pass


# ------------------------------------------------------------------
# PluginInfo
# ------------------------------------------------------------------


class TestPluginInfo:
    """Tests for the PluginInfo dataclass."""

    def test_default_values(self):
        info = PluginInfo(name="test")
        assert info.name == "test"
        assert info.module == ""
        assert info.plugin_type == ""
        assert info.loaded is False
        assert info.error is None
        assert info.metadata == {}

    def test_full_construction(self):
        info = PluginInfo(
            name="my_backend",
            module="my_pkg.backend",
            plugin_type="backend",
            loaded=True,
            metadata={"version": "1.0"},
        )
        assert info.name == "my_backend"
        assert info.module == "my_pkg.backend"
        assert info.loaded is True
        assert info.metadata["version"] == "1.0"


# ------------------------------------------------------------------
# Backend discovery
# ------------------------------------------------------------------


class TestDiscoverBackends:
    """Tests for entry-point-based backend discovery."""

    def test_discover_returns_dict(self):
        result = discover_backends()
        assert isinstance(result, dict)

    def test_discover_with_mock_entry_points(self):
        mock_ep = MagicMock()
        mock_ep.name = "test_backend"
        mock_ep.value = "test_pkg.backend:TestBackend"

        mock_eps = MagicMock()
        mock_eps.select.return_value = [mock_ep]

        with patch("qufin.plugins.importlib.metadata.entry_points", return_value=mock_eps):
            result = discover_backends()

        assert "test_backend" in result
        assert result["test_backend"].module == "test_pkg.backend:TestBackend"
        assert result["test_backend"].plugin_type == "backend"


class TestLoadBackend:
    """Tests for loading backends from entry points."""

    def test_missing_backend_raises(self):
        mock_eps = MagicMock()
        mock_eps.select.return_value = []

        with (
            patch("qufin.plugins.importlib.metadata.entry_points", return_value=mock_eps),
            pytest.raises(PluginError, match="not found"),
        ):
            load_backend("nonexistent")

    def test_load_valid_backend(self):
        mock_ep = MagicMock()
        mock_ep.name = "fake"
        mock_ep.load.return_value = _FakeBackendCls

        mock_eps = MagicMock()
        mock_eps.select.return_value = [mock_ep]

        with patch("qufin.plugins.importlib.metadata.entry_points", return_value=mock_eps):
            cls = load_backend("fake")
        assert cls is _FakeBackendCls

    def test_load_backend_import_error(self):
        mock_ep = MagicMock()
        mock_ep.name = "broken"
        mock_ep.load.side_effect = ImportError("no module")

        mock_eps = MagicMock()
        mock_eps.select.return_value = [mock_ep]

        with (
            patch("qufin.plugins.importlib.metadata.entry_points", return_value=mock_eps),
            pytest.raises(PluginError, match="Failed to load"),
        ):
            load_backend("broken")


class TestValidateBackend:
    """Tests for backend validation."""

    def test_valid_backend_passes(self):
        # Should not raise
        _validate_backend(_FakeBackendCls, "fake")

    def test_invalid_backend_raises(self):
        with pytest.raises(PluginError, match="missing required attribute"):
            _validate_backend(_BadBackendCls, "bad")


# ------------------------------------------------------------------
# Strategy registry
# ------------------------------------------------------------------


class TestStrategyRegistry:
    """Tests for decorator-based strategy registration."""

    def setup_method(self):
        clear_strategies()

    def teardown_method(self):
        clear_strategies()

    def test_register_and_get(self):
        @register_strategy("test_strat", category="portfolio")
        def my_strat(**kwargs):
            return 42

        func = get_strategy("test_strat")
        assert func() == 42

    def test_get_unregistered_raises(self):
        with pytest.raises(PluginError, match="not registered"):
            get_strategy("nonexistent")

    def test_list_strategies_all(self):
        @register_strategy("s1", category="portfolio")
        def s1():
            pass

        @register_strategy("s2", category="risk")
        def s2():
            pass

        results = list_strategies()
        names = [r["name"] for r in results]
        assert "s1" in names
        assert "s2" in names

    def test_list_strategies_by_category(self):
        @register_strategy("s1", category="portfolio")
        def s1():
            pass

        @register_strategy("s2", category="risk")
        def s2():
            pass

        results = list_strategies(category="portfolio")
        assert len(results) == 1
        assert results[0]["name"] == "s1"

    def test_unregister_strategy(self):
        @register_strategy("temp", category="test")
        def temp():
            pass

        unregister_strategy("temp")
        with pytest.raises(PluginError):
            get_strategy("temp")

    def test_overwrite_warning(self):
        @register_strategy("dup", category="test")
        def v1():
            return 1

        @register_strategy("dup", category="test")
        def v2():
            return 2

        assert get_strategy("dup")() == 2

    def test_strategy_description(self):
        @register_strategy("desc_test", category="portfolio", description="My desc")
        def strat():
            pass

        results = list_strategies()
        entry = next(r for r in results if r["name"] == "desc_test")
        assert entry["description"] == "My desc"

    def test_clear_strategies(self):
        @register_strategy("to_clear", category="test")
        def temp():
            pass

        clear_strategies()
        assert list_strategies() == []


# ------------------------------------------------------------------
# Data source plugins
# ------------------------------------------------------------------


class TestDataSourceRegistry:
    """Tests for data source plugin registration."""

    def setup_method(self):
        clear_data_sources()

    def teardown_method(self):
        clear_data_sources()

    def test_register_and_get(self):
        src = _DummyDataSource()
        register_data_source(src)
        retrieved = get_data_source("dummy")
        assert retrieved is src

    def test_get_unregistered_raises(self):
        with pytest.raises(PluginError, match="not registered"):
            get_data_source("nonexistent")

    def test_register_non_plugin_raises(self):
        with pytest.raises(PluginError, match="DataSourcePlugin instance"):
            register_data_source("not_a_plugin")  # type: ignore[arg-type]

    def test_register_no_name_raises(self):
        src = _NoNameDataSource()
        with pytest.raises(PluginError, match="non-empty"):
            register_data_source(src)

    def test_list_data_sources(self):
        register_data_source(_DummyDataSource())
        names = list_data_sources()
        assert "dummy" in names

    def test_unregister_data_source(self):
        register_data_source(_DummyDataSource())
        unregister_data_source("dummy")
        assert "dummy" not in list_data_sources()

    def test_clear_data_sources(self):
        register_data_source(_DummyDataSource())
        clear_data_sources()
        assert list_data_sources() == []

    def test_fetch_method(self):
        src = _DummyDataSource()
        result = src.fetch(["AAPL"], "2024-01-01", "2024-12-31")
        assert result["tickers"] == ["AAPL"]

    def test_validate_method(self):
        src = _DummyDataSource()
        assert src.validate() is True


# ------------------------------------------------------------------
# Unified discovery
# ------------------------------------------------------------------


class TestDiscoverAll:
    """Tests for the discover_all function."""

    def setup_method(self):
        clear_strategies()
        clear_data_sources()

    def teardown_method(self):
        clear_strategies()
        clear_data_sources()

    def test_returns_all_categories(self):
        result = discover_all()
        assert "backends" in result
        assert "strategies" in result
        assert "data_sources" in result

    def test_includes_registered_strategies(self):
        @register_strategy("disc_test", category="portfolio")
        def strat():
            pass

        result = discover_all()
        strategy_names = [s.name for s in result["strategies"]]
        assert "disc_test" in strategy_names

    def test_includes_registered_data_sources(self):
        register_data_source(_DummyDataSource())
        result = discover_all()
        ds_names = [ds.name for ds in result["data_sources"]]
        assert "dummy" in ds_names


# ------------------------------------------------------------------
# PluginError
# ------------------------------------------------------------------


class TestPluginError:
    """Tests for the PluginError exception."""

    def test_is_exception(self):
        assert issubclass(PluginError, Exception)

    def test_message(self):
        exc = PluginError("test message")
        assert str(exc) == "test message"
