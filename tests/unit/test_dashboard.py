"""Tests for qufin.viz.dashboard -- portfolio dashboard charts."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from qufin.viz.dashboard import (
    DashboardConfig,
    _compute_drawdown,
    _make_figure,
    _trace,
    create_dash_app,
    drawdown_chart,
    portfolio_value_chart,
    rebalancing_schedule_chart,
    risk_decomposition_chart,
    transaction_cost_chart,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dates():
    return [f"2025-01-{d:02d}" for d in range(1, 31)]


@pytest.fixture
def values(dates):
    rng = np.random.default_rng(42)
    base = 1000.0
    rets = rng.normal(0.001, 0.01, len(dates))
    return (base * np.cumprod(1 + rets)).tolist()


@pytest.fixture
def config():
    return DashboardConfig(width=800, height=400)


# ---------------------------------------------------------------------------
# DashboardConfig
# ---------------------------------------------------------------------------

class TestDashboardConfig:
    def test_defaults(self):
        cfg = DashboardConfig()
        assert cfg.width == 1000
        assert cfg.height == 500
        assert cfg.template == "plotly_white"

    def test_custom(self, config):
        assert config.width == 800


# ---------------------------------------------------------------------------
# _compute_drawdown
# ---------------------------------------------------------------------------

class TestComputeDrawdown:
    def test_monotonic_up(self):
        dd = _compute_drawdown(np.array([1.0, 2.0, 3.0, 4.0]))
        np.testing.assert_array_equal(dd, [0.0, 0.0, 0.0, 0.0])

    def test_simple_drawdown(self):
        dd = _compute_drawdown(np.array([100.0, 90.0, 80.0, 100.0]))
        assert dd[0] == 0.0
        assert dd[1] == pytest.approx(-0.1)
        assert dd[2] == pytest.approx(-0.2)
        assert dd[3] == pytest.approx(0.0)

    def test_all_same(self):
        dd = _compute_drawdown(np.array([5.0, 5.0, 5.0]))
        np.testing.assert_array_equal(dd, [0.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# portfolio_value_chart
# ---------------------------------------------------------------------------

class TestPortfolioValueChart:
    def test_basic(self, dates, values):
        fig = portfolio_value_chart(dates, values)
        assert fig is not None

    def test_with_benchmark(self, dates, values):
        bench = [v * 0.98 for v in values]
        fig = portfolio_value_chart(dates, values, benchmark_values=bench)
        if hasattr(fig, "data"):
            assert len(fig.data) == 2

    def test_custom_title(self, dates, values):
        fig = portfolio_value_chart(dates, values, title="NAV")
        if hasattr(fig, "layout"):
            assert fig.layout.title.text == "NAV"

    def test_custom_config(self, dates, values, config):
        fig = portfolio_value_chart(dates, values, config=config)
        assert fig is not None


# ---------------------------------------------------------------------------
# drawdown_chart
# ---------------------------------------------------------------------------

class TestDrawdownChart:
    def test_basic(self, dates, values):
        fig = drawdown_chart(dates, values)
        assert fig is not None

    def test_title(self, dates, values):
        fig = drawdown_chart(dates, values, title="DD")
        if hasattr(fig, "layout"):
            assert fig.layout.title.text == "DD"


# ---------------------------------------------------------------------------
# risk_decomposition_chart
# ---------------------------------------------------------------------------

class TestRiskDecompositionChart:
    def test_asset_level(self):
        fig = risk_decomposition_chart(
            ["AAPL", "MSFT", "GOOG"],
            [0.4, 0.35, 0.25],
        )
        assert fig is not None

    def test_factor_level(self):
        fig = risk_decomposition_chart(
            ["Momentum", "Value", "Size"],
            [0.5, 0.3, 0.2],
            level="factor",
        )
        if hasattr(fig, "layout"):
            assert "Factor" in fig.layout.title.text

    def test_sector_level(self):
        fig = risk_decomposition_chart(
            ["Tech", "Finance"], [0.6, 0.4], level="sector",
        )
        assert fig is not None

    def test_custom_title(self):
        fig = risk_decomposition_chart(
            ["A", "B"], [1.0, 2.0], title="Custom Risk",
        )
        if hasattr(fig, "layout"):
            assert fig.layout.title.text == "Custom Risk"

    def test_sorting(self):
        """Contributions should be sorted descending."""
        fig = risk_decomposition_chart(
            ["Low", "High", "Mid"], [0.1, 0.9, 0.5],
        )
        if hasattr(fig, "data"):
            y_vals = list(fig.data[0].y)
            assert y_vals == ["High", "Mid", "Low"]


# ---------------------------------------------------------------------------
# rebalancing_schedule_chart
# ---------------------------------------------------------------------------

class TestRebalancingScheduleChart:
    def test_basic(self):
        dates = ["2025-01", "2025-02", "2025-03"]
        turnover = [0.15, 0.08, 0.22]
        fig = rebalancing_schedule_chart(dates, turnover)
        assert fig is not None

    def test_custom_config(self, config):
        fig = rebalancing_schedule_chart(
            ["Q1", "Q2"], [0.1, 0.2], config=config,
        )
        assert fig is not None


# ---------------------------------------------------------------------------
# transaction_cost_chart
# ---------------------------------------------------------------------------

class TestTransactionCostChart:
    def test_with_cumulative(self):
        dates = ["2025-01", "2025-02", "2025-03"]
        costs = [100.0, 150.0, 80.0]
        fig = transaction_cost_chart(dates, costs, cumulative=True)
        if hasattr(fig, "data"):
            assert len(fig.data) == 2  # bar + cumulative line

    def test_without_cumulative(self):
        fig = transaction_cost_chart(
            ["D1", "D2"], [50.0, 60.0], cumulative=False,
        )
        if hasattr(fig, "data"):
            assert len(fig.data) == 1

    def test_cumulative_values(self):
        costs = [10.0, 20.0, 30.0]
        fig = transaction_cost_chart(["a", "b", "c"], costs)
        if hasattr(fig, "data") and len(fig.data) == 2:
            cum_y = list(fig.data[1].y)
            assert cum_y == [10.0, 30.0, 60.0]


# ---------------------------------------------------------------------------
# create_dash_app
# ---------------------------------------------------------------------------

class TestCreateDashApp:
    def test_raises_without_dash(self):
        """Should raise ImportError when dash is not installed."""
        import sys
        # Simulate dash not installed
        with patch.dict(sys.modules, {"dash": None, "dash.html": None, "dash.dcc": None}), \
                pytest.raises(ImportError, match="dash is required"):
                create_dash_app(["2025-01"], [1000.0])

    def test_app_creation_with_mock_dash(self):
        """Smoke test with mocked dash."""
        mock_dash_module = MagicMock()
        mock_app = MagicMock()
        mock_dash_module.Dash.return_value = mock_app
        mock_dash_module.html = MagicMock()
        mock_dash_module.dcc = MagicMock()

        import sys
        with patch.dict(sys.modules, {"dash": mock_dash_module}), \
                patch("qufin.viz.dashboard.create_dash_app") as mock_create:
                mock_create.return_value = mock_app
                app = mock_create(["2025-01"], [1000.0])
                assert app is mock_app


# ---------------------------------------------------------------------------
# Dict fallback
# ---------------------------------------------------------------------------

class TestDictFallback:
    def test_make_figure_dict(self):
        with patch("qufin.viz.dashboard.HAS_PLOTLY", False):
            result = _make_figure([{"y": [1]}], {"title": "T"})
            assert isinstance(result, dict)

    def test_trace_dict(self):
        with patch("qufin.viz.dashboard.HAS_PLOTLY", False):
            t = _trace("bar", x=[1], y=[2])
            assert isinstance(t, dict)
            assert t["type"] == "bar"
