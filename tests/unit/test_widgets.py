"""Tests for qufin.viz.widgets -- Jupyter widget helpers."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from qufin.viz.widgets import (
    ChartConfig,
    _make_figure,
    _trace_dict,
    asset_selector,
    efficient_frontier,
    parameter_sensitivity,
    quantum_vs_classical,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def frontier_data():
    rng = np.random.default_rng(99)
    n = 50
    vols = np.linspace(0.05, 0.35, n)
    rets = 0.02 + 0.3 * vols - 0.2 * vols**2 + rng.normal(0, 0.002, n)
    sharpe = rets / vols
    return vols, rets, sharpe


@pytest.fixture
def config():
    return ChartConfig(width=800, height=400, template="plotly_dark")


# ---------------------------------------------------------------------------
# ChartConfig
# ---------------------------------------------------------------------------

class TestChartConfig:
    def test_defaults(self):
        cfg = ChartConfig()
        assert cfg.width == 900
        assert cfg.height == 600
        assert cfg.template == "plotly_white"
        assert cfg.title_font_size == 16
        assert len(cfg.colorway) == 8

    def test_custom(self, config):
        assert config.width == 800
        assert config.template == "plotly_dark"


# ---------------------------------------------------------------------------
# asset_selector
# ---------------------------------------------------------------------------

class TestAssetSelector:
    def test_all_selected_by_default(self):
        result = asset_selector(["AAPL", "MSFT", "GOOG"])
        assert result["available"] == ["AAPL", "MSFT", "GOOG"]
        assert result["selected"] == ["AAPL", "MSFT", "GOOG"]

    def test_subset_selection(self):
        result = asset_selector(["AAPL", "MSFT", "GOOG"], selected=["AAPL", "GOOG"])
        assert result["selected"] == ["AAPL", "GOOG"]

    def test_invalid_ticker_filtered(self):
        result = asset_selector(["AAPL", "MSFT"], selected=["AAPL", "TSLA"])
        assert result["selected"] == ["AAPL"]

    def test_empty_universe(self):
        result = asset_selector([])
        assert result["available"] == []
        assert result["selected"] == []

    def test_preserves_order(self):
        result = asset_selector(["C", "B", "A"], selected=["A", "C"])
        # selected order follows the selected input but filtered by available
        assert result["selected"] == ["A", "C"]


# ---------------------------------------------------------------------------
# efficient_frontier
# ---------------------------------------------------------------------------

class TestEfficientFrontier:
    def test_basic_figure(self, frontier_data):
        vols, rets, _ = frontier_data
        fig = efficient_frontier(rets, vols)
        # Should be a plotly Figure or dict
        assert fig is not None

    def test_with_sharpe(self, frontier_data):
        vols, rets, sharpe = frontier_data
        fig = efficient_frontier(rets, vols, sharpe_ratios=sharpe)
        if hasattr(fig, "data"):
            assert len(fig.data) >= 1

    def test_optimal_marker(self, frontier_data):
        vols, rets, sharpe = frontier_data
        opt = int(np.argmax(sharpe))
        fig = efficient_frontier(rets, vols, optimal_idx=opt)
        if hasattr(fig, "data"):
            # Should have frontier + optimal star
            assert len(fig.data) == 2

    def test_optimal_out_of_range(self, frontier_data):
        vols, rets, _ = frontier_data
        fig = efficient_frontier(rets, vols, optimal_idx=9999)
        if hasattr(fig, "data"):
            assert len(fig.data) == 1  # no star trace

    def test_labels(self, frontier_data):
        vols, rets, _ = frontier_data
        labels = [f"P{i}" for i in range(len(rets))]
        fig = efficient_frontier(rets, vols, labels=labels)
        assert fig is not None

    def test_custom_config(self, frontier_data, config):
        vols, rets, _ = frontier_data
        fig = efficient_frontier(rets, vols, config=config)
        if hasattr(fig, "layout"):
            assert fig.layout.template.layout.to_plotly_json() is not None


# ---------------------------------------------------------------------------
# quantum_vs_classical
# ---------------------------------------------------------------------------

class TestQuantumVsClassical:
    def test_basic_bars(self):
        fig = quantum_vs_classical(
            ["Return", "Risk", "Sharpe"],
            [0.12, 0.15, 0.8],
            [0.11, 0.14, 0.79],
        )
        assert fig is not None

    def test_trace_count(self):
        fig = quantum_vs_classical(
            ["A", "B"], [1.0, 2.0], [1.5, 2.5],
        )
        if hasattr(fig, "data"):
            assert len(fig.data) == 2  # classical + quantum

    def test_custom_title(self):
        fig = quantum_vs_classical(
            ["X"], [1.0], [2.0], title="My Comparison",
        )
        if hasattr(fig, "layout"):
            assert fig.layout.title.text == "My Comparison"

    def test_custom_config(self, config):
        fig = quantum_vs_classical(
            ["A"], [1.0], [2.0], config=config,
        )
        assert fig is not None


# ---------------------------------------------------------------------------
# parameter_sensitivity
# ---------------------------------------------------------------------------

class TestParameterSensitivity:
    def test_basic_line(self):
        fig = parameter_sensitivity(
            "risk_aversion", [0.1, 0.5, 1.0, 2.0], [0.9, 0.7, 0.5, 0.3],
        )
        assert fig is not None

    def test_default_title(self):
        fig = parameter_sensitivity("gamma", [1, 2, 3], [4, 5, 6])
        if hasattr(fig, "layout"):
            assert "gamma" in fig.layout.title.text

    def test_custom_title(self):
        fig = parameter_sensitivity(
            "k", [1, 2], [3, 4], title="Custom",
        )
        if hasattr(fig, "layout"):
            assert fig.layout.title.text == "Custom"

    def test_extra_series(self):
        fig = parameter_sensitivity(
            "alpha", [0.1, 0.2, 0.3], [1.0, 2.0, 3.0],
            extra_series={"Upper": [1.1, 2.1, 3.1], "Lower": [0.9, 1.9, 2.9]},
        )
        if hasattr(fig, "data"):
            assert len(fig.data) == 3  # primary + 2 extras

    def test_series_name(self):
        fig = parameter_sensitivity(
            "p", [1], [2], series_name="Cost",
        )
        if hasattr(fig, "data"):
            assert fig.data[0].name == "Cost"


# ---------------------------------------------------------------------------
# Fallback (no plotly) -- test dict mode
# ---------------------------------------------------------------------------

class TestDictFallback:
    def test_make_figure_returns_dict_without_plotly(self):
        with patch("qufin.viz.widgets.HAS_PLOTLY", False):
            result = _make_figure([{"x": [1]}], {"title": "T"})
            assert isinstance(result, dict)
            assert result["layout"]["title"] == "T"

    def test_trace_dict_returns_dict_without_plotly(self):
        with patch("qufin.viz.widgets.HAS_PLOTLY", False):
            t = _trace_dict(x=[1, 2], y=[3, 4], trace_type="scatter")
            assert isinstance(t, dict)
            assert t["x"] == [1, 2]
            assert "trace_type" not in t  # stripped
