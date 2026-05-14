"""Unit tests for multi-period portfolio optimization."""

from __future__ import annotations

import numpy as np
import pytest

from qufin.portfolio.optimizers.multi_period import (
    MultiPeriodConfig,
    MultiPeriodResult,
    compute_turnover,
    multi_period_backtest,
    multi_period_optimize,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def simple_data():
    """3-asset, 4-period data."""
    rng = np.random.default_rng(42)
    n_assets = 3
    n_periods = 4
    mu_series = [rng.standard_normal(n_assets) * 0.01 for _ in range(n_periods)]
    cov_series = []
    for _ in range(n_periods):
        A = rng.standard_normal((n_assets, n_assets)) * 0.01
        cov_series.append(A.T @ A + np.eye(n_assets) * 0.001)
    return mu_series, cov_series


# ---------------------------------------------------------------------------
# compute_turnover
# ---------------------------------------------------------------------------

class TestComputeTurnover:
    def test_identical_weights(self) -> None:
        w = np.array([0.5, 0.3, 0.2])
        assert compute_turnover(w, w) == pytest.approx(0.0)

    def test_full_turnover(self) -> None:
        w_old = np.array([1.0, 0.0, 0.0])
        w_new = np.array([0.0, 0.0, 1.0])
        assert compute_turnover(w_old, w_new) == pytest.approx(1.0)

    def test_partial_turnover(self) -> None:
        w_old = np.array([0.5, 0.5, 0.0])
        w_new = np.array([0.5, 0.0, 0.5])
        assert compute_turnover(w_old, w_new) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# multi_period_optimize (classical)
# ---------------------------------------------------------------------------

class TestMultiPeriodOptimize:
    def test_default_config(self, simple_data) -> None:
        mu_series, cov_series = simple_data
        result = multi_period_optimize(mu_series, cov_series)
        assert isinstance(result, MultiPeriodResult)
        assert len(result.allocations) == 4
        assert len(result.objectives) == 4
        assert len(result.turnovers) == 3  # T-1

    def test_weights_sum_to_one(self, simple_data) -> None:
        mu_series, cov_series = simple_data
        result = multi_period_optimize(mu_series, cov_series)
        for w in result.allocations:
            assert np.sum(w) == pytest.approx(1.0, abs=1e-4)

    def test_weights_non_negative(self, simple_data) -> None:
        mu_series, cov_series = simple_data
        result = multi_period_optimize(mu_series, cov_series)
        for w in result.allocations:
            assert np.all(w >= -1e-6)

    def test_total_objective_finite(self, simple_data) -> None:
        mu_series, cov_series = simple_data
        result = multi_period_optimize(mu_series, cov_series)
        assert np.isfinite(result.total_objective)

    def test_total_turnover_non_negative(self, simple_data) -> None:
        mu_series, cov_series = simple_data
        result = multi_period_optimize(mu_series, cov_series)
        assert result.total_turnover >= 0.0

    def test_turnover_penalty_reduces_turnover(self, simple_data) -> None:
        mu_series, cov_series = simple_data
        config_low = MultiPeriodConfig(turnover_penalty=0.001)
        config_high = MultiPeriodConfig(turnover_penalty=1.0)
        r_low = multi_period_optimize(mu_series, cov_series, config=config_low)
        r_high = multi_period_optimize(mu_series, cov_series, config=config_high)
        assert r_high.total_turnover <= r_low.total_turnover + 1e-6

    def test_holding_cost(self, simple_data) -> None:
        mu_series, cov_series = simple_data
        config = MultiPeriodConfig(holding_cost=0.01)
        result = multi_period_optimize(mu_series, cov_series, config=config)
        assert len(result.allocations) == 4

    def test_classical_method(self, simple_data) -> None:
        mu_series, cov_series = simple_data
        config = MultiPeriodConfig(method="classical")
        result = multi_period_optimize(mu_series, cov_series, config=config)
        assert len(result.allocations) == 4

    def test_empty_input(self) -> None:
        result = multi_period_optimize([], [])
        assert len(result.allocations) == 0

    def test_mismatched_lengths(self) -> None:
        mu = [np.array([0.01, 0.02])]
        cov = [np.eye(2), np.eye(2)]
        with pytest.raises(ValueError, match="length"):
            multi_period_optimize(mu, cov)

    def test_single_period(self) -> None:
        mu = [np.array([0.01, 0.02, 0.03])]
        cov = [np.eye(3) * 0.01]
        result = multi_period_optimize(mu, cov)
        assert len(result.allocations) == 1
        assert len(result.turnovers) == 0

    def test_with_cardinality(self, simple_data) -> None:
        mu_series, cov_series = simple_data
        config = MultiPeriodConfig(cardinality=2)
        result = multi_period_optimize(mu_series, cov_series, config=config)
        for w in result.allocations:
            assert np.sum(w > 1e-6) <= 2


# ---------------------------------------------------------------------------
# multi_period_backtest
# ---------------------------------------------------------------------------

class TestMultiPeriodBacktest:
    def test_basic_backtest(self) -> None:
        rng = np.random.default_rng(42)
        prices = 100 + np.cumsum(rng.standard_normal((100, 3)) * 0.5, axis=0)
        prices = np.maximum(prices, 1.0)  # no negative prices
        allocs = [
            np.array([0.5, 0.3, 0.2]),
            np.array([0.3, 0.4, 0.3]),
        ]
        rebalance_dates = [0, 50]
        result = multi_period_backtest(prices, allocs, rebalance_dates)
        assert "portfolio_values" in result
        assert "returns" in result
        assert "sharpe" in result
        assert "max_drawdown" in result
        assert np.isfinite(result["sharpe"])
        assert 0 <= result["max_drawdown"] <= 1.0

    def test_portfolio_values_positive(self) -> None:
        prices = np.array([[100, 100], [101, 99], [102, 98], [103, 97]])
        allocs = [np.array([0.5, 0.5])]
        rebalance_dates = [0]
        result = multi_period_backtest(prices, allocs, rebalance_dates)
        assert np.all(result["portfolio_values"] > 0)

    def test_mismatched_lengths_raises(self) -> None:
        prices = np.ones((10, 2))
        with pytest.raises(ValueError, match="length"):
            multi_period_backtest(prices, [np.array([0.5, 0.5])], [0, 5])

    def test_no_rebalance(self) -> None:
        prices = np.array([[100, 100], [101, 101], [102, 102]])
        allocs = [np.array([0.5, 0.5])]
        rebalance_dates = [1]
        result = multi_period_backtest(prices, allocs, rebalance_dates)
        assert len(result["returns"]) == 2
