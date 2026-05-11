"""Tests for the walk-forward backtesting engine and performance metrics."""

from __future__ import annotations

import numpy as np
import pytest

from qufin.backtesting.engine import BacktestEngine, BacktestResult
from qufin.backtesting.metrics import (
    PerformanceSummary,
    annualized_return,
    annualized_volatility,
    calmar_ratio,
    cvar_historical,
    hit_rate,
    information_ratio,
    max_drawdown,
    performance_summary,
    sharpe_ratio,
    sortino_ratio,
    tracking_error,
    turnover,
    var_historical,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def daily_returns():
    """Synthetic daily returns: 2 years, 5 assets with strong positive drift."""
    rng = np.random.default_rng(42)
    # Strong positive drift to ensure positive returns regardless of noise
    mu = np.array([0.001, 0.0008, 0.0012, 0.0006, 0.0015])
    returns = rng.normal(loc=mu, scale=0.005, size=(504, 5))
    return returns


@pytest.fixture
def flat_returns():
    """Constant positive daily returns (zero vol)."""
    return np.full(252, 0.0004)


@pytest.fixture
def negative_returns():
    """Constant negative daily returns."""
    return np.full(252, -0.001)


# ---------------------------------------------------------------------------
# Metric unit tests
# ---------------------------------------------------------------------------

class TestAnnualizedReturn:
    def test_positive_drift(self, daily_returns):
        port = daily_returns[:, 0]
        ar = annualized_return(port)
        # With ~0.03% daily drift, annual should be positive
        assert ar > 0

    def test_zero_length(self):
        ar = annualized_return(np.array([]))
        assert ar == 0.0

    def test_single_period(self):
        ar = annualized_return(np.array([0.10]), periods_per_year=1)
        assert abs(ar - 0.10) < 1e-10

    def test_known_value(self):
        # 1% daily for 252 days
        rets = np.full(252, 0.01)
        ar = annualized_return(rets, periods_per_year=252)
        expected = (1.01**252) - 1  # ~1,127%
        assert abs(ar - expected) < 0.01


class TestAnnualizedVolatility:
    def test_positive(self, daily_returns):
        vol = annualized_volatility(daily_returns[:, 0])
        assert vol > 0

    def test_zero_vol(self, flat_returns):
        vol = annualized_volatility(flat_returns)
        assert vol < 1e-10

    def test_scaling(self):
        rng = np.random.default_rng(99)
        rets = rng.normal(0, 0.01, 252)
        daily_std = np.std(rets, ddof=1)
        ann_vol = annualized_volatility(rets)
        assert abs(ann_vol - daily_std * np.sqrt(252)) < 1e-10


class TestSharpeRatio:
    def test_positive_sharpe(self, daily_returns):
        sr = sharpe_ratio(daily_returns[:, 0])
        # Positive drift should give positive Sharpe
        assert sr > 0

    def test_risk_free_reduces_sharpe(self, daily_returns):
        sr_0 = sharpe_ratio(daily_returns[:, 0], risk_free_rate=0.0)
        sr_5 = sharpe_ratio(daily_returns[:, 0], risk_free_rate=0.05)
        assert sr_0 > sr_5

    def test_zero_vol_returns_zero(self, flat_returns):
        sr = sharpe_ratio(flat_returns)
        assert sr == 0.0


class TestSortinoRatio:
    def test_positive(self, daily_returns):
        sr = sortino_ratio(daily_returns[:, 0])
        assert sr > 0

    def test_no_downside(self):
        rets = np.full(100, 0.001)
        sr = sortino_ratio(rets)
        assert sr == 0.0  # no downside returns


class TestMaxDrawdown:
    def test_always_negative_or_zero(self, daily_returns):
        mdd = max_drawdown(daily_returns[:, 0])
        assert mdd <= 0

    def test_known_drawdown(self):
        # Up 10%, then down 20% from peak
        rets = np.array([0.10, -0.20])
        mdd = max_drawdown(rets)
        # Peak at 1.1, trough at 1.1*0.8=0.88, dd = (0.88-1.1)/1.1 = -0.2
        assert abs(mdd - (-0.20)) < 1e-10

    def test_monotone_up(self):
        rets = np.full(100, 0.01)
        mdd = max_drawdown(rets)
        assert mdd == 0.0


class TestCalmarRatio:
    def test_positive_with_drawdown(self, daily_returns):
        cr = calmar_ratio(daily_returns[:, 0])
        # Positive return and negative drawdown → positive Calmar
        assert cr > 0

    def test_no_drawdown(self):
        rets = np.full(252, 0.001)
        cr = calmar_ratio(rets)
        assert cr == 0.0  # |mdd| < 1e-12


class TestTurnover:
    def test_no_turnover(self):
        w = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
        assert turnover(w) == 0.0

    def test_full_turnover(self):
        w = np.array([[1.0, 0.0], [0.0, 1.0]])
        to = turnover(w)
        # |diff| = [1.0, 1.0], sum=2.0, /2 = 1.0
        assert abs(to - 1.0) < 1e-10

    def test_single_row(self):
        w = np.array([[0.5, 0.5]])
        assert turnover(w) == 0.0


class TestTrackingError:
    def test_identical(self):
        rets = np.random.default_rng(1).normal(0, 0.01, 252)
        te = tracking_error(rets, rets)
        assert te < 1e-10

    def test_positive(self):
        rng = np.random.default_rng(2)
        a = rng.normal(0, 0.01, 252)
        b = rng.normal(0, 0.01, 252)
        te = tracking_error(a, b)
        assert te > 0


class TestInformationRatio:
    def test_outperformance(self):
        rng = np.random.default_rng(3)
        bench = rng.normal(0, 0.01, 252)
        port = bench + 0.001  # constant outperformance
        information_ratio(port, bench)
        # Constant spread → tracking error ≈ 0, so IR may be 0 due to guard
        # Use non-constant outperformance instead
        port2 = bench + rng.uniform(0.0005, 0.002, 252)
        ir2 = information_ratio(port2, bench)
        assert ir2 > 0


class TestHitRate:
    def test_all_positive(self):
        assert hit_rate(np.full(100, 0.01)) == 1.0

    def test_all_negative(self):
        assert hit_rate(np.full(100, -0.01)) == 0.0

    def test_empty(self):
        assert hit_rate(np.array([])) == 0.0


class TestVaR:
    def test_positive(self, daily_returns):
        v = var_historical(daily_returns[:, 0])
        assert v > 0  # VaR is a positive loss number

    def test_cvar_ge_var(self, daily_returns):
        port = daily_returns[:, 0]
        v = var_historical(port)
        cv = cvar_historical(port)
        assert cv >= v - 1e-10


class TestPerformanceSummary:
    def test_all_fields_populated(self, daily_returns):
        ps = performance_summary(daily_returns[:, 0])
        assert isinstance(ps, PerformanceSummary)
        assert ps.n_periods == len(daily_returns[:, 0])
        assert ps.periods_per_year == 252

    def test_with_weights(self, daily_returns):
        n = len(daily_returns)
        w = np.full((n, 5), 0.2)
        port = daily_returns @ np.full(5, 0.2)
        ps = performance_summary(port, weights_history=w)
        assert ps.avg_turnover == 0.0  # constant weights


# ---------------------------------------------------------------------------
# Engine tests
# ---------------------------------------------------------------------------

def equal_weight_strategy(mu, cov):
    """Simple equal-weight strategy."""
    n = len(mu)
    return np.ones(n) / n


def min_var_strategy(mu, cov):
    """Minimum variance (analytical for unconstrained)."""
    try:
        inv_cov = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        return np.ones(len(mu)) / len(mu)
    ones = np.ones(len(mu))
    w = inv_cov @ ones
    return w / np.sum(w)


def max_return_strategy(mu, cov):
    """Go all-in on the highest expected return asset."""
    w = np.zeros(len(mu))
    w[np.argmax(mu)] = 1.0
    return w


class TestBacktestEngine:
    def test_basic_run(self, daily_returns):
        engine = BacktestEngine(daily_returns, train_window=126, test_window=21)
        result = engine.run(equal_weight_strategy, strategy_name="equal_weight")

        assert isinstance(result, BacktestResult)
        assert result.strategy_name == "equal_weight"
        assert len(result.portfolio_returns) > 0
        assert result.summary is not None
        assert result.metadata["train_window"] == 126
        assert result.metadata["test_window"] == 21

    def test_returns_length(self, daily_returns):
        engine = BacktestEngine(daily_returns, train_window=126, test_window=21)
        result = engine.run(equal_weight_strategy)
        # Should have returns for all periods after the first train window
        expected_len = len(daily_returns) - 126
        assert len(result.portfolio_returns) == expected_len

    def test_weights_normalized(self, daily_returns):
        engine = BacktestEngine(daily_returns, train_window=126, test_window=21)
        result = engine.run(equal_weight_strategy)
        for w in result.weights_history:
            assert abs(np.sum(np.abs(w)) - 1.0) < 1e-10

    def test_rebalance_count(self, daily_returns):
        engine = BacktestEngine(daily_returns, train_window=126, test_window=21)
        result = engine.run(equal_weight_strategy)
        expected_rebalances = (len(daily_returns) - 126 + 20) // 21
        assert len(result.rebalance_dates) == expected_rebalances

    def test_transaction_costs_reduce_returns(self, daily_returns):
        engine_0 = BacktestEngine(
            daily_returns, train_window=126, test_window=21, transaction_cost=0.0)
        engine_tc = BacktestEngine(
            daily_returns, train_window=126, test_window=21, transaction_cost=0.01)

        r0 = engine_0.run(equal_weight_strategy)
        rtc = engine_tc.run(equal_weight_strategy)

        # With transaction costs, cumulative return should be lower
        cum0 = np.prod(1 + r0.portfolio_returns)
        cumtc = np.prod(1 + rtc.portfolio_returns)
        assert cumtc < cum0

    def test_compare(self, daily_returns):
        engine = BacktestEngine(daily_returns, train_window=126, test_window=21)
        strategies = {
            "equal_weight": equal_weight_strategy,
            "min_var": min_var_strategy,
            "max_return": max_return_strategy,
        }
        results = engine.compare(strategies)
        assert len(results) == 3
        assert all(isinstance(r, BacktestResult) for r in results.values())

    def test_comparison_table(self, daily_returns):
        engine = BacktestEngine(daily_returns, train_window=126, test_window=21)
        strategies = {
            "equal_weight": equal_weight_strategy,
            "min_var": min_var_strategy,
        }
        results = engine.compare(strategies)
        table = engine.comparison_table(results)
        assert len(table) == 2
        assert all("Strategy" in row for row in table)
        assert all("Sharpe" in row for row in table)

    def test_failing_strategy_fallback(self, daily_returns):
        def bad_strategy(mu, cov):
            raise ValueError("intentional failure")

        engine = BacktestEngine(daily_returns, train_window=126, test_window=21)
        result = engine.run(bad_strategy, strategy_name="bad")
        # Should not raise, should fall back to equal weight
        assert len(result.portfolio_returns) > 0

    def test_custom_dates(self, daily_returns):
        import pandas as pd
        dates = pd.date_range("2020-01-01", periods=len(daily_returns), freq="B")
        engine = BacktestEngine(daily_returns, dates=dates, train_window=126, test_window=21)
        result = engine.run(equal_weight_strategy)
        assert isinstance(result.rebalance_dates[0], pd.Timestamp)

    def test_small_dataset(self):
        rng = np.random.default_rng(7)
        rets = rng.normal(0, 0.01, (60, 3))
        engine = BacktestEngine(rets, train_window=30, test_window=10)
        result = engine.run(equal_weight_strategy)
        assert len(result.portfolio_returns) == 30

    def test_single_asset(self):
        rng = np.random.default_rng(8)
        rets = rng.normal(0.0003, 0.01, (252, 1))
        engine = BacktestEngine(rets, train_window=126, test_window=21)
        result = engine.run(equal_weight_strategy)
        assert result.weights_history.shape[1] == 1
        assert np.allclose(result.weights_history, 1.0)
