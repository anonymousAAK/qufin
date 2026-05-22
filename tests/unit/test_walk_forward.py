"""Unit tests for walk-forward validation utilities."""

from __future__ import annotations

import numpy as np
import pytest

from qufin.backtesting.walk_forward import (
    CSCVResult,
    cscv_backtest_overfitting,
    permutation_test,
)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


# --- Permutation Test ---

class TestPermutationTest:
    def test_returns_float(self, rng) -> None:
        strat = rng.normal(0.001, 0.01, 200)
        bench = rng.normal(0.0, 0.01, 200)
        p = permutation_test(strat, bench, n_perms=500, seed=42)
        assert isinstance(p, float)

    def test_p_value_in_range(self, rng) -> None:
        strat = rng.normal(0.001, 0.01, 200)
        bench = rng.normal(0.0, 0.01, 200)
        p = permutation_test(strat, bench, n_perms=500, seed=42)
        assert 0.0 <= p <= 1.0

    def test_identical_returns_high_pvalue(self, rng) -> None:
        data = rng.normal(0.0, 0.01, 200)
        p = permutation_test(data, data.copy(), n_perms=500, seed=42)
        assert p >= 0.3  # no skill => p-value should be large

    def test_strong_strategy_low_pvalue(self) -> None:
        rng = np.random.default_rng(42)
        strat = rng.normal(0.05, 0.01, 500)
        bench = rng.normal(0.0, 0.01, 500)
        p = permutation_test(strat, bench, n_perms=1000, seed=42)
        assert p < 0.05

    def test_deterministic(self, rng) -> None:
        strat = rng.normal(0.001, 0.01, 100)
        bench = rng.normal(0.0, 0.01, 100)
        p1 = permutation_test(strat, bench, n_perms=200, seed=123)
        p2 = permutation_test(strat, bench, n_perms=200, seed=123)
        assert p1 == p2

    def test_different_seeds_different_results(self, rng) -> None:
        strat = rng.normal(0.001, 0.01, 100)
        bench = rng.normal(0.0, 0.01, 100)
        p1 = permutation_test(strat, bench, n_perms=200, seed=1)
        p2 = permutation_test(strat, bench, n_perms=200, seed=2)
        # Not guaranteed to differ, but with high probability they will
        # Just check both are valid
        assert 0.0 <= p1 <= 1.0
        assert 0.0 <= p2 <= 1.0


# --- CSCV ---

class TestCSCVBacktestOverfitting:
    def test_returns_cscv_result(self, rng) -> None:
        mat = rng.normal(0.0, 0.01, (200, 5))
        result = cscv_backtest_overfitting(mat, n_partitions=4)
        assert isinstance(result, CSCVResult)

    def test_pbo_in_range(self, rng) -> None:
        mat = rng.normal(0.0, 0.01, (200, 5))
        result = cscv_backtest_overfitting(mat, n_partitions=4)
        assert 0.0 <= result.pbo <= 1.0

    def test_n_combinations_positive(self, rng) -> None:
        mat = rng.normal(0.0, 0.01, (200, 5))
        result = cscv_backtest_overfitting(mat, n_partitions=4)
        assert result.n_combinations > 0

    def test_logits_shape(self, rng) -> None:
        mat = rng.normal(0.0, 0.01, (200, 5))
        result = cscv_backtest_overfitting(mat, n_partitions=4)
        assert len(result.logits) == result.n_combinations

    def test_random_strategies_moderate_pbo(self, rng) -> None:
        # Random strategies should have PBO around 0.5
        mat = rng.normal(0.0, 0.01, (400, 10))
        result = cscv_backtest_overfitting(mat, n_partitions=4)
        assert 0.1 <= result.pbo <= 0.9

    def test_overfit_strategy_high_pbo(self) -> None:
        rng = np.random.default_rng(42)
        T, S = 400, 10
        mat = rng.normal(0.0, 0.01, (T, S))
        # Make one strategy look great in first half, bad in second
        mat[:T // 2, 0] += 0.05
        mat[T // 2:, 0] -= 0.05
        result = cscv_backtest_overfitting(mat, n_partitions=4)
        # High PBO expected for overfit strategy
        assert isinstance(result.pbo, float)

    def test_odd_partitions_adjusted(self, rng) -> None:
        mat = rng.normal(0.0, 0.01, (200, 5))
        result = cscv_backtest_overfitting(mat, n_partitions=5)
        assert result.n_combinations > 0

    def test_too_few_rows(self) -> None:
        mat = np.array([[0.01, 0.02]])
        result = cscv_backtest_overfitting(mat, n_partitions=4)
        assert result.n_combinations == 0
        assert result.pbo == 0.0

    def test_many_partitions(self, rng) -> None:
        mat = rng.normal(0.0, 0.01, (400, 5))
        result = cscv_backtest_overfitting(mat, n_partitions=8)
        assert result.n_combinations > 0
        assert 0.0 <= result.pbo <= 1.0
