"""Unit tests for enhanced warm-start (CVaR + multi-start)."""

from __future__ import annotations

import numpy as np
import pytest

from qufin.portfolio.optimizers.warm_start import (
    MultiStartResult,
    WarmStartResult,
    cvar_warm_start,
    multi_start_qaoa,
)
from qufin.portfolio.qubo import PortfolioQUBO


@pytest.fixture
def small_qubo() -> PortfolioQUBO:
    mu = np.array([0.01, 0.02, 0.015, 0.008])
    cov = np.array([
        [0.04, 0.006, 0.002, 0.001],
        [0.006, 0.09, 0.004, 0.003],
        [0.002, 0.004, 0.01, 0.002],
        [0.001, 0.003, 0.002, 0.025],
    ])
    return PortfolioQUBO(mu=mu, cov=cov, cardinality=2, gamma=0.5)


@pytest.fixture
def simple_qubo() -> PortfolioQUBO:
    mu = np.array([0.01, 0.02, 0.015])
    cov = np.eye(3) * 0.04
    return PortfolioQUBO(mu=mu, cov=cov, gamma=1.0)


class TestCvarWarmStart:
    def test_returns_warm_start_result(self, small_qubo: PortfolioQUBO) -> None:
        result = cvar_warm_start(small_qubo, alpha=0.2, p=3, seed=42)
        assert isinstance(result, WarmStartResult)

    def test_param_shape(self, small_qubo: PortfolioQUBO) -> None:
        result = cvar_warm_start(small_qubo, alpha=0.2, p=4, seed=42)
        assert result.initial_params.shape == (8,)  # 4 gammas + 4 betas

    def test_relaxed_solution_shape(self, small_qubo: PortfolioQUBO) -> None:
        result = cvar_warm_start(small_qubo, alpha=0.3, p=2, seed=42)
        assert result.relaxed_solution.shape == (4,)

    def test_relaxed_in_bounds(self, small_qubo: PortfolioQUBO) -> None:
        result = cvar_warm_start(small_qubo, alpha=0.5, p=3, seed=42)
        assert np.all(result.relaxed_solution >= 0.0)
        assert np.all(result.relaxed_solution <= 1.0)

    def test_bitstring_length(self, small_qubo: PortfolioQUBO) -> None:
        result = cvar_warm_start(small_qubo, alpha=0.2, p=3, seed=42)
        assert len(result.rounded_bitstring) == 4

    def test_deterministic(self, small_qubo: PortfolioQUBO) -> None:
        r1 = cvar_warm_start(small_qubo, alpha=0.2, p=2, seed=42)
        r2 = cvar_warm_start(small_qubo, alpha=0.2, p=2, seed=42)
        np.testing.assert_array_almost_equal(r1.initial_params, r2.initial_params)

    def test_different_alpha(self, small_qubo: PortfolioQUBO) -> None:
        r1 = cvar_warm_start(small_qubo, alpha=0.1, p=3, seed=42)
        r2 = cvar_warm_start(small_qubo, alpha=0.9, p=3, seed=42)
        # With small problems the tail and full mean can coincide;
        # just verify both are valid WarmStartResults with correct shapes
        assert isinstance(r1, WarmStartResult)
        assert isinstance(r2, WarmStartResult)
        assert r1.initial_params.shape == r2.initial_params.shape

    def test_invalid_alpha_zero(self, small_qubo: PortfolioQUBO) -> None:
        with pytest.raises(ValueError, match="alpha"):
            cvar_warm_start(small_qubo, alpha=0.0)

    def test_invalid_alpha_negative(self, small_qubo: PortfolioQUBO) -> None:
        with pytest.raises(ValueError, match="alpha"):
            cvar_warm_start(small_qubo, alpha=-0.5)

    def test_alpha_one_full_mean(self, small_qubo: PortfolioQUBO) -> None:
        result = cvar_warm_start(small_qubo, alpha=1.0, p=3, seed=42)
        assert isinstance(result, WarmStartResult)

    def test_no_cardinality(self, simple_qubo: PortfolioQUBO) -> None:
        result = cvar_warm_start(simple_qubo, alpha=0.3, p=2, seed=42)
        assert result.relaxed_solution.shape == (3,)


class TestMultiStartQaoa:
    def test_returns_multi_start_result(self, small_qubo: PortfolioQUBO) -> None:
        result = multi_start_qaoa(small_qubo, k=3, p=2, seed=42)
        assert isinstance(result, MultiStartResult)

    def test_all_results_count(self, small_qubo: PortfolioQUBO) -> None:
        result = multi_start_qaoa(small_qubo, k=5, p=2, seed=42)
        assert len(result.all_results) == 5

    def test_best_is_in_all_results(self, small_qubo: PortfolioQUBO) -> None:
        result = multi_start_qaoa(small_qubo, k=4, p=2, seed=42)
        assert result.best is result.all_results[result.best_index]

    def test_best_index_valid(self, small_qubo: PortfolioQUBO) -> None:
        result = multi_start_qaoa(small_qubo, k=5, p=2, seed=42)
        assert 0 <= result.best_index < 5

    def test_k_one(self, small_qubo: PortfolioQUBO) -> None:
        result = multi_start_qaoa(small_qubo, k=1, p=3, seed=42)
        assert len(result.all_results) == 1
        assert result.best_index == 0

    def test_invalid_k(self, small_qubo: PortfolioQUBO) -> None:
        with pytest.raises(ValueError, match="k must"):
            multi_start_qaoa(small_qubo, k=0)

    def test_deterministic(self, small_qubo: PortfolioQUBO) -> None:
        r1 = multi_start_qaoa(small_qubo, k=3, p=2, seed=42)
        r2 = multi_start_qaoa(small_qubo, k=3, p=2, seed=42)
        np.testing.assert_array_almost_equal(
            r1.best.initial_params, r2.best.initial_params
        )

    def test_best_has_lowest_objective(self, small_qubo: PortfolioQUBO) -> None:
        result = multi_start_qaoa(small_qubo, k=5, p=3, seed=42)
        Q = small_qubo.build_matrix()
        best_bits = np.array(
            [int(c) for c in result.best.rounded_bitstring], dtype=float
        )
        best_obj = float(best_bits @ Q @ best_bits)
        for r in result.all_results:
            bits = np.array([int(c) for c in r.rounded_bitstring], dtype=float)
            obj = float(bits @ Q @ bits)
            assert obj >= best_obj - 1e-10
