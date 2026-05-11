"""Unit tests for warm-start strategies."""

from __future__ import annotations

import numpy as np
import pytest

from qufin.portfolio.optimizers.warm_start import (
    WarmStartResult,
    continuous_relaxation,
    round_solution,
    warm_start_qaoa,
    warm_start_vqe,
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


class TestContinuousRelaxation:
    def test_solution_in_bounds(self, small_qubo: PortfolioQUBO) -> None:
        x = continuous_relaxation(small_qubo)
        assert x.shape == (4,)
        assert np.all(x >= -1e-10)
        assert np.all(x <= 1.0 + 1e-10)

    def test_cardinality_sum(self, small_qubo: PortfolioQUBO) -> None:
        x = continuous_relaxation(small_qubo)
        # Should approximately sum to cardinality (K=2)
        assert abs(np.sum(x) - 2.0) < 0.1

    def test_no_cardinality(self) -> None:
        mu = np.array([0.01, 0.02, 0.015])
        cov = np.eye(3) * 0.04
        qubo = PortfolioQUBO(mu=mu, cov=cov, gamma=1.0)
        x = continuous_relaxation(qubo)
        assert x.shape == (3,)
        assert np.all(x >= -1e-10)
        assert np.all(x <= 1.0 + 1e-10)


class TestRoundSolution:
    def test_with_cardinality(self) -> None:
        x = np.array([0.1, 0.9, 0.7, 0.3])
        bs = round_solution(x, cardinality=2)
        assert sum(int(c) for c in bs) == 2
        # Should select indices 1 and 2 (highest values)
        assert bs[1] == "1"
        assert bs[2] == "1"

    def test_without_cardinality(self) -> None:
        x = np.array([0.1, 0.9, 0.7, 0.3])
        bs = round_solution(x)
        assert len(bs) == 4
        assert bs == "0110"

    def test_edge_values(self) -> None:
        x = np.array([0.0, 1.0, 0.5, 0.5])
        bs = round_solution(x)
        # 0.5 should round to 1
        assert bs == "0111"


class TestWarmStartQAOA:
    def test_returns_valid_result(self, small_qubo: PortfolioQUBO) -> None:
        result = warm_start_qaoa(small_qubo, p=3, seed=42)
        assert isinstance(result, WarmStartResult)
        assert result.initial_params.shape == (6,)  # 3 gammas + 3 betas
        assert result.relaxed_solution.shape == (4,)
        assert len(result.rounded_bitstring) == 4

    def test_deterministic(self, small_qubo: PortfolioQUBO) -> None:
        r1 = warm_start_qaoa(small_qubo, p=2, seed=42)
        r2 = warm_start_qaoa(small_qubo, p=2, seed=42)
        np.testing.assert_array_almost_equal(r1.initial_params, r2.initial_params)

    def test_gammas_small(self, small_qubo: PortfolioQUBO) -> None:
        result = warm_start_qaoa(small_qubo, p=3, seed=42)
        gammas = result.initial_params[:3]
        # Gammas should be small (close to identity problem unitary)
        assert np.all(np.abs(gammas) < 1.0)


class TestWarmStartVQE:
    def test_returns_valid_result(self, small_qubo: PortfolioQUBO) -> None:
        result = warm_start_vqe(small_qubo, n_params=24, seed=42)
        assert isinstance(result, WarmStartResult)
        assert result.initial_params.shape == (24,)
        assert result.relaxed_solution.shape == (4,)

    def test_deterministic(self, small_qubo: PortfolioQUBO) -> None:
        r1 = warm_start_vqe(small_qubo, n_params=16, seed=42)
        r2 = warm_start_vqe(small_qubo, n_params=16, seed=42)
        np.testing.assert_array_almost_equal(r1.initial_params, r2.initial_params)
