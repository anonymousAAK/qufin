"""Unit tests for ADMM-based QUBO decomposition for portfolio optimization."""

from __future__ import annotations

import numpy as np
import pytest

from qufin.backends.mock import MockBackend
from qufin.portfolio.optimizers.admm import ADMMConfig, ADMMPortfolio, ADMMResult
from qufin.portfolio.qubo import PortfolioQUBO

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def mock_backend() -> MockBackend:
    return MockBackend(seed=42)


@pytest.fixture
def small_qubo(rng: np.random.Generator) -> PortfolioQUBO:
    """A small 5-asset QUBO for tractable exhaustive search."""
    n = 5
    mu = rng.normal(0.001, 0.005, n)
    cov = rng.normal(0, 0.01, (n, n))
    cov = cov @ cov.T + 0.01 * np.eye(n)  # ensure PSD
    return PortfolioQUBO(
        mu=mu,
        cov=cov,
        gamma=1.0,
        cardinality=2,
        encoding="one_hot",
        budget_penalty=10.0,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestADMMPortfolio:
    def test_exhaustive_finds_feasible_solution(
        self, small_qubo: PortfolioQUBO, mock_backend: MockBackend
    ) -> None:
        config = ADMMConfig(
            sub_problem_size=5,
            max_iterations=10,
            sub_solver="exhaustive",
            rho=1.0,
            seed=42,
        )
        solver = ADMMPortfolio(small_qubo, config, mock_backend)
        result = solver.run()

        assert isinstance(result, ADMMResult)
        assert len(result.best_bitstring) == 5
        # The bitstring should be composed of 0s and 1s
        assert all(c in "01" for c in result.best_bitstring)

    def test_result_has_correct_fields(
        self, small_qubo: PortfolioQUBO, mock_backend: MockBackend
    ) -> None:
        config = ADMMConfig(
            sub_problem_size=5,
            max_iterations=5,
            sub_solver="exhaustive",
            seed=42,
        )
        solver = ADMMPortfolio(small_qubo, config, mock_backend)
        result = solver.run()

        assert hasattr(result, "best_bitstring")
        assert hasattr(result, "weights")
        assert hasattr(result, "objective")
        assert hasattr(result, "converged")
        assert isinstance(result.converged, bool)
        assert isinstance(result.objective, float)

    def test_residuals_tracked(
        self, small_qubo: PortfolioQUBO, mock_backend: MockBackend
    ) -> None:
        config = ADMMConfig(
            sub_problem_size=5,
            max_iterations=5,
            sub_solver="exhaustive",
            seed=42,
        )
        solver = ADMMPortfolio(small_qubo, config, mock_backend)
        result = solver.run()

        assert len(result.primal_residuals) > 0
        assert len(result.dual_residuals) > 0
        assert len(result.primal_residuals) == result.n_iterations

    def test_sub_problem_objectives_tracked(
        self, small_qubo: PortfolioQUBO, mock_backend: MockBackend
    ) -> None:
        config = ADMMConfig(
            sub_problem_size=5,
            max_iterations=5,
            sub_solver="exhaustive",
            seed=42,
        )
        solver = ADMMPortfolio(small_qubo, config, mock_backend)
        result = solver.run()

        assert len(result.sub_problem_objectives) > 0
        # Each iteration should have at least one sub-problem objective
        for iter_objs in result.sub_problem_objectives:
            assert len(iter_objs) >= 1

    def test_weights_shape(
        self, small_qubo: PortfolioQUBO, mock_backend: MockBackend
    ) -> None:
        config = ADMMConfig(
            sub_problem_size=5,
            max_iterations=10,
            sub_solver="exhaustive",
            seed=42,
        )
        solver = ADMMPortfolio(small_qubo, config, mock_backend)
        result = solver.run()

        assert result.weights.ndim == 1
        assert len(result.weights) > 0

    def test_full_problem_single_block(
        self, small_qubo: PortfolioQUBO, mock_backend: MockBackend
    ) -> None:
        """When sub_problem_size >= n_qubits, only one block exists."""
        n_qubits = small_qubo.build_matrix().shape[0]
        config = ADMMConfig(
            sub_problem_size=n_qubits,
            max_iterations=5,
            sub_solver="exhaustive",
            seed=42,
        )
        solver = ADMMPortfolio(small_qubo, config, mock_backend)
        result = solver.run()

        # Should still produce valid results
        assert isinstance(result, ADMMResult)
        assert len(result.best_bitstring) == n_qubits

    def test_n_iterations_bounded(
        self, small_qubo: PortfolioQUBO, mock_backend: MockBackend
    ) -> None:
        max_iter = 3
        config = ADMMConfig(
            sub_problem_size=5,
            max_iterations=max_iter,
            sub_solver="exhaustive",
            seed=42,
        )
        solver = ADMMPortfolio(small_qubo, config, mock_backend)
        result = solver.run()

        assert result.n_iterations <= max_iter

    def test_objective_finite(
        self, small_qubo: PortfolioQUBO, mock_backend: MockBackend
    ) -> None:
        config = ADMMConfig(
            sub_problem_size=5,
            max_iterations=5,
            sub_solver="exhaustive",
            seed=42,
        )
        solver = ADMMPortfolio(small_qubo, config, mock_backend)
        result = solver.run()

        assert np.isfinite(result.objective)
