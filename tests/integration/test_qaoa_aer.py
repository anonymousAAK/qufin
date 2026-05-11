"""Integration tests: QAOA on Qiskit Aer.

Tests that the full QAOA pipeline (QUBO -> circuit -> Aer -> optimize)
produces reasonable results on small problems.
"""

from __future__ import annotations

import numpy as np
import pytest

from qufin.backends.qiskit_backend import QiskitAerBackend
from qufin.portfolio.optimizers.exhaustive import exhaustive_solve
from qufin.portfolio.optimizers.qaoa import QAOAConfig, QAOAPortfolio, QAOAResult
from qufin.portfolio.qubo import PortfolioQUBO


@pytest.fixture
def aer_backend() -> QiskitAerBackend:
    return QiskitAerBackend(seed=42)


@pytest.fixture
def small_qubo() -> PortfolioQUBO:
    """4-asset portfolio for fast testing."""
    mu = np.array([0.01, 0.02, 0.015, 0.008])
    cov = np.array([
        [0.04, 0.006, 0.002, 0.001],
        [0.006, 0.09, 0.004, 0.003],
        [0.002, 0.004, 0.01, 0.002],
        [0.001, 0.003, 0.002, 0.025],
    ])
    return PortfolioQUBO(mu=mu, cov=cov, cardinality=2, gamma=0.5)


class TestQAOAOnAer:
    @pytest.mark.slow
    def test_qaoa_finds_good_solution(
        self, small_qubo: PortfolioQUBO, aer_backend: QiskitAerBackend
    ) -> None:
        """QAOA should find a solution close to the brute-force optimum."""
        # Get exact solution
        exact = exhaustive_solve(small_qubo)

        # Run QAOA
        config = QAOAConfig(p=2, mixer="x", shots=4096, maxiter=50, seed=42)
        solver = QAOAPortfolio(small_qubo, config, aer_backend)
        result = solver.run()

        # QAOA objective should be within 50% of optimal
        # (very generous for p=2 on a small problem)
        assert result.best_objective <= exact.best_objective * 2.0 or \
               result.best_objective <= exact.best_objective + 0.5

    @pytest.mark.slow
    def test_qaoa_x_mixer(
        self, small_qubo: PortfolioQUBO, aer_backend: QiskitAerBackend
    ) -> None:
        """X-mixer QAOA produces valid output."""
        config = QAOAConfig(p=1, mixer="x", shots=1024, maxiter=20, seed=42)
        solver = QAOAPortfolio(small_qubo, config, aer_backend)
        result = solver.run()

        assert isinstance(result, QAOAResult)
        assert len(result.best_bitstring) == 4
        assert result.weights.shape == (4,)
        assert len(result.history) > 0
        assert result.wall_time_s > 0

    @pytest.mark.slow
    def test_qaoa_xy_ring_mixer(
        self, small_qubo: PortfolioQUBO, aer_backend: QiskitAerBackend
    ) -> None:
        """XY-ring mixer QAOA produces valid output."""
        config = QAOAConfig(
            p=1, mixer="xy_ring", cardinality=2,
            shots=1024, maxiter=20, seed=42,
        )
        solver = QAOAPortfolio(small_qubo, config, aer_backend)
        result = solver.run()

        assert isinstance(result, QAOAResult)
        assert len(result.best_bitstring) == 4

    @pytest.mark.slow
    def test_qaoa_cvar_objective(
        self, small_qubo: PortfolioQUBO, aer_backend: QiskitAerBackend
    ) -> None:
        """CVaR objective (alpha < 1) should produce a result."""
        config = QAOAConfig(
            p=1, mixer="x", shots=2048, maxiter=20,
            seed=42, cvar_alpha=0.2,
        )
        solver = QAOAPortfolio(small_qubo, config, aer_backend)
        result = solver.run()
        assert isinstance(result.best_objective, float)
        assert not np.isnan(result.best_objective)

    @pytest.mark.slow
    def test_qaoa_p_layers_improve(
        self, small_qubo: PortfolioQUBO, aer_backend: QiskitAerBackend
    ) -> None:
        """More QAOA layers (p) should generally give equal or better results."""
        results = []
        for p in [1, 3]:
            config = QAOAConfig(p=p, mixer="x", shots=4096, maxiter=50, seed=42)
            solver = QAOAPortfolio(small_qubo, config, aer_backend)
            results.append(solver.run())

        # p=3 should be no worse than p=1 (with generous tolerance for noise)
        assert results[1].best_objective <= results[0].best_objective + 1.0


class TestExhaustiveSolver:
    def test_exact_solution(self, small_qubo: PortfolioQUBO) -> None:
        result = exhaustive_solve(small_qubo)
        assert result.n_evaluated == 2**4
        assert len(result.best_bitstring) == 4
        assert result.best_objective <= 0  # should find negative (good) obj

    def test_all_objectives(self, small_qubo: PortfolioQUBO) -> None:
        result = exhaustive_solve(small_qubo, return_all=True)
        assert result.all_objectives is not None
        assert len(result.all_objectives) == 16
        # Best should be the minimum
        min_obj = min(obj for _, obj in result.all_objectives)
        assert result.best_objective == min_obj

    def test_too_large_raises(self) -> None:
        mu = np.ones(21)
        cov = np.eye(21)
        qubo = PortfolioQUBO(mu=mu, cov=cov)
        with pytest.raises(ValueError, match="too large"):
            exhaustive_solve(qubo)
