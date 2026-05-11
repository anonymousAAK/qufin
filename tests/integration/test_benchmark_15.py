"""Integration test: 15-asset benchmark on Aer.

Verifies that both classical (CVXPY) and quantum (QAOA) solvers
can handle the 15-asset portfolio benchmark problem.
"""

from __future__ import annotations

import numpy as np
import pytest

from qufin.benchmarks.problems import portfolio_small
from qufin.benchmarks.metrics import solution_quality, feasibility_rate
from qufin.portfolio.classical.mean_variance import Objective, mean_variance
from qufin.portfolio.qubo import PortfolioQUBO


class TestClassicalBaseline15:
    """Test classical solver on the 15-asset benchmark."""

    def test_min_variance_15_assets(self) -> None:
        prob = portfolio_small()
        result = mean_variance(
            prob.mu, prob.cov,
            objective=Objective.MIN_VARIANCE,
            cardinality=prob.cardinality,
        )
        assert abs(result.weights.sum() - 1.0) < 1e-4
        assert np.sum(result.weights > 1e-4) <= prob.cardinality + 1  # MIQP tolerance
        assert result.volatility > 0

    def test_solution_quality_15(self) -> None:
        prob = portfolio_small()
        result = mean_variance(prob.mu, prob.cov, objective=Objective.MIN_VARIANCE)
        quality = solution_quality(result.weights, prob.mu, prob.cov)
        assert quality["volatility"] > 0
        assert isinstance(quality["sharpe_ratio"], float)


class TestQUBO15:
    """Test QUBO formulation on 15 assets."""

    def test_qubo_builds(self) -> None:
        prob = portfolio_small()
        qubo = PortfolioQUBO(
            mu=prob.mu, cov=prob.cov,
            cardinality=prob.cardinality,
            sector_map=prob.sector_map,
            sector_caps=prob.sector_caps,
            gamma=1.0,
        )
        Q = qubo.build_matrix()
        assert Q.shape == (15, 15)
        # Symmetric
        np.testing.assert_array_almost_equal(Q, Q.T)

    def test_feasibility_check_15(self) -> None:
        prob = portfolio_small()
        qubo = PortfolioQUBO(
            mu=prob.mu, cov=prob.cov,
            cardinality=prob.cardinality,
            sector_map=prob.sector_map,
            sector_caps=prob.sector_caps,
        )
        # 5 assets selected, respecting sector caps [2,2,3]
        # Sector 0: assets 0-4, cap 2 -> select 2
        # Sector 1: assets 5-8, cap 2 -> select 1
        # Sector 2: assets 9-14, cap 3 -> select 2
        bs = "110001000010010"  # assets 0,1,5,10,13
        feas = qubo.feasibility_check(bs)
        assert feas["cardinality"] is True
        assert feas["sector"] is True

    def test_infeasible_sector_15(self) -> None:
        prob = portfolio_small()
        qubo = PortfolioQUBO(
            mu=prob.mu, cov=prob.cov,
            cardinality=prob.cardinality,
            sector_map=prob.sector_map,
            sector_caps=prob.sector_caps,
        )
        # 3 from sector 0 (cap=2) -> infeasible
        bs = "111000000010010"  # assets 0,1,2,10,13
        feas = qubo.feasibility_check(bs)
        assert feas["sector"] is False


@pytest.mark.slow
class TestQAOA15OnAer:
    """Run QAOA on the 15-asset problem with Aer.

    These tests are slow (10-30s each) due to circuit simulation.
    """

    def test_qaoa_15_runs(self) -> None:
        from qufin.backends.qiskit_backend import QiskitAerBackend
        from qufin.portfolio.optimizers.qaoa import QAOAConfig, QAOAPortfolio

        prob = portfolio_small()
        qubo = PortfolioQUBO(
            mu=prob.mu, cov=prob.cov,
            cardinality=prob.cardinality,
            gamma=1.0,
        )
        backend = QiskitAerBackend(seed=42)
        config = QAOAConfig(
            p=1, mixer="x", shots=2048,
            maxiter=15, seed=42,
        )
        solver = QAOAPortfolio(qubo, config, backend)
        result = solver.run()

        assert len(result.best_bitstring) == 15
        assert result.weights.shape == (15,)
        assert result.wall_time_s > 0
        assert len(result.history) > 0
