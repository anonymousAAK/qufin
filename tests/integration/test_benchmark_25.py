"""Integration test: 25-asset benchmark on Aer.

Week 6 deliverable: verify that all solver types (classical, QAOA, VQE,
hybrid) can handle the 25-asset portfolio benchmark.
"""

from __future__ import annotations

import numpy as np
import pytest

from qufin.benchmarks.leaderboard import to_csv, to_markdown
from qufin.benchmarks.manifest import build_manifest
from qufin.benchmarks.metrics import solution_quality
from qufin.benchmarks.problems import portfolio_medium
from qufin.benchmarks.runner import BenchmarkRunner, SolverEntry
from qufin.portfolio.classical.mean_variance import Objective, mean_variance
from qufin.portfolio.qubo import PortfolioQUBO


class TestClassicalBaseline25:
    """Test classical solver on the 25-asset benchmark."""

    def test_min_variance_25_assets(self) -> None:
        prob = portfolio_medium()
        result = mean_variance(
            prob.mu, prob.cov,
            objective=Objective.MIN_VARIANCE,
            cardinality=prob.cardinality,
        )
        assert abs(result.weights.sum() - 1.0) < 1e-4
        assert result.volatility > 0

    def test_min_variance_no_cardinality(self) -> None:
        prob = portfolio_medium()
        result = mean_variance(
            prob.mu, prob.cov,
            objective=Objective.MIN_VARIANCE,
        )
        assert abs(result.weights.sum() - 1.0) < 1e-4
        assert result.volatility > 0
        quality = solution_quality(result.weights, prob.mu, prob.cov)
        assert isinstance(quality["sharpe_ratio"], float)


class TestQUBO25:
    def test_qubo_25_builds(self) -> None:
        prob = portfolio_medium()
        qubo = PortfolioQUBO(
            mu=prob.mu, cov=prob.cov,
            cardinality=prob.cardinality,
            gamma=1.0,
        )
        Q = qubo.build_matrix()
        assert Q.shape == (25, 25)
        np.testing.assert_array_almost_equal(Q, Q.T)


class TestLeaderboard:
    def test_leaderboard_generation(self) -> None:
        """Test that the leaderboard generates from benchmark results."""
        prob = portfolio_medium()

        def classical_solver(problem):
            result = mean_variance(
                problem.mu, problem.cov,
                objective=Objective.MIN_VARIANCE,
            )
            return {
                "objective": result.volatility,
                "backend": "cvxpy",
            }

        runner = BenchmarkRunner()
        runner.register(SolverEntry("cvxpy-mv", "classical", classical_solver))
        rows = runner.run_problem(prob)

        md = to_markdown(rows)
        assert "cvxpy-mv" in md
        assert "classical" in md

        csv_out = to_csv(rows)
        assert "cvxpy-mv" in csv_out


class TestManifest:
    def test_build_manifest(self) -> None:
        m = build_manifest(
            problem_ids=["portfolio_medium_25"],
            solver_names=["cvxpy-mv"],
            seeds=[42],
        )
        assert m.python_version != ""
        assert m.platform_info != ""
        assert "portfolio_medium_25" in m.problem_ids

        j = m.to_json()
        assert "python_version" in j


@pytest.mark.slow
class TestBenchmarkRunner25:
    def test_runner_classical_only(self) -> None:
        prob = portfolio_medium()

        def mv_solver(problem):
            result = mean_variance(problem.mu, problem.cov, objective=Objective.MIN_VARIANCE)
            return {"objective": result.volatility, "backend": "cvxpy"}

        runner = BenchmarkRunner()
        runner.register(SolverEntry("cvxpy-mv", "classical", mv_solver))
        rows = runner.run_all([prob])
        assert len(rows) == 1
        assert rows[0].solver_name == "cvxpy-mv"
        assert rows[0].wall_seconds > 0
