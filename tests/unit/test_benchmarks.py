"""Tests for benchmark infrastructure."""

from __future__ import annotations

import numpy as np
import pytest

from qufin.benchmarks.metrics import (
    approximation_ratio,
    feasibility_rate,
    relative_error,
    solution_quality,
)
from qufin.benchmarks.problems import (
    all_problems,
    option_european_atm,
    portfolio_large,
    portfolio_medium,
    portfolio_small,
)
from qufin.benchmarks.runner import BenchmarkRow, BenchmarkRunner, SolverEntry


class TestProblems:
    def test_portfolio_small(self) -> None:
        prob = portfolio_small()
        assert prob.problem_id == "portfolio_small_15"
        assert prob.mu.shape == (15,)
        assert prob.cov.shape == (15, 15)
        assert prob.cardinality == 5
        assert prob.sector_map is not None
        assert prob.sector_caps is not None
        assert len(prob.tickers) == 15

    def test_portfolio_medium(self) -> None:
        prob = portfolio_medium()
        assert prob.mu.shape == (25,)
        assert prob.cardinality == 8

    def test_portfolio_large(self) -> None:
        prob = portfolio_large()
        assert prob.mu.shape == (50,)
        assert prob.cardinality == 15

    def test_cov_positive_definite(self) -> None:
        for prob_fn in [portfolio_small, portfolio_medium, portfolio_large]:
            prob = prob_fn()
            eigvals = np.linalg.eigvalsh(prob.cov)
            assert np.all(eigvals > 0), f"Cov not PD for {prob.problem_id}"

    def test_cov_symmetric(self) -> None:
        for prob_fn in [portfolio_small, portfolio_medium, portfolio_large]:
            prob = prob_fn()
            np.testing.assert_array_almost_equal(prob.cov, prob.cov.T)

    def test_option_european(self) -> None:
        prob = option_european_atm()
        assert prob.reference_value == pytest.approx(10.4506, abs=0.01)

    def test_all_problems_returns_list(self) -> None:
        probs = all_problems()
        assert len(probs) >= 4

    def test_reproducible_seed(self) -> None:
        p1 = portfolio_small(seed=42)
        p2 = portfolio_small(seed=42)
        np.testing.assert_array_equal(p1.mu, p2.mu)
        np.testing.assert_array_equal(p1.cov, p2.cov)

    def test_different_seeds(self) -> None:
        p1 = portfolio_small(seed=42)
        p2 = portfolio_small(seed=99)
        assert not np.allclose(p1.mu, p2.mu)


class TestMetrics:
    def test_relative_error(self) -> None:
        assert relative_error(10.0, 10.0) == 0.0
        assert relative_error(11.0, 10.0) == pytest.approx(0.1)

    def test_relative_error_zero_ref(self) -> None:
        assert relative_error(0.5, 0.0) == 0.5

    def test_approximation_ratio(self) -> None:
        assert approximation_ratio(2.0, 1.0) == 2.0
        assert approximation_ratio(1.0, 1.0) == 1.0

    def test_solution_quality(self) -> None:
        mu = np.array([0.01, 0.02])
        cov = np.array([[0.04, 0.01], [0.01, 0.09]])
        w = np.array([0.6, 0.4])
        q = solution_quality(w, mu, cov)
        assert q["expected_return"] == pytest.approx(0.014)
        assert q["volatility"] > 0

    def test_feasibility_rate(self) -> None:
        bitstrings = ["110", "101", "111", "100"]
        rate = feasibility_rate(bitstrings, cardinality=2)
        assert rate == pytest.approx(0.5)  # 2 out of 4 have HW=2

    def test_feasibility_rate_no_constraint(self) -> None:
        bitstrings = ["110", "101"]
        rate = feasibility_rate(bitstrings, cardinality=None)
        assert rate == 1.0


class TestRunner:
    def test_runner_registers_and_runs(self) -> None:
        runner = BenchmarkRunner()

        def dummy_solver(problem):
            return {"objective": 42.0, "backend": "dummy"}

        runner.register(SolverEntry("dummy", "classical", dummy_solver))
        prob = option_european_atm()
        rows = runner.run_problem(prob)
        assert len(rows) == 1
        assert rows[0].objective == 42.0
        assert rows[0].solver_name == "dummy"
        assert rows[0].rel_error is not None

    def test_runner_handles_error(self) -> None:
        runner = BenchmarkRunner()

        def failing_solver(problem):
            raise RuntimeError("boom")

        runner.register(SolverEntry("fail", "quantum", failing_solver))
        rows = runner.run_problem(option_european_atm())
        assert len(rows) == 1
        assert np.isnan(rows[0].objective)

    def test_runner_multiple_solvers(self) -> None:
        runner = BenchmarkRunner()
        runner.register(SolverEntry("s1", "classical", lambda p: {"objective": 1.0}))
        runner.register(SolverEntry("s2", "quantum", lambda p: {"objective": 2.0}))
        rows = runner.run_problem(option_european_atm())
        assert len(rows) == 2
