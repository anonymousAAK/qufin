"""Benchmark runner: dispatches problems to registered solvers."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from qufin.benchmarks.problems import Problem


@dataclass
class BenchmarkRow:
    """Single benchmark result row."""

    problem_id: str
    solver_name: str
    solver_family: str  # "quantum" or "classical"
    objective: float
    rel_error: float | None
    wall_seconds: float
    circuit_depth: int | None
    n_qubits: int | None
    backend: str
    seed: int | None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class SolverEntry:
    """A registered solver."""

    name: str
    family: str  # "quantum" or "classical"
    solve_fn: Callable[[Problem], dict[str, Any]]


class BenchmarkRunner:
    """Runs benchmark problems against registered solvers.

    Usage
    -----
    runner = BenchmarkRunner()
    runner.register(SolverEntry("qaoa-p3-x", "quantum", my_qaoa_fn))
    runner.register(SolverEntry("cvxpy-mv", "classical", my_mv_fn))
    results = runner.run_problem(portfolio_small())
    """

    def __init__(self) -> None:
        self._solvers: list[SolverEntry] = []

    def register(self, entry: SolverEntry) -> None:
        self._solvers.append(entry)

    def run_problem(self, problem: Problem) -> list[BenchmarkRow]:
        """Run all registered solvers on a single problem."""
        rows = []
        for solver in self._solvers:
            start = time.perf_counter()
            try:
                result = solver.solve_fn(problem)
            except Exception as e:
                result = {"objective": float("nan"), "error": str(e)}
            wall = time.perf_counter() - start

            objective = result.get("objective", float("nan"))
            rel_error = None
            if problem.reference_value is not None and problem.reference_value != 0:
                rel_error = abs(objective - problem.reference_value) / abs(
                    problem.reference_value
                )

            rows.append(
                BenchmarkRow(
                    problem_id=problem.problem_id,
                    solver_name=solver.name,
                    solver_family=solver.family,
                    objective=objective,
                    rel_error=rel_error,
                    wall_seconds=wall,
                    circuit_depth=result.get("circuit_depth"),
                    n_qubits=result.get("n_qubits"),
                    backend=result.get("backend", ""),
                    seed=result.get("seed"),
                    extra={
                        k: v
                        for k, v in result.items()
                        if k not in ("objective", "circuit_depth", "n_qubits", "backend", "seed")
                    },
                )
            )
        return rows

    def run_all(self, problems: list[Problem]) -> list[BenchmarkRow]:
        """Run all solvers on all problems."""
        rows = []
        for problem in problems:
            rows.extend(self.run_problem(problem))
        return rows
