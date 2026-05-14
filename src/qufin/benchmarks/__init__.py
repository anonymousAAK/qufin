"""Benchmark harness: standardized problems, runner, metrics, leaderboard."""

from __future__ import annotations

from qufin.benchmarks.leaderboard import generate_leaderboard, to_csv, to_markdown
from qufin.benchmarks.manifest import Manifest, build_manifest
from qufin.benchmarks.problems import (
    PortfolioProblem,
    Problem,
    all_problems,
    portfolio_large,
    portfolio_medium,
    portfolio_small,
)
from qufin.benchmarks.resource_estimation import (
    AlgorithmResource,
    BreakEvenPoint,
    SurfaceCodeOverhead,
    break_even_summary,
    compute_break_even_timeline,
    compute_surface_code_overhead,
    estimate_qae_pricing,
    estimate_qaoa_optimization,
    estimate_quantum_credit,
    estimate_quantum_var,
    generate_resource_table,
    resource_table_to_dicts,
)
from qufin.benchmarks.runner import BenchmarkRunner, SolverEntry

__all__ = [
    "AlgorithmResource",
    "BenchmarkRunner",
    "BreakEvenPoint",
    "Manifest",
    "PortfolioProblem",
    "Problem",
    "SolverEntry",
    "SurfaceCodeOverhead",
    "all_problems",
    "break_even_summary",
    "build_manifest",
    "compute_break_even_timeline",
    "compute_surface_code_overhead",
    "estimate_qae_pricing",
    "estimate_qaoa_optimization",
    "estimate_quantum_credit",
    "estimate_quantum_var",
    "generate_leaderboard",
    "generate_resource_table",
    "portfolio_large",
    "portfolio_medium",
    "portfolio_small",
    "resource_table_to_dicts",
    "to_csv",
    "to_markdown",
]
