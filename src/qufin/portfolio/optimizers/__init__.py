"""Quantum portfolio optimizers: QAOA, VQE, warm-start, hybrid, ADMM, multi-period."""

from __future__ import annotations

from qufin.portfolio.optimizers.admm import ADMMConfig, ADMMPortfolio, ADMMResult
from qufin.portfolio.optimizers.hybrid import HybridConfig, HybridOptimizer, HybridResult
from qufin.portfolio.optimizers.multi_period import (
    MultiPeriodConfig,
    MultiPeriodResult,
    compute_turnover,
    multi_period_backtest,
    multi_period_optimize,
)
from qufin.portfolio.optimizers.qaoa import QAOAConfig, QAOAPortfolio, QAOAResult
from qufin.portfolio.optimizers.vqe import VQEConfig, VQEPortfolio, VQEResult
from qufin.portfolio.optimizers.warm_start import (
    WarmStartResult,
    continuous_relaxation,
    round_solution,
    warm_start_qaoa,
    warm_start_vqe,
)

__all__ = [
    "ADMMConfig",
    "ADMMPortfolio",
    "ADMMResult",
    "HybridConfig",
    "HybridOptimizer",
    "HybridResult",
    "MultiPeriodConfig",
    "MultiPeriodResult",
    "QAOAConfig",
    "QAOAPortfolio",
    "QAOAResult",
    "VQEConfig",
    "VQEPortfolio",
    "VQEResult",
    "WarmStartResult",
    "compute_turnover",
    "continuous_relaxation",
    "multi_period_backtest",
    "multi_period_optimize",
    "round_solution",
    "warm_start_qaoa",
    "warm_start_vqe",
]
