"""Quantum portfolio optimizers.

QAOA, VQE, warm-start, hybrid, ADMM, multi-period, robust, Grover, IPM.
"""

from __future__ import annotations

from qufin.portfolio.optimizers.admm import ADMMConfig, ADMMPortfolio, ADMMResult
from qufin.portfolio.optimizers.grover_search import (
    GroverAdaptiveSearch,
    GroverSearchConfig,
    GroverSearchResult,
    branch_and_bound_solve,
    compare_optimizers,
)
from qufin.portfolio.optimizers.hybrid import HybridConfig, HybridOptimizer, HybridResult
from qufin.portfolio.optimizers.multi_period import (
    MultiPeriodConfig,
    MultiPeriodResult,
    compute_turnover,
    multi_period_backtest,
    multi_period_optimize,
)
from qufin.portfolio.optimizers.qaoa import QAOAConfig, QAOAPortfolio, QAOAResult
from qufin.portfolio.optimizers.qaoa_warmstart import (
    WarmStartQAOA,
    WarmStartQAOAResult,
)
from qufin.portfolio.optimizers.quantum_ipm import (
    QuantumIPMConfig,
    QuantumIPMOptimizer,
    QuantumIPMResult,
    ResourceEstimate,
    classical_ipm_solve,
    quantum_ipm_solve,
)
from qufin.portfolio.optimizers.quantum_walk import (
    SzegedyWalkConfig,
    SzegedyWalkOptimizer,
    SzegedyWalkResult,
    classical_random_walk,
)
from qufin.portfolio.optimizers.robust import (
    EllipsoidalUncertaintySet,
    RobustPortfolioOptimizer,
    RobustPortfolioResult,
    build_ellipsoidal_uncertainty,
    robust_classical,
)
from qufin.portfolio.optimizers.vqe import VQEConfig, VQEPortfolio, VQEResult
from qufin.portfolio.optimizers.warm_start import (
    MultiStartResult,
    WarmStartResult,
    continuous_relaxation,
    cvar_warm_start,
    multi_start_qaoa,
    round_solution,
    warm_start_qaoa,
    warm_start_vqe,
)

__all__ = [
    "ADMMConfig",
    "ADMMPortfolio",
    "ADMMResult",
    "EllipsoidalUncertaintySet",
    "GroverAdaptiveSearch",
    "GroverSearchConfig",
    "GroverSearchResult",
    "HybridConfig",
    "HybridOptimizer",
    "HybridResult",
    "MultiPeriodConfig",
    "MultiPeriodResult",
    "MultiStartResult",
    "QAOAConfig",
    "QAOAPortfolio",
    "QAOAResult",
    "QuantumIPMConfig",
    "QuantumIPMOptimizer",
    "QuantumIPMResult",
    "ResourceEstimate",
    "RobustPortfolioOptimizer",
    "RobustPortfolioResult",
    "SzegedyWalkConfig",
    "SzegedyWalkOptimizer",
    "SzegedyWalkResult",
    "VQEConfig",
    "VQEPortfolio",
    "VQEResult",
    "WarmStartQAOA",
    "WarmStartQAOAResult",
    "WarmStartResult",
    "branch_and_bound_solve",
    "build_ellipsoidal_uncertainty",
    "classical_ipm_solve",
    "classical_random_walk",
    "compare_optimizers",
    "compute_turnover",
    "continuous_relaxation",
    "cvar_warm_start",
    "multi_period_backtest",
    "multi_period_optimize",
    "multi_start_qaoa",
    "quantum_ipm_solve",
    "robust_classical",
    "round_solution",
    "warm_start_qaoa",
    "warm_start_vqe",
]
