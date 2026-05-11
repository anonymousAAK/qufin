"""Hybrid classical-quantum optimizer: SDP relaxation + QAOA refinement.

Combines classical continuous optimization with quantum combinatorial
search: solve the relaxed problem classically, round to a feasible
binary solution, then refine with QAOA warm-started from the relaxation.

References
----------
Egger et al., Quantum 5, 479 (2021) — Warm-starting quantum optimization.
Goemans & Williamson, JACM 42(6), 1995 — SDP relaxation for MAX-CUT.
Brandhofer et al., arXiv:2207.10555 — portfolio QAOA benchmarking.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from qufin.backends.base import Backend
from qufin.portfolio.optimizers.qaoa import QAOAConfig, QAOAPortfolio
from qufin.portfolio.optimizers.warm_start import (
    continuous_relaxation,
    round_solution,
    warm_start_qaoa,
)
from qufin.portfolio.qubo import PortfolioQUBO
from qufin.utils.results import Result


@dataclass
class HybridConfig:
    """Configuration for hybrid classical-quantum optimizer."""

    # QAOA refinement config
    qaoa_p: int = 2
    qaoa_mixer: str = "x"
    qaoa_optimizer: str = "COBYLA"
    qaoa_maxiter: int = 150
    qaoa_shots: int = 8192
    qaoa_cvar_alpha: float = 0.5
    seed: int | None = 42


@dataclass
class HybridResult(Result):
    """Result from hybrid optimization."""

    # Classical relaxation stage
    relaxed_solution: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    relaxed_objective: float = float("inf")
    rounded_bitstring: str = ""
    rounded_objective: float = float("inf")

    # Quantum refinement stage
    qaoa_bitstring: str = ""
    qaoa_objective: float = float("inf")

    # Final result
    best_bitstring: str = ""
    best_objective: float = float("inf")
    weights: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    feasible: bool = False

    # Timings
    classical_time_s: float = 0.0
    quantum_time_s: float = 0.0


class HybridOptimizer:
    """Hybrid classical-quantum portfolio optimizer.

    Pipeline:
    1. Solve continuous relaxation of the QUBO (classical, fast).
    2. Round the relaxed solution to a feasible binary solution.
    3. Warm-start QAOA from the relaxation to refine the solution.
    4. Return the best solution found across both stages.
    """

    def __init__(
        self,
        qubo: PortfolioQUBO,
        config: HybridConfig,
        backend: Backend,
    ) -> None:
        self.qubo = qubo
        self.config = config
        self.backend = backend

    def run(self) -> HybridResult:
        """Run the hybrid optimization pipeline."""
        total_start = time.perf_counter()

        # Stage 1: Classical relaxation + rounding
        classical_start = time.perf_counter()
        x_relaxed = continuous_relaxation(self.qubo)
        Q = self.qubo.build_matrix()
        relaxed_obj = float(x_relaxed @ Q @ x_relaxed)

        rounded_bs = round_solution(x_relaxed, self.qubo.cardinality)
        rounded_obj = self.qubo.evaluate(rounded_bs)
        classical_time = time.perf_counter() - classical_start

        # Stage 2: QAOA refinement with warm-start
        quantum_start = time.perf_counter()
        ws = warm_start_qaoa(
            self.qubo,
            p=self.config.qaoa_p,
            seed=self.config.seed,
        )

        qaoa_config = QAOAConfig(
            p=self.config.qaoa_p,
            mixer=self.config.qaoa_mixer,
            optimizer=self.config.qaoa_optimizer,
            maxiter=self.config.qaoa_maxiter,
            shots=self.config.qaoa_shots,
            seed=self.config.seed,
            cvar_alpha=self.config.qaoa_cvar_alpha,
            initial_gammas=ws.initial_params[: self.config.qaoa_p],
            initial_betas=ws.initial_params[self.config.qaoa_p :],
        )

        qaoa = QAOAPortfolio(self.qubo, qaoa_config, self.backend)
        qaoa_result = qaoa.run()
        quantum_time = time.perf_counter() - quantum_start

        # Stage 3: Pick the best solution
        qaoa_bs = qaoa_result.best_bitstring
        qaoa_obj = qaoa_result.best_objective

        if qaoa_obj <= rounded_obj:
            best_bs = qaoa_bs
            best_obj = qaoa_obj
        else:
            best_bs = rounded_bs
            best_obj = rounded_obj

        weights = self.qubo.decode_weights(best_bs)
        feasibility = self.qubo.feasibility_check(best_bs)
        wall_time = time.perf_counter() - total_start

        return HybridResult(
            value=best_obj,
            n_shots=self.config.qaoa_shots,
            wall_time_s=wall_time,
            backend_id=self.backend.backend_id,
            seed=self.config.seed,
            relaxed_solution=x_relaxed,
            relaxed_objective=relaxed_obj,
            rounded_bitstring=rounded_bs,
            rounded_objective=rounded_obj,
            qaoa_bitstring=qaoa_bs,
            qaoa_objective=qaoa_obj,
            best_bitstring=best_bs,
            best_objective=best_obj,
            weights=weights,
            feasible=all(feasibility.values()) if feasibility else True,
            classical_time_s=classical_time,
            quantum_time_s=quantum_time,
        )
