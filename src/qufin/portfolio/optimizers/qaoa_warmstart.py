"""QAOA with warm-start from continuous relaxation.

Solves the continuous (mean-variance) relaxation first, maps the solution
to informed initial QAOA parameters, then runs QAOA with that
initialization.  Typically converges faster and to better optima than
random initialization.

References
----------
Egger et al., Quantum 5, 479 (2021) -- Warm-starting quantum optimization.
Brandhofer et al., arXiv:2207.10555 -- portfolio QAOA benchmarking.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from qufin.backends.base import Backend
from qufin.portfolio.optimizers.qaoa import QAOAConfig, QAOAPortfolio, QAOAResult
from qufin.portfolio.optimizers.warm_start import continuous_relaxation, round_solution
from qufin.portfolio.qubo import PortfolioQUBO


@dataclass
class WarmStartQAOAResult(QAOAResult):
    """Extended QAOA result with warm-start metadata."""

    relaxed_solution: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(0)
    )
    relaxed_objective: float = float("inf")
    rounded_bitstring_ws: str = ""
    improvement_over_random: float = 0.0


class WarmStartQAOA:
    """QAOA solver warm-started from the continuous relaxation.

    Parameters
    ----------
    qubo : PortfolioQUBO
        The portfolio QUBO problem.
    config : QAOAConfig
        QAOA hyper-parameters (``initial_betas``/``initial_gammas`` are
        overridden by the warm-start heuristic).
    backend : Backend
        Quantum backend.
    """

    def __init__(
        self,
        qubo: PortfolioQUBO,
        config: QAOAConfig,
        backend: Backend,
    ) -> None:
        self.qubo = qubo
        self.config = config
        self.backend = backend

    # ------------------------------------------------------------------
    def _warm_params(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], float, str]:
        """Solve continuous relaxation and derive initial QAOA params."""
        x_relaxed = continuous_relaxation(self.qubo)
        Q = self.qubo.build_matrix()
        relaxed_obj = float(x_relaxed @ Q @ x_relaxed)
        rounded_bs = round_solution(x_relaxed, self.qubo.cardinality)

        # Egger heuristic: theta_i = arcsin(sqrt(x_i))
        thetas = np.arcsin(np.sqrt(np.clip(x_relaxed, 0.0, 1.0)))
        avg_theta = float(np.mean(thetas))

        p = self.config.p
        rng = np.random.default_rng(self.config.seed)

        betas = np.full(p, avg_theta * 0.3) + rng.normal(0, 0.05, p)
        energy_scale = max(float(np.max(np.abs(Q))), 1e-8)
        gammas = np.full(p, 0.1 / energy_scale) + rng.normal(0, 0.02 / energy_scale, p)

        return x_relaxed, gammas, betas, relaxed_obj, rounded_bs

    # ------------------------------------------------------------------
    def run(self) -> WarmStartQAOAResult:
        """Run warm-started QAOA and return result."""
        start = time.perf_counter()

        x_relaxed, gammas, betas, relaxed_obj, rounded_bs = self._warm_params()

        # Override config with warm-start params
        ws_config = QAOAConfig(
            p=self.config.p,
            mixer=self.config.mixer,
            cardinality=self.config.cardinality,
            optimizer=self.config.optimizer,
            maxiter=self.config.maxiter,
            shots=self.config.shots,
            seed=self.config.seed,
            cvar_alpha=self.config.cvar_alpha,
            initial_betas=betas,
            initial_gammas=gammas,
        )

        solver = QAOAPortfolio(self.qubo, ws_config, self.backend)
        qaoa_result = solver.run()

        wall_time = time.perf_counter() - start

        return WarmStartQAOAResult(
            value=qaoa_result.value,
            n_shots=qaoa_result.n_shots,
            wall_time_s=wall_time,
            backend_id=qaoa_result.backend_id,
            seed=qaoa_result.seed,
            best_bitstring=qaoa_result.best_bitstring,
            best_objective=qaoa_result.best_objective,
            weights=qaoa_result.weights,
            betas=qaoa_result.betas,
            gammas=qaoa_result.gammas,
            history=qaoa_result.history,
            feasible=qaoa_result.feasible,
            relaxed_solution=x_relaxed,
            relaxed_objective=relaxed_obj,
            rounded_bitstring_ws=rounded_bs,
        )
