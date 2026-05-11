"""QAOA portfolio optimizer (X-mixer and XY-ring/Dicke for cardinality).

References
----------
Farhi, Goldstone, Gutmann, arXiv:1411.4028.
Hadfield et al., Algorithms 12:34 (2019) — Quantum Alternating Operator Ansatz.
Wang, Rubin, Dominy, Rieffel, PRA 101:012320 (2020) — XY mixers.
Bartschi et al., npj QI (2024) — Dicke initial state for cardinality.
Brandhofer et al., arXiv:2207.10555 — portfolio QAOA benchmarking.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from qufin.backends.base import Backend, CircuitResult
from qufin.portfolio.mixers import DickeInitialState, get_mixer
from qufin.portfolio.qubo import PortfolioQUBO
from qufin.utils.results import Result


@dataclass
class QAOAConfig:
    p: int = 3
    mixer: Literal["x", "xy_ring", "xy_full", "grover"] = "x"
    cardinality: int | None = None
    optimizer: str = "COBYLA"
    maxiter: int = 200
    shots: int = 8192
    seed: int | None = 42
    cvar_alpha: float = 1.0  # 1.0 = standard mean, <1.0 = CVaR
    initial_betas: NDArray[np.float64] | None = None
    initial_gammas: NDArray[np.float64] | None = None


@dataclass
class QAOAResult(Result):
    best_bitstring: str = ""
    best_objective: float = float("inf")
    weights: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    betas: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    gammas: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    history: list[float] = field(default_factory=list)
    feasible: bool = False


class QAOAPortfolio:
    """QAOA solver for the cardinality-constrained Markowitz QUBO."""

    def __init__(self, qubo: PortfolioQUBO, config: QAOAConfig, backend: Backend) -> None:
        self.qubo = qubo
        self.config = config
        self.backend = backend
        self._Q = qubo.build_matrix()
        self._mixer = get_mixer(config.mixer, qubo.n_qubits)
        self._history: list[float] = []

    def _build_circuit(self, betas: NDArray[np.float64], gammas: NDArray[np.float64]) -> object:
        """Build the QAOA circuit for given parameters."""
        from qiskit.circuit import QuantumCircuit

        n = self.qubo.n_qubits
        qc = QuantumCircuit(n, n)

        # Initial state
        if (
            self.config.cardinality is not None
            and self._mixer.preserves_hamming_weight
        ):
            dicke = DickeInitialState(n, self.config.cardinality)
            qc.compose(dicke.circuit(), inplace=True)
        else:
            qc.h(range(n))

        # QAOA layers
        for layer in range(self.config.p):
            gamma = gammas[layer]
            # Problem unitary: exp(-i * gamma * C)
            for i in range(n):
                if abs(self._Q[i, i]) > 1e-10:
                    qc.rz(2 * gamma * self._Q[i, i], i)
            for i in range(n):
                for j in range(i + 1, n):
                    q_ij = self._Q[i, j] + self._Q[j, i]
                    if abs(q_ij) > 1e-10:
                        qc.cx(i, j)
                        qc.rz(2 * gamma * q_ij, j)
                        qc.cx(i, j)

            # Mixer unitary
            mixer_circ = self._mixer.circuit(betas[layer])
            qc.compose(mixer_circ, inplace=True)

        qc.measure(range(n), range(n))
        return qc

    def _evaluate_counts(self, result: CircuitResult) -> float:
        """Compute CVaR-alpha objective from measurement counts."""
        objectives = []
        for bitstring, count in result.counts.items():
            obj = self.qubo.evaluate(bitstring)
            objectives.extend([obj] * count)

        objectives.sort()
        alpha = self.config.cvar_alpha
        cutoff = max(1, int(len(objectives) * alpha))
        return float(np.mean(objectives[:cutoff]))

    def _objective(self, params: NDArray[np.float64]) -> float:
        """QAOA objective function for the classical optimizer."""
        p = self.config.p
        gammas = params[:p]
        betas = params[p:]
        circuit = self._build_circuit(betas, gammas)
        result = self.backend.run(circuit, shots=self.config.shots)
        val = self._evaluate_counts(result)
        self._history.append(val)
        return val

    def run(self) -> QAOAResult:
        """Optimize QAOA parameters and return the best portfolio."""
        start = time.perf_counter()
        p = self.config.p
        rng = np.random.default_rng(self.config.seed)

        gammas0 = (
            self.config.initial_gammas
            if self.config.initial_gammas is not None
            else rng.uniform(0, np.pi, p)
        )
        betas0 = (
            self.config.initial_betas
            if self.config.initial_betas is not None
            else rng.uniform(0, np.pi, p)
        )
        x0 = np.concatenate([gammas0, betas0])

        self._history = []
        opt_result = minimize(
            self._objective,
            x0,
            method=self.config.optimizer,
            options={"maxiter": self.config.maxiter},
        )

        opt_gammas = opt_result.x[:p]
        opt_betas = opt_result.x[p:]

        # Final evaluation to get best bitstring
        circuit = self._build_circuit(opt_betas, opt_gammas)
        final_result = self.backend.run(circuit, shots=self.config.shots)
        best_bs = final_result.most_frequent
        best_obj = self.qubo.evaluate(best_bs)

        # Decode weights and check feasibility
        weights = self.qubo.decode_weights(best_bs)
        feasibility = self.qubo.feasibility_check(best_bs)

        wall_time = time.perf_counter() - start

        return QAOAResult(
            value=best_obj,
            n_shots=self.config.shots,
            wall_time_s=wall_time,
            backend_id=self.backend.backend_id,
            seed=self.config.seed,
            best_bitstring=best_bs,
            best_objective=best_obj,
            weights=weights,
            betas=opt_betas,
            gammas=opt_gammas,
            history=self._history,
            feasible=all(feasibility.values()) if feasibility else True,
        )
