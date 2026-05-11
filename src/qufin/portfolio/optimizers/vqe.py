"""VQE portfolio optimizer with CVaR objective.

Uses a TwoLocal hardware-efficient ansatz with CVaR expectation
as the objective function for combinatorial optimization.

References
----------
Barkoutsos et al., Quantum 4, 256 (2020) — CVaR-VQE.
Kandala et al., Nature 549, 242 (2017) — hardware-efficient ansatz.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from qufin.backends.base import Backend, CircuitResult
from qufin.portfolio.qubo import PortfolioQUBO
from qufin.utils.results import Result


@dataclass
class VQEConfig:
    """Configuration for VQE portfolio optimizer."""

    reps: int = 3
    entanglement: Literal["linear", "circular", "full"] = "linear"
    rotation_blocks: list[str] = field(default_factory=lambda: ["ry", "rz"])
    entanglement_blocks: str = "cx"
    optimizer: str = "COBYLA"
    maxiter: int = 300
    shots: int = 8192
    seed: int | None = 42
    cvar_alpha: float = 0.5  # <1.0 = CVaR (tail optimization)
    initial_params: NDArray[np.float64] | None = None


@dataclass
class VQEResult(Result):
    """Result from VQE portfolio optimization."""

    best_bitstring: str = ""
    best_objective: float = float("inf")
    weights: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    optimal_params: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    history: list[float] = field(default_factory=list)
    feasible: bool = False


class VQEPortfolio:
    """VQE solver for the cardinality-constrained Markowitz QUBO.

    Uses a TwoLocal hardware-efficient ansatz (Kandala et al. 2017)
    with CVaR objective (Barkoutsos et al. 2020).
    """

    def __init__(self, qubo: PortfolioQUBO, config: VQEConfig, backend: Backend) -> None:
        self.qubo = qubo
        self.config = config
        self.backend = backend
        self._Q = qubo.build_matrix()
        self._history: list[float] = []

    def _n_params(self) -> int:
        """Number of variational parameters in the TwoLocal ansatz."""
        n = self.qubo.n_qubits
        n_rot = len(self.config.rotation_blocks)
        # TwoLocal: (reps + 1) rotation layers, each with n_rot * n params
        return n_rot * n * (self.config.reps + 1)

    def _build_circuit(self, params: NDArray[np.float64]) -> object:
        """Build the TwoLocal hardware-efficient ansatz circuit."""
        from qiskit.circuit import QuantumCircuit

        n = self.qubo.n_qubits
        reps = self.config.reps
        rot_blocks = self.config.rotation_blocks
        qc = QuantumCircuit(n, n)

        param_idx = 0

        for layer in range(reps + 1):
            # Rotation layer
            for gate_name in rot_blocks:
                for qubit in range(n):
                    gate_fn = getattr(qc, gate_name)
                    gate_fn(params[param_idx], qubit)
                    param_idx += 1

            # Entanglement layer (skip after last rotation)
            if layer < reps:
                if self.config.entanglement == "linear":
                    for i in range(n - 1):
                        qc.cx(i, i + 1)
                elif self.config.entanglement == "circular":
                    for i in range(n - 1):
                        qc.cx(i, i + 1)
                    if n > 2:
                        qc.cx(n - 1, 0)
                elif self.config.entanglement == "full":
                    for i in range(n):
                        for j in range(i + 1, n):
                            qc.cx(i, j)

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
        """VQE objective function for the classical optimizer."""
        circuit = self._build_circuit(params)
        result = self.backend.run(circuit, shots=self.config.shots)
        val = self._evaluate_counts(result)
        self._history.append(val)
        return val

    def run(self) -> VQEResult:
        """Optimize VQE parameters and return the best portfolio."""
        start = time.perf_counter()
        rng = np.random.default_rng(self.config.seed)

        n_params = self._n_params()
        if self.config.initial_params is not None:
            x0 = self.config.initial_params
        else:
            x0 = rng.uniform(0, 2 * np.pi, n_params)

        self._history = []
        opt_result = minimize(
            self._objective,
            x0,
            method=self.config.optimizer,
            options={"maxiter": self.config.maxiter},
        )

        opt_params = opt_result.x

        # Final evaluation to get best bitstring
        circuit = self._build_circuit(opt_params)
        final_result = self.backend.run(circuit, shots=self.config.shots)
        best_bs = final_result.most_frequent
        best_obj = self.qubo.evaluate(best_bs)

        # Decode weights and check feasibility
        weights = self.qubo.decode_weights(best_bs)
        feasibility = self.qubo.feasibility_check(best_bs)

        wall_time = time.perf_counter() - start

        return VQEResult(
            value=best_obj,
            n_shots=self.config.shots,
            wall_time_s=wall_time,
            backend_id=self.backend.backend_id,
            seed=self.config.seed,
            best_bitstring=best_bs,
            best_objective=best_obj,
            weights=weights,
            optimal_params=opt_params,
            history=self._history,
            feasible=all(feasibility.values()) if feasibility else True,
        )
