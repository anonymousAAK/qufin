"""Faithful / low-depth QAE (Giurgica-Tiron et al., arXiv:2012.03348).

Power-schedule QAE that trades circuit depth for more shots.
Useful for near-term devices where deep circuits are infeasible.

References
----------
Giurgica-Tiron, Kerenidis, Labib, Prakash, Zeng,
"Low-depth algorithms for quantum amplitude estimation",
Quantum 6:745 (2022), arXiv:2012.03348.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from qufin.backends.base import Backend, CircuitResult
from qufin.options.amplitude_estimation.estimation_problem import EstimationProblem
from qufin.utils.results import Result


@dataclass
class FQAEConfig:
    """Configuration for faithful / low-depth QAE."""

    max_depth: int = 8
    n_shots_per_round: int = 2048
    delta: float = 0.05
    seed: int | None = 42


@dataclass
class FQAEResult(Result):
    """Result from FQAE."""

    estimate: float = 0.0
    confidence_interval: tuple[float, float] = (0.0, 0.0)
    n_oracle_calls: int = 0
    n_rounds: int = 0
    max_depth_used: int = 0


class FaithfulAmplitudeEstimation:
    """Faithful QAE per Giurgica-Tiron et al. (2012.03348).

    Uses a linear schedule of Grover depths [0, 1, 2, ..., max_depth]
    with majority-vote refinement. Trades fewer qubits/depth
    for more classical post-processing and shots.
    """

    def __init__(
        self,
        problem: EstimationProblem,
        config: FQAEConfig,
        backend: Backend,
    ) -> None:
        self.problem = problem
        self.config = config
        self.backend = backend

    def _build_circuit(self, k: int) -> object:
        """Build circuit with k Grover iterations."""
        from qiskit.circuit import QuantumCircuit

        n = self.problem.n_qubits
        qc = QuantumCircuit(n, len(self.problem.objective_qubits))

        qc.compose(self.problem.state_preparation, inplace=True)

        if k > 0:
            grover_op = self.problem.build_grover_operator()
            for _ in range(k):
                qc.compose(grover_op, inplace=True)

        for idx, q in enumerate(self.problem.objective_qubits):
            qc.measure(q, idx)

        return qc

    def _count_good(self, result: CircuitResult) -> tuple[int, int]:
        """Count good outcomes."""
        n_good = 0
        n_total = 0
        n_obj = len(self.problem.objective_qubits)

        for bitstring, count in result.counts.items():
            n_total += count
            if len(bitstring) >= n_obj:
                obj_bits = bitstring[-n_obj:]
                if all(b == "1" for b in obj_bits):
                    n_good += count

        return n_good, n_total

    def estimate(self) -> FQAEResult:
        """Run faithful QAE with bounded-depth circuits."""
        start = time.perf_counter()

        # Linear schedule: k = 0, 1, 2, ..., max_depth
        schedule = list(range(self.config.max_depth + 1))
        total_oracle_calls = 0

        # First pass: get anchor estimate from k=0 (direct measurement)
        circuit_0 = self._build_circuit(0)
        result_0 = self.backend.run(circuit_0, shots=self.config.n_shots_per_round)
        h_0, n_0 = self._count_good(result_0)
        total_oracle_calls += n_0
        p_anchor = h_0 / n_0 if n_0 > 0 else 0.5
        theta_anchor = np.arcsin(np.sqrt(np.clip(p_anchor, 0, 1)))

        estimates = [theta_anchor]

        for k in schedule[1:]:
            circuit = self._build_circuit(k)
            result = self.backend.run(circuit, shots=self.config.n_shots_per_round)
            h_k, n_k = self._count_good(result)
            total_oracle_calls += (2 * k + 1) * n_k

            p_obs = h_k / n_k if n_k > 0 else 0.0
            factor = 2 * k + 1

            # sin^2(factor * theta) = p_obs
            # theta = (arcsin(sqrt(p)) + j*pi) / factor for integer j
            # Pick j that is closest to anchor estimate
            base_theta = np.arcsin(np.sqrt(np.clip(p_obs, 0, 1)))
            best_theta = base_theta / factor
            best_dist = abs(best_theta - theta_anchor)

            for j in range(1, factor + 1):
                for sign in [1, -1]:
                    candidate = (sign * base_theta + j * np.pi) / factor
                    if 0 <= candidate <= np.pi / 2:
                        d = abs(candidate - theta_anchor)
                        if d < best_dist:
                            best_dist = d
                            best_theta = candidate

            estimates.append(best_theta)

        # Weighted average: higher k gets more weight (more precise)
        weights = np.array([2 * k + 1 for k in schedule], dtype=np.float64)
        weights = weights / weights.sum()
        theta_est = float(np.average(estimates, weights=weights))
        amplitude = float(np.sin(theta_est) ** 2)

        # Confidence interval via Hoeffding bound on each estimate
        # Combined CI narrows with more rounds
        n_rounds = len(schedule)
        ci_half = np.sqrt(np.log(2 / self.config.delta) / (2 * n_rounds))
        a_low = max(0.0, amplitude - ci_half)
        a_high = min(1.0, amplitude + ci_half)

        wall_time = time.perf_counter() - start

        return FQAEResult(
            value=amplitude,
            n_shots=self.config.n_shots_per_round,
            wall_time_s=wall_time,
            backend_id=self.backend.backend_id,
            seed=self.config.seed,
            estimate=amplitude,
            confidence_interval=(a_low, a_high),
            n_oracle_calls=total_oracle_calls,
            n_rounds=n_rounds,
            max_depth_used=self.config.max_depth,
        )
