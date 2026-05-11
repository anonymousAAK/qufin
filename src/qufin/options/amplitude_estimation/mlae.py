"""Maximum Likelihood Amplitude Estimation (Suzuki et al., 2020).

QPE-free amplitude estimation using maximum likelihood inference
over multiple rounds of Grover iterations with different depths.

References
----------
Suzuki, Uno, Raymond, Tanaka, Onodera, Yamamoto,
"Amplitude estimation without phase estimation", QIP 19:75 (2020).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

from qufin.backends.base import Backend, CircuitResult
from qufin.options.amplitude_estimation.estimation_problem import EstimationProblem
from qufin.utils.results import Result


@dataclass
class MLAEConfig:
    """Configuration for MLAE."""

    evaluation_schedule: list[int] | None = None
    n_shots_per_round: int = 1024
    seed: int | None = 42


@dataclass
class MLAEResult(Result):
    """Result from MLAE."""

    estimate: float = 0.0
    confidence_interval: tuple[float, float] = (0.0, 0.0)
    n_oracle_calls: int = 0
    n_rounds: int = 0
    log_likelihood: float = 0.0


class MaximumLikelihoodAmplitudeEstimation:
    """MLAE per Suzuki et al. (2020).

    Runs circuits with different numbers of Grover iterations
    m_0, m_1, ..., m_L, records the fraction of "good" outcomes,
    then finds the angle theta that maximizes the log-likelihood:

    L(theta) = sum_k [ h_k * log(sin^2((2m_k+1)*theta))
                      + (N_k - h_k) * log(cos^2((2m_k+1)*theta)) ]

    where h_k = number of "good" outcomes in round k, N_k = shots.
    """

    def __init__(
        self,
        problem: EstimationProblem,
        config: MLAEConfig,
        backend: Backend,
    ) -> None:
        self.problem = problem
        self.config = config
        self.backend = backend

    def _build_circuit(self, k: int) -> object:
        """Build circuit: A |0> then k Grover iterations, measure objective."""
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
        """Count good outcomes from circuit result."""
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

    def _log_likelihood(
        self,
        theta: float,
        data: list[tuple[int, int, int]],
    ) -> float:
        """Compute log-likelihood for a given theta.

        Parameters
        ----------
        theta : float
            Candidate angle.
        data : list of (m_k, h_k, N_k)
            Grover depth, good count, total count per round.
        """
        ll = 0.0
        eps = 1e-15
        for m_k, h_k, n_k in data:
            p = np.sin((2 * m_k + 1) * theta) ** 2
            p = np.clip(p, eps, 1 - eps)
            ll += h_k * np.log(p) + (n_k - h_k) * np.log(1 - p)
        return ll

    def estimate(self) -> MLAEResult:
        """Run MLAE and return the amplitude estimate."""
        start = time.perf_counter()

        # Default evaluation schedule: powers of 2
        if self.config.evaluation_schedule is not None:
            schedule = self.config.evaluation_schedule
        else:
            schedule = [0, 1, 2, 4, 8, 16]

        data: list[tuple[int, int, int]] = []
        total_oracle_calls = 0

        for m_k in schedule:
            circuit = self._build_circuit(m_k)
            result = self.backend.run(circuit, shots=self.config.n_shots_per_round)
            h_k, n_k = self._count_good(result)
            data.append((m_k, h_k, n_k))
            total_oracle_calls += (2 * m_k + 1) * n_k

        # Maximize log-likelihood over theta in (0, pi/2)
        opt = minimize_scalar(
            lambda t: -self._log_likelihood(t, data),
            bounds=(1e-6, np.pi / 2 - 1e-6),
            method="bounded",
        )
        theta_opt = opt.x
        amplitude = float(np.sin(theta_opt) ** 2)
        max_ll = float(-opt.fun)

        # Fisher information-based confidence interval
        # Approximate Cramér-Rao bound
        fisher = 0.0
        eps = 1e-8
        for m_k, _h_k, n_k in data:
            factor = 2 * m_k + 1
            p = np.sin(factor * theta_opt) ** 2
            p = np.clip(p, 1e-15, 1 - 1e-15)
            dp = factor * np.sin(2 * factor * theta_opt)
            fisher += n_k * dp**2 / (p * (1 - p) + eps)

        if fisher > 0:
            std_theta = 1.0 / np.sqrt(fisher)
            theta_low = max(0, theta_opt - 1.96 * std_theta)
            theta_high = min(np.pi / 2, theta_opt + 1.96 * std_theta)
            ci = (float(np.sin(theta_low) ** 2), float(np.sin(theta_high) ** 2))
        else:
            ci = (0.0, 1.0)

        wall_time = time.perf_counter() - start

        return MLAEResult(
            value=amplitude,
            n_shots=self.config.n_shots_per_round,
            wall_time_s=wall_time,
            backend_id=self.backend.backend_id,
            seed=self.config.seed,
            estimate=amplitude,
            confidence_interval=ci,
            n_oracle_calls=total_oracle_calls,
            n_rounds=len(schedule),
            log_likelihood=max_ll,
        )
