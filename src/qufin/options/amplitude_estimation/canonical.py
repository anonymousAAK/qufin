"""Canonical Amplitude Estimation with QPE (Brassard et al., 2002).

Uses Quantum Phase Estimation to estimate the eigenphase of the
Grover operator, which encodes the amplitude a = sin^2(theta_a).

Requires m evaluation qubits; the estimate has precision O(2^{-m}).

References
----------
Brassard, Hoyer, Mosca, Tapp, "Quantum Amplitude Amplification
and Estimation", Contemporary Mathematics 305 (2002).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from qufin.backends.base import Backend
from qufin.options.amplitude_estimation.estimation_problem import EstimationProblem
from qufin.utils.results import Result


@dataclass
class CanonicalQAEConfig:
    """Configuration for canonical (QPE-based) QAE."""

    n_eval_qubits: int = 4
    shots: int = 4096
    seed: int | None = 42


@dataclass
class CanonicalQAEResult(Result):
    """Result from canonical QAE."""

    estimate: float = 0.0
    theta_estimate: float = 0.0
    confidence_interval: tuple[float, float] = (0.0, 0.0)
    n_oracle_calls: int = 0
    most_likely_phase: int = 0


class CanonicalAmplitudeEstimation:
    """Canonical QAE using Quantum Phase Estimation.

    Estimates a = sin^2(theta) where theta is the eigenphase of
    the Grover operator Q.

    Circuit structure:
    - m evaluation qubits initialized with H
    - Controlled-Q^{2^k} for k = 0, ..., m-1
    - Inverse QFT on evaluation register
    - Measure evaluation register -> y
    - Estimate: a = sin^2(pi * y / 2^m)

    The Grover operator Q has eigenvalues e^{±2i*theta_a}, so QPE
    returns y/2^m ≈ theta_a/pi or 1 - theta_a/pi. Both map to the
    same amplitude via a = sin^2(pi * y / 2^m).
    """

    def __init__(
        self,
        problem: EstimationProblem,
        config: CanonicalQAEConfig,
        backend: Backend,
    ) -> None:
        self.problem = problem
        self.config = config
        self.backend = backend

    def _build_circuit(self) -> object:
        """Build the QPE circuit for amplitude estimation."""
        from qiskit.circuit import QuantumCircuit

        m = self.config.n_eval_qubits
        n = self.problem.n_qubits
        total_qubits = m + n

        qc = QuantumCircuit(total_qubits, m)

        # State preparation on system register
        qc.compose(
            self.problem.state_preparation,
            qubits=list(range(m, m + n)),
            inplace=True,
        )

        # Hadamard on evaluation register
        qc.h(range(m))

        # Controlled-Q^{2^k}
        grover_op = self.problem.build_grover_operator()
        for k in range(m):
            power = 2**k
            # Build Q^power
            q_power = grover_op.copy()
            for _ in range(power - 1):
                q_power.compose(grover_op, inplace=True)

            # Controlled version
            c_q = q_power.control(1)
            qc.compose(
                c_q,
                qubits=[k, *list(range(m, m + n))],
                inplace=True,
            )

        # Inverse QFT on evaluation register
        self._inverse_qft(qc, m)

        # Measure evaluation register
        qc.measure(range(m), range(m))

        return qc

    def _inverse_qft(self, qc: object, n: int) -> None:
        """Apply inverse QFT to the first n qubits."""
        for j in range(n // 2):
            qc.swap(j, n - j - 1)

        for j in range(n):
            for k in range(j):
                qc.cp(-np.pi / 2 ** (j - k), k, j)
            qc.h(j)

    def estimate(self) -> CanonicalQAEResult:
        """Run canonical QAE and return the amplitude estimate."""
        start = time.perf_counter()
        m = self.config.n_eval_qubits
        M = 2**m

        circuit = self._build_circuit()
        result = self.backend.run(circuit, shots=self.config.shots)

        # Collect all measurement outcomes with counts.
        # Qiskit returns bitstrings in big-endian (MSB first) for the
        # classical register, matching our IQFT convention where qubit 0
        # maps to classical bit 0 (rightmost in the bitstring).
        # int(bitstring, 2) gives the correct integer y.
        counts_by_y: dict[int, int] = {}
        for bitstring, count in result.counts.items():
            y = int(bitstring, 2)
            counts_by_y[y] = counts_by_y.get(y, 0) + count

        # QPE returns y such that y/M ≈ theta_a/pi (or 1 - theta_a/pi).
        # Both eigenphases give the same amplitude:
        #   a = sin^2(pi * y / M)
        # Find the y with highest count, then pick the branch (y or M-y)
        # that gives a <= 0.5 (by convention, we assume a <= 0.5;
        # if a > 0.5, the user should flip the objective).
        best_y = max(counts_by_y, key=counts_by_y.get)

        # The two candidate amplitudes from y and M-y are the same:
        # sin^2(pi*y/M) = sin^2(pi*(M-y)/M) = sin^2(pi - pi*y/M)
        # So we just compute once.
        theta = np.pi * best_y / M
        amplitude = float(np.sin(theta) ** 2)

        # Confidence interval (based on QPE precision ±1 bin)
        delta_theta = np.pi / M
        theta_low = max(0, theta - delta_theta)
        theta_high = min(np.pi / 2, theta + delta_theta)
        ci = (float(np.sin(theta_low) ** 2), float(np.sin(theta_high) ** 2))

        # Total oracle calls: sum of 2^k for k = 0..m-1 = 2^m - 1
        n_oracle_calls = M - 1

        wall_time = time.perf_counter() - start

        return CanonicalQAEResult(
            value=amplitude,
            n_shots=self.config.shots,
            circuit_depth=circuit.depth() if hasattr(circuit, "depth") else 0,
            wall_time_s=wall_time,
            backend_id=self.backend.backend_id,
            seed=self.config.seed,
            estimate=amplitude,
            theta_estimate=float(theta),
            confidence_interval=ci,
            n_oracle_calls=n_oracle_calls,
            most_likely_phase=best_y,
        )
