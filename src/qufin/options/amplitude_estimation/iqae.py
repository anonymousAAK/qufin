"""Iterative Quantum Amplitude Estimation (Grinko-Gacon-Zoufal-Woerner, npj QI 2021).

arXiv:1912.05559. No QPE; quadratic speedup up to a double-log factor.
Uses adaptive Grover iterations with confidence interval refinement.

The algorithm maintains a confidence interval [theta_low, theta_high] for
the angle theta_a, and iteratively applies Q^k with carefully chosen k to
narrow the interval until sin^2(theta_high) - sin^2(theta_low) <= 2*epsilon.

Key insight: after k Grover iterations, the measurement probability is
p_k = sin^2((2k+1)*theta_a). Since sin^2 is periodic, a single measurement
of p_k yields multiple candidate theta values. The algorithm intersects
these candidates with the current interval to refine the estimate.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.stats import beta as beta_dist

from qufin.backends.base import Backend, CircuitResult
from qufin.options.amplitude_estimation.estimation_problem import EstimationProblem
from qufin.utils.results import Result


@dataclass
class IQAEConfig:
    """Configuration for IQAE."""

    epsilon_target: float = 0.01
    alpha: float = 0.05
    shots_per_round: int = 1024
    max_iterations: int = 50
    confint_method: str = "beta"  # "beta" or "clopper_pearson"
    seed: int | None = 42


@dataclass
class IQAEResult(Result):
    """Result from IQAE."""

    estimate: float = 0.0
    confidence_interval: tuple[float, float] = (0.0, 0.0)
    n_oracle_calls: int = 0
    n_rounds: int = 0
    round_details: list[dict] | None = None

    def __post_init__(self):
        if self.round_details is None:
            self.round_details = []


class IterativeAmplitudeEstimation:
    """IQAE per Grinko et al. (1912.05559).

    Key idea: instead of QPE, adaptively choose Grover iteration depths
    k_i and estimate a from the fraction of "good" measurements. Uses
    Clopper-Pearson or beta distribution confidence intervals.

    The critical step is the multi-branch resolution: for depth k,
    the probability sin^2((2k+1)*theta) = p has O(k) solutions in
    [0, pi/2]. We enumerate all candidate theta intervals and intersect
    with the running interval to disambiguate.
    """

    def __init__(
        self,
        problem: EstimationProblem,
        config: IQAEConfig,
        backend: Backend,
    ) -> None:
        self.problem = problem
        self.config = config
        self.backend = backend

    def _build_circuit(self, k: int) -> object:
        """Build circuit with k Grover iterations.

        |psi> = Q^k A |0>

        Then measure the objective qubit(s).
        """
        from qiskit.circuit import QuantumCircuit

        n = self.problem.n_qubits
        qc = QuantumCircuit(n, len(self.problem.objective_qubits))

        # State preparation A|0>
        qc.compose(self.problem.state_preparation, inplace=True)

        # Apply Q^k
        if k > 0:
            grover_op = self.problem.build_grover_operator()
            for _ in range(k):
                qc.compose(grover_op, inplace=True)

        # Measure objective qubits
        for idx, q in enumerate(self.problem.objective_qubits):
            qc.measure(q, idx)

        return qc

    def _count_good(self, result: CircuitResult) -> tuple[int, int]:
        """Count 'good' (objective=1) vs total measurements."""
        n_good = 0
        n_total = 0
        n_obj = len(self.problem.objective_qubits)

        for bitstring, count in result.counts.items():
            n_total += count
            # Check if all objective qubits are |1>
            if len(bitstring) >= n_obj:
                obj_bits = bitstring[-n_obj:]  # rightmost bits
                if all(b == "1" for b in obj_bits):
                    n_good += count

        return n_good, n_total

    def _confidence_interval(
        self, n_good: int, n_total: int, alpha: float
    ) -> tuple[float, float]:
        """Compute confidence interval for the success probability."""
        if n_total == 0:
            return (0.0, 1.0)

        if self.config.confint_method == "beta":
            # Bayesian with uniform prior (Beta(1,1))
            a_low = float(beta_dist.ppf(alpha / 2, n_good + 1, n_total - n_good + 1))
            a_high = float(beta_dist.ppf(1 - alpha / 2, n_good + 1, n_total - n_good + 1))
        else:
            # Clopper-Pearson exact
            if n_good == 0:
                a_low = 0.0
            else:
                a_low = float(beta_dist.ppf(alpha / 2, n_good, n_total - n_good + 1))
            if n_good == n_total:
                a_high = 1.0
            else:
                a_high = float(beta_dist.ppf(1 - alpha / 2, n_good + 1, n_total - n_good))

        return (max(0.0, a_low), min(1.0, a_high))

    def _theta_intervals_from_measurement(
        self, p_low: float, p_high: float, k: int
    ) -> list[tuple[float, float]]:
        """Compute all candidate theta intervals consistent with measured probability.

        Given that sin^2((2k+1)*theta) lies in [p_low, p_high], find all
        theta intervals in [0, pi/2] that are consistent.

        The function sin^2(x) = p has solutions x = arcsin(sqrt(p)) + j*pi
        and x = pi - arcsin(sqrt(p)) + j*pi for integer j >= 0.

        So (2k+1)*theta = arcsin(sqrt(p)) + j*pi
          => theta = (arcsin(sqrt(p)) + j*pi) / (2k+1)
        or (2k+1)*theta = pi - arcsin(sqrt(p)) + j*pi
          => theta = (pi - arcsin(sqrt(p)) + j*pi) / (2k+1)
        """
        factor = 2 * k + 1
        asin_low = np.arcsin(np.sqrt(max(p_low, 0.0)))
        asin_high = np.arcsin(np.sqrt(min(p_high, 1.0)))

        intervals = []
        for j in range(factor):
            # Branch A: theta = (arcsin(sqrt(p)) + j*pi) / factor
            # p in [p_low, p_high] => arcsin(sqrt(p)) in [asin_low, asin_high]
            t_lo = (asin_low + j * np.pi) / factor
            t_hi = (asin_high + j * np.pi) / factor
            if t_lo <= np.pi / 2 and t_hi >= 0:
                intervals.append((max(t_lo, 0.0), min(t_hi, np.pi / 2)))

            # Branch B: theta = (pi - arcsin(sqrt(p)) + j*pi) / factor
            # Note: higher p => smaller arcsin => larger pi - arcsin
            # So the interval is reversed: [pi - asin_high, pi - asin_low]
            t_lo_b = (np.pi - asin_high + j * np.pi) / factor
            t_hi_b = (np.pi - asin_low + j * np.pi) / factor
            if t_lo_b <= np.pi / 2 and t_hi_b >= 0:
                intervals.append((max(t_lo_b, 0.0), min(t_hi_b, np.pi / 2)))

        return intervals

    def _find_next_k(self, theta_low: float, theta_high: float, k_last: int) -> int:
        """Find the next number of Grover iterations.

        Choose K such that the current theta interval maps to at most
        one half-period of sin^2((2K+1)*theta), ensuring unambiguous
        refinement. Per Grinko et al., K_max ≈ pi / (4 * delta_theta).
        """
        delta = theta_high - theta_low
        if delta <= 0:
            return k_last + 1

        # K such that (2K+1) * delta < pi/2 (one half-period)
        k_new = max(1, int(np.pi / (4 * delta) - 0.5))

        # Ensure monotonic progress
        if k_new <= k_last:
            k_new = k_last + 1

        return min(k_new, 1000)

    def estimate(self) -> IQAEResult:
        """Run IQAE main loop until CI half-width <= epsilon_target."""
        start = time.perf_counter()

        epsilon = self.config.epsilon_target
        alpha = self.config.alpha

        # Initialize: theta in [0, pi/2], so a in [0, 1]
        theta_low = 0.0
        theta_high = np.pi / 2

        k = 0  # start with no Grover iterations (k=0 means just A|0>)
        total_oracle_calls = 0
        round_details = []

        for round_idx in range(self.config.max_iterations):
            # Distribute confidence among rounds (Bonferroni)
            alpha_round = alpha / (2 * self.config.max_iterations)

            # Run circuit with k Grover iterations
            circuit = self._build_circuit(k)
            result = self.backend.run(circuit, shots=self.config.shots_per_round)

            n_good, n_total = self._count_good(result)
            total_oracle_calls += (2 * k + 1) * n_total

            # CI for the measured probability p_k = sin^2((2k+1)*theta_a)
            p_low, p_high = self._confidence_interval(n_good, n_total, alpha_round)

            # Enumerate ALL candidate theta intervals from the measurement
            candidates = self._theta_intervals_from_measurement(p_low, p_high, k)

            # Intersect each candidate with the current [theta_low, theta_high]
            # and keep the valid intersection(s)
            best_interval = None
            best_overlap = 0.0
            for c_lo, c_hi in candidates:
                # Intersection
                lo = max(theta_low, c_lo)
                hi = min(theta_high, c_hi)
                if lo < hi:
                    overlap = hi - lo
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_interval = (lo, hi)

            if best_interval is not None:
                theta_low, theta_high = best_interval
            # If no intersection found (shouldn't happen with correct alpha),
            # keep the current interval and try a different k next round.

            # Compute current amplitude estimate
            theta_mid = (theta_low + theta_high) / 2
            a_estimate = np.sin(theta_mid) ** 2
            a_low = np.sin(theta_low) ** 2
            a_high = np.sin(theta_high) ** 2

            round_details.append({
                "round": round_idx,
                "k": k,
                "n_good": n_good,
                "n_total": n_total,
                "estimate": float(a_estimate),
                "ci": (float(a_low), float(a_high)),
            })

            # Check convergence
            if (a_high - a_low) / 2 <= epsilon:
                break

            # Find next k
            k = self._find_next_k(theta_low, theta_high, k)

        theta_final = (theta_low + theta_high) / 2
        estimate = float(np.sin(theta_final) ** 2)
        ci = (float(np.sin(theta_low) ** 2), float(np.sin(theta_high) ** 2))

        wall_time = time.perf_counter() - start

        return IQAEResult(
            value=estimate,
            n_shots=self.config.shots_per_round,
            wall_time_s=wall_time,
            backend_id=self.backend.backend_id,
            seed=self.config.seed,
            estimate=estimate,
            confidence_interval=ci,
            n_oracle_calls=total_oracle_calls,
            n_rounds=len(round_details),
            round_details=round_details,
        )
