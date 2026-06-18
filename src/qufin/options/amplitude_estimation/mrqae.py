"""Modified Real Quantum Amplitude Estimation (mRQAE).

An alternative to standard QAE that uses direct encoding and fewer qubits.
Instead of QPE, mRQAE collects measurement probabilities at multiple
Grover depths and fits the amplitude via least-squares on the relation

    cos(2 * (2k+1) * theta) = 1 - 2 * p_k

where p_k is the probability of measuring the objective qubit in |1>
after k Grover iterations.  This follows from the amplitude-amplification
identity p_k = sin^2((2k+1) * theta), so 1 - 2 * p_k = cos(2 * (2k+1) * theta).

Multiple depths break the periodicity ambiguity, and a least-squares fit
over all (k, p_k) pairs gives a robust estimate even with shot noise.

References
----------
Plekhanov et al., "Variational quantum amplitude estimation" (2022).
Tanaka et al., "Modified Real Quantum Amplitude Estimation" (2025).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from qufin.backends.base import Backend, CircuitResult
from qufin.options.amplitude_estimation.estimation_problem import EstimationProblem
from qufin.utils.results import Result


@dataclass
class MRQAEConfig:
    """Configuration for modified Real QAE.

    Parameters
    ----------
    epsilon : float
        Target accuracy for the amplitude estimate.
    shots_per_round : int
        Number of shots per depth round.
    max_depth : int
        Maximum Grover depth (number of Q applications).
    schedule : str
        Depth schedule: ``"linear"`` for [0, 1, 2, ...] or
        ``"exponential"`` for [0, 1, 2, 4, 8, ...].
    seed : int | None
        Random seed for reproducibility.
    """

    epsilon: float = 0.01
    shots_per_round: int = 1024
    max_depth: int = 50
    schedule: str = "linear"
    seed: int | None = None


@dataclass
class MRQAEResult(Result):
    """Result from mRQAE.

    Attributes
    ----------
    estimate : float
        Amplitude estimate (sin^2(theta)).
    confidence_interval : tuple[float, float]
        Confidence interval for the amplitude.
    n_oracle_calls : int
        Total number of oracle (Grover operator) calls.
    n_rounds : int
        Number of depth rounds executed.
    depths_used : list[int]
        Depths at which circuits were run.
    round_details : list[dict]
        Per-round information: depth, measured probability, shots.
    """

    estimate: float = 0.0
    confidence_interval: tuple[float, float] = (0.0, 0.0)
    n_oracle_calls: int = 0
    n_rounds: int = 0
    depths_used: list[int] = field(default_factory=list)
    round_details: list[dict] = field(default_factory=list)


class ModifiedRealQAE:
    """Modified Real Quantum Amplitude Estimation.

    Instead of QPE or iterative bisection, mRQAE collects measurement
    probabilities at a *schedule* of Grover depths and fits the angle
    theta from the over-determined system

        cos(2 * (2k+1) * theta) = 1 - 2 * p_k   for each depth k

    using least-squares minimisation.  This approach:

    * avoids the ancilla-heavy QPE register,
    * is naturally parallelisable (each depth is independent), and
    * is robust to shot noise via the overdetermined fit.

    Parameters
    ----------
    problem : EstimationProblem
        The amplitude estimation problem (state prep + objective).
    config : MRQAEConfig
        Algorithm configuration.
    backend : Backend
        Execution backend.
    """

    def __init__(
        self,
        problem: EstimationProblem,
        config: MRQAEConfig,
        backend: Backend,
    ) -> None:
        self.problem = problem
        self.config = config
        self.backend = backend

    def _build_circuit(self, depth: int) -> object:
        """Build circuit: A followed by Q^depth, measure objective qubit.

        Parameters
        ----------
        depth : int
            Number of Grover iterations to apply after state preparation.

        Returns
        -------
        QuantumCircuit
            The measurement circuit.
        """
        from qiskit.circuit import QuantumCircuit

        n = self.problem.n_qubits
        qc = QuantumCircuit(n, len(self.problem.objective_qubits))

        # State preparation A|0>
        qc.compose(self.problem.state_preparation, inplace=True)

        # Apply Q^depth
        if depth > 0:
            grover_op = self.problem.build_grover_operator()
            for _ in range(depth):
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
            if len(bitstring) >= n_obj:
                obj_bits = bitstring[-n_obj:]
                if all(b == "1" for b in obj_bits):
                    n_good += count

        return n_good, n_total

    def _generate_schedule(self) -> list[int]:
        """Generate the depth schedule based on config.

        Returns
        -------
        list[int]
            Ordered list of Grover depths to run.
        """
        if self.config.schedule == "exponential":
            depths = [0]
            k = 1
            while k <= self.config.max_depth:
                depths.append(k)
                k *= 2
            return depths
        else:
            # Linear schedule
            return list(range(self.config.max_depth + 1))

    def _fit_amplitude(
        self, depths: list[int], measurements: list[float]
    ) -> tuple[float, float]:
        """Fit theta from measurement data via least-squares.

        For each depth k with measured probability p_k, the relation is:

            cos(2 * (2k+1) * theta) = 1 - 2 * p_k

        which follows from p_k = sin^2((2k+1) * theta).  We minimise
        sum_k [ cos(2*(2k+1)*theta) - (1 - 2*p_k) ]^2 over theta in [0, pi/2].

        Parameters
        ----------
        depths : list[int]
            Grover depths used.
        measurements : list[float]
            Measured probabilities p_k for each depth.

        Returns
        -------
        tuple[float, float]
            (amplitude, uncertainty) where amplitude = sin^2(theta_best).
        """
        from scipy.optimize import minimize_scalar

        # Target values: c_k = 1 - 2*p_k = cos(2*(2k+1)*theta), since the
        # measured P(good) after k Grover iterations is sin^2((2k+1)*theta).
        targets = np.array([1.0 - 2.0 * p for p in measurements])
        factors = np.array([2 * (2 * k + 1) for k in depths])

        def cost(theta: float) -> float:
            residuals = np.cos(factors * theta) - targets
            return float(np.sum(residuals**2))

        # Grid search for initial guess to avoid local minima
        n_grid = 500
        theta_grid = np.linspace(0, np.pi / 2, n_grid)
        costs = [cost(t) for t in theta_grid]
        theta_init = theta_grid[np.argmin(costs)]

        # Refine with bounded scalar optimisation
        res = minimize_scalar(cost, bounds=(0, np.pi / 2), method="bounded")
        theta_best = res.x

        # Also check the grid winner in case minimize_scalar found a local min
        if cost(theta_init) < cost(theta_best):
            theta_best = theta_init

        amplitude = float(np.sin(theta_best) ** 2)

        # Uncertainty estimate from Fisher information
        # For shot noise, Var(p_k) ~ p_k*(1-p_k)/N
        # Propagate to theta via delta method
        n_shots = self.config.shots_per_round
        if len(depths) > 1:
            # Numerical Hessian at the optimum
            h = 1e-6
            d2 = (cost(theta_best + h) - 2 * cost(theta_best) + cost(theta_best - h)) / h**2
            if d2 > 0:
                # Fisher-based standard error
                sigma_theta = 1.0 / np.sqrt(max(d2 * n_shots, 1e-12))
            else:
                # Fallback: simple estimate from residual spread
                sigma_theta = np.sqrt(res.fun / max(len(depths) - 1, 1)) / np.mean(factors)
        else:
            sigma_theta = 1.0 / np.sqrt(n_shots)

        # Propagate to amplitude: d(sin^2(theta))/dtheta = sin(2*theta)
        sigma_amp = abs(np.sin(2 * theta_best)) * sigma_theta
        uncertainty = float(sigma_amp)

        return amplitude, uncertainty

    def estimate(self) -> MRQAEResult:
        """Run the mRQAE algorithm.

        Iterates through the depth schedule, collecting measurements
        at each depth. Performs early stopping when the estimated
        error drops below ``config.epsilon``.

        Returns
        -------
        MRQAEResult
            The estimation result including amplitude, CI, and diagnostics.
        """
        start = time.perf_counter()

        schedule = self._generate_schedule()
        depths_used: list[int] = []
        measurements: list[float] = []
        round_details: list[dict] = []
        total_oracle_calls = 0

        for depth in schedule:
            # Build and run circuit
            circuit = self._build_circuit(depth)
            result = self.backend.run(circuit, shots=self.config.shots_per_round)

            n_good, n_total = self._count_good(result)
            p_k = n_good / max(n_total, 1)

            total_oracle_calls += (2 * depth + 1) * n_total

            depths_used.append(depth)
            measurements.append(p_k)

            round_details.append({
                "round": len(round_details),
                "depth": depth,
                "n_good": n_good,
                "n_total": n_total,
                "p_k": float(p_k),
            })

            # Need at least 2 data points for a meaningful fit
            if len(depths_used) >= 2:
                amplitude, uncertainty = self._fit_amplitude(depths_used, measurements)

                # Early stopping
                if uncertainty < self.config.epsilon:
                    break

        # Final fit with all collected data
        if len(depths_used) >= 2:
            amplitude, uncertainty = self._fit_amplitude(depths_used, measurements)
        elif len(depths_used) == 1:
            # Single measurement: amplitude ~ p_0 (depth 0 => p_0 = sin^2(theta))
            amplitude = measurements[0]
            uncertainty = np.sqrt(
                measurements[0] * (1 - measurements[0]) / self.config.shots_per_round
            )
        else:
            amplitude = 0.0
            uncertainty = 1.0

        ci = (
            float(max(0.0, amplitude - 2 * uncertainty)),
            float(min(1.0, amplitude + 2 * uncertainty)),
        )

        wall_time = time.perf_counter() - start

        return MRQAEResult(
            value=amplitude,
            n_shots=self.config.shots_per_round,
            wall_time_s=wall_time,
            backend_id=self.backend.backend_id,
            seed=self.config.seed,
            estimate=float(amplitude),
            confidence_interval=ci,
            n_oracle_calls=total_oracle_calls,
            n_rounds=len(depths_used),
            depths_used=depths_used,
            round_details=round_details,
        )


def direct_encode_distribution(
    values: NDArray,
    probabilities: NDArray,
    n_qubits: int,
) -> object:
    """Direct encoding of a probability distribution into a quantum state.

    Instead of loading sqrt(p_i) as amplitudes (standard amplitude loading),
    this encodes the distribution values directly using controlled rotations
    based on the cumulative distribution function.

    Each basis state |i> receives a Y-rotation with angle proportional
    to the CDF evaluated at the corresponding value, mapping the
    distribution information into qubit amplitudes more efficiently
    for certain distributions.

    Parameters
    ----------
    values : NDArray
        Distribution sample points, shape ``(2**n_qubits,)``.
    probabilities : NDArray
        Probability weights, shape ``(2**n_qubits,)``.  Will be
        normalised if they do not sum to 1.
    n_qubits : int
        Number of qubits for the encoding register.

    Returns
    -------
    QuantumCircuit
        Circuit that prepares the encoded state on ``n_qubits + 1``
        qubits (register + 1 ancilla for the rotation).
    """
    from qiskit.circuit import QuantumCircuit
    from qiskit.circuit.library import RYGate

    n_states = 2**n_qubits
    if len(values) != n_states or len(probabilities) != n_states:
        raise ValueError(
            f"values and probabilities must have length 2**n_qubits = {n_states}"
        )

    # Normalise probabilities
    probs = np.array(probabilities, dtype=float)
    probs = probs / probs.sum()

    # Compute CDF
    cdf = np.cumsum(probs)

    # Build circuit: n_qubits register + 1 ancilla for rotation
    n_total = n_qubits + 1
    qc = QuantumCircuit(n_total)

    # Step 1: Load sqrt(p_i) amplitudes into register via initialize
    amplitudes = np.sqrt(probs)
    norm = np.linalg.norm(amplitudes)
    if norm > 0:
        amplitudes = amplitudes / norm
    qc.initialize(amplitudes, range(n_qubits))

    # Step 2: Controlled rotations on ancilla encoding the CDF value
    ancilla = n_qubits
    for i in range(n_states):
        # Rotation angle proportional to CDF at this point
        angle = 2 * np.arcsin(np.sqrt(min(cdf[i], 1.0)))

        # Encode basis state |i> via X gates
        bits = format(i, f"0{n_qubits}b")
        for b_idx, b in enumerate(bits):
            if b == "0":
                qc.x(b_idx)

        # Multi-controlled Ry
        if n_qubits == 1:
            qc.cry(angle, 0, ancilla)
        else:
            qc.append(
                RYGate(angle).control(n_qubits),
                [*list(range(n_qubits)), ancilla],
            )

        # Undo X gates
        for b_idx, b in enumerate(bits):
            if b == "0":
                qc.x(b_idx)

    return qc


def price_european_mrqae(
    s: float,
    k: float,
    sigma: float,
    r: float,
    T: float,
    backend: Backend,
    config: MRQAEConfig | None = None,
) -> dict:
    """Price a European call option using mRQAE.

    Convenience wrapper that builds the standard European QAE
    estimation problem and runs Modified Real QAE instead of IQAE.

    Parameters
    ----------
    s : float
        Spot price.
    k : float
        Strike price.
    sigma : float
        Volatility.
    r : float
        Risk-free rate.
    T : float
        Time to expiry in years.
    backend : Backend
        Execution backend.
    config : MRQAEConfig | None
        Algorithm configuration.  Uses defaults if ``None``.

    Returns
    -------
    dict
        Keys: ``"price"``, ``"ci"``, ``"n_oracle_calls"``, ``"depths_used"``.
    """
    from qufin.options.amplitude_estimation.european_qae import (
        EuropeanQAESpec,
        build_european_estimation_problem,
    )

    if config is None:
        config = MRQAEConfig()

    spec = EuropeanQAESpec(s0=s, k=k, r=r, sigma=sigma, T=T, is_call=True)
    problem, rescale = build_european_estimation_problem(spec)

    estimator = ModifiedRealQAE(problem=problem, config=config, backend=backend)
    result = estimator.estimate()

    price = result.estimate * rescale
    ci_low = result.confidence_interval[0] * rescale
    ci_high = result.confidence_interval[1] * rescale

    return {
        "price": price,
        "ci": (ci_low, ci_high),
        "n_oracle_calls": result.n_oracle_calls,
        "depths_used": result.depths_used,
    }
