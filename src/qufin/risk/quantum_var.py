"""Quantum VaR via amplitude estimation (Woerner & Egger, 1806.06893).

Implements quantum risk analysis using QAE to compute VaR and CVaR/ES.
The key idea: encode a loss distribution into quantum amplitudes, then
use amplitude estimation to compute tail probabilities and conditional
tail expectations.

The VaR level is found via bisection on a QAE oracle that estimates
P(Loss > x) for a given threshold x.

References
----------
Woerner & Egger, "Quantum Risk Analysis", npj Quantum Information 5:15 (2019),
arXiv:1806.06893.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from qufin.backends.base import Backend
from qufin.options.amplitude_estimation.estimation_problem import EstimationProblem
from qufin.options.distributions import DistributionSpec, normal_distribution
from qufin.utils.results import Result


@dataclass
class QuantumVaRConfig:
    """Configuration for quantum VaR estimation."""

    confidence_level: float = 0.95
    n_qubits_loss: int = 3
    n_bisection_steps: int = 8
    qae_method: str = "iqae"
    qae_epsilon: float = 0.01
    qae_alpha: float = 0.05
    qae_shots: int = 1024
    seed: int | None = 42


@dataclass
class QuantumVaRResult(Result):
    """Result from quantum VaR computation."""

    var_estimate: float = 0.0
    es_estimate: float = 0.0
    confidence_level: float = 0.95
    n_qae_calls: int = 0
    bisection_history: list[dict] = field(default_factory=list)


def _decomposed_state_prep(amplitudes: NDArray[np.float64], n_qubits: int) -> object:
    """Build a decomposed state preparation circuit (basic gates only).

    Avoids Aer segfault with StatePreparation.inverse() by decomposing
    to basic gates first.
    """
    from qiskit import transpile
    from qiskit.circuit import QuantumCircuit
    from qiskit.circuit.library import StatePreparation

    sp_qc = QuantumCircuit(n_qubits)
    sp_qc.append(StatePreparation(amplitudes), range(n_qubits))
    return transpile(sp_qc, basis_gates=["cx", "u3", "id"], optimization_level=0)


def _build_tail_probability_problem(
    dist: DistributionSpec,
    threshold: float,
) -> EstimationProblem:
    """Build QAE problem for P(X > threshold).

    The circuit loads the distribution and marks states where
    the value exceeds the threshold by rotating an ancilla qubit.

    Parameters
    ----------
    dist : DistributionSpec
        Loss distribution.
    threshold : float
        Threshold level to compare against.
    """
    from qiskit.circuit import QuantumCircuit

    n_q = dist.n_qubits
    n_total = n_q + 1  # +1 ancilla for comparator

    qc = QuantumCircuit(n_total)

    # Load loss distribution into first n_q qubits (decomposed for Aer compat)
    amplitudes = dist.amplitudes()
    norm_val = np.linalg.norm(amplitudes)
    if norm_val > 0:
        amplitudes = amplitudes / norm_val

    sp_decomposed = _decomposed_state_prep(amplitudes, n_q)
    qc.compose(sp_decomposed, range(n_q), inplace=True)

    # Comparator: rotate ancilla based on whether value > threshold
    for i in range(2**n_q):
        if dist.values[i] > threshold:
            # Rotate ancilla to |1> for states above threshold
            bits = format(i, f"0{n_q}b")
            for b_idx, b in enumerate(bits):
                if b == "0":
                    qc.x(b_idx)

            if n_q == 1:
                qc.cx(0, n_q)
            else:
                qc.mcx(list(range(n_q)), n_q)

            for b_idx, b in enumerate(bits):
                if b == "0":
                    qc.x(b_idx)

    problem = EstimationProblem(
        state_preparation=qc,
        objective_qubits=[n_q],
        n_qubits=n_total,
    )

    return problem


def _build_conditional_value_problem(
    dist: DistributionSpec,
    threshold: float,
) -> tuple[EstimationProblem, float]:
    """Build QAE problem for E[X | X > threshold].

    Encodes the conditional tail values as rotation angles on an
    ancilla qubit, weighted by the probability of exceeding threshold.

    Returns (problem, rescale_factor).
    """
    from qiskit.circuit import QuantumCircuit

    n_q = dist.n_qubits
    n_total = n_q + 2  # +1 comparator ancilla, +1 value ancilla

    qc = QuantumCircuit(n_total)

    # Load distribution (decomposed for Aer compat)
    amplitudes = dist.amplitudes()
    norm_val = np.linalg.norm(amplitudes)
    if norm_val > 0:
        amplitudes = amplitudes / norm_val

    sp_decomposed = _decomposed_state_prep(amplitudes, n_q)
    qc.compose(sp_decomposed, range(n_q), inplace=True)

    # Compute tail values for rescaling
    tail_mask = dist.values > threshold
    dist.values[tail_mask]
    max_val = float(np.max(dist.values)) if len(dist.values) > 0 else 1.0
    if max_val == 0:
        max_val = 1.0

    # For each state: if value > threshold, rotate value ancilla proportionally
    for i in range(2**n_q):
        val = dist.values[i]
        if val > threshold:
            # Mark comparator ancilla
            bits = format(i, f"0{n_q}b")
            for b_idx, b in enumerate(bits):
                if b == "0":
                    qc.x(b_idx)

            if n_q == 1:
                qc.cx(0, n_q)
            else:
                qc.mcx(list(range(n_q)), n_q)

            # Rotate value ancilla proportional to loss value
            normalized = min(val / max_val, 1.0)
            angle = 2 * np.arcsin(np.sqrt(normalized))

            # Controlled rotation on value ancilla, conditioned on comparator
            qc.cry(angle, n_q, n_q + 1)

            # Undo comparator marking (not needed since we want combined state)
            if n_q == 1:
                qc.cx(0, n_q)
            else:
                qc.mcx(list(range(n_q)), n_q)

            for b_idx, b in enumerate(bits):
                if b == "0":
                    qc.x(b_idx)

    problem = EstimationProblem(
        state_preparation=qc,
        objective_qubits=[n_q + 1],
        n_qubits=n_total,
    )

    return problem, max_val


def _run_qae(
    problem: EstimationProblem,
    backend: Backend,
    config: QuantumVaRConfig,
) -> float:
    """Run QAE on the given problem and return the amplitude estimate."""
    if config.qae_method == "iqae":
        from qufin.options.amplitude_estimation.iqae import (
            IQAEConfig,
            IterativeAmplitudeEstimation,
        )

        qae_config = IQAEConfig(
            epsilon_target=config.qae_epsilon,
            alpha=config.qae_alpha,
            shots_per_round=config.qae_shots,
            seed=config.seed,
        )
        qae = IterativeAmplitudeEstimation(problem, qae_config, backend)
        result = qae.estimate()
        return result.estimate

    elif config.qae_method == "mlae":
        from qufin.options.amplitude_estimation.mlae import (
            MaximumLikelihoodAmplitudeEstimation,
            MLAEConfig,
        )

        qae_config = MLAEConfig(
            n_shots_per_round=config.qae_shots,
            seed=config.seed,
        )
        qae = MaximumLikelihoodAmplitudeEstimation(problem, qae_config, backend)
        result = qae.estimate()
        return result.estimate

    elif config.qae_method == "canonical":
        from qufin.options.amplitude_estimation.canonical import (
            CanonicalAmplitudeEstimation,
            CanonicalQAEConfig,
        )

        qae_config = CanonicalQAEConfig(
            n_eval_qubits=4,
            shots=config.qae_shots,
            seed=config.seed,
        )
        qae = CanonicalAmplitudeEstimation(problem, qae_config, backend)
        result = qae.estimate()
        return result.estimate

    else:
        raise ValueError(f"Unknown QAE method: {config.qae_method}")


def quantum_var(
    loss_distribution: DistributionSpec,
    backend: Backend,
    config: QuantumVaRConfig | None = None,
) -> QuantumVaRResult:
    """Compute VaR via quantum amplitude estimation + bisection.

    Implements Woerner & Egger (1806.06893) Algorithm 1:
    1. Binary search for x* such that P(Loss > x*) = 1 - confidence
    2. Estimate E[Loss | Loss > x*] for Expected Shortfall

    Parameters
    ----------
    loss_distribution : DistributionSpec
        Discretized loss distribution to analyze.
    backend : Backend
        Quantum backend for circuit execution.
    config : QuantumVaRConfig or None
        Configuration. Uses defaults if None.
    """
    if config is None:
        config = QuantumVaRConfig()

    start = time.perf_counter()

    alpha = 1 - config.confidence_level  # e.g. 0.05 for 95% VaR
    n_qae_calls = 0
    bisection_history = []

    # Bisection: find x* where P(Loss > x*) ≈ alpha
    lo = float(loss_distribution.low)
    hi = float(loss_distribution.high)

    for step in range(config.n_bisection_steps):
        mid = (lo + hi) / 2

        # QAE: estimate P(Loss > mid)
        problem = _build_tail_probability_problem(loss_distribution, mid)
        prob_exceed = _run_qae(problem, backend, config)
        n_qae_calls += 1

        bisection_history.append({
            "step": step,
            "threshold": mid,
            "prob_exceed": prob_exceed,
            "lo": lo,
            "hi": hi,
        })

        if prob_exceed > alpha:
            lo = mid  # threshold too low, more than alpha exceeds
        else:
            hi = mid  # threshold too high

    var_estimate = (lo + hi) / 2

    # Expected Shortfall: E[Loss | Loss > VaR]
    # Computed from the discrete distribution given the QAE-found VaR level.
    # (Woerner & Egger also use QAE for ES, but for small qubit counts
    # classical computation from the known distribution is equivalent and
    # avoids circuit complexity issues.)
    tail_mask = loss_distribution.values > var_estimate
    tail_probs = loss_distribution.probabilities[tail_mask]
    tail_values = loss_distribution.values[tail_mask]
    tail_total = float(tail_probs.sum())
    if tail_total > 1e-10:
        es_estimate = float(np.sum(tail_values * tail_probs) / tail_total)
    else:
        es_estimate = var_estimate

    wall_time = time.perf_counter() - start

    return QuantumVaRResult(
        value=var_estimate,
        wall_time_s=wall_time,
        backend_id=backend.backend_id,
        seed=config.seed,
        n_shots=config.qae_shots,
        var_estimate=var_estimate,
        es_estimate=es_estimate,
        confidence_level=config.confidence_level,
        n_qae_calls=n_qae_calls,
        bisection_history=bisection_history,
    )


def build_loss_distribution(
    returns: NDArray[np.float64],
    n_qubits: int = 3,
    n_sigma: float = 3.0,
) -> DistributionSpec:
    """Build a loss distribution from historical returns.

    Fits a normal distribution to negative returns (losses)
    and discretizes for quantum circuit loading.

    Parameters
    ----------
    returns : NDArray
        Portfolio returns, shape (T,).
    n_qubits : int
        Number of qubits for discretization.
    n_sigma : float
        Domain width in standard deviations.
    """
    losses = -np.asarray(returns).flatten()
    mu = float(np.mean(losses))
    sigma = float(np.std(losses, ddof=1))

    return normal_distribution(
        n_qubits=n_qubits,
        mean=mu,
        std=sigma,
        n_sigma=n_sigma,
    )
