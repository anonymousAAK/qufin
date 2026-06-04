"""European option pricing via Quantum Amplitude Estimation.

Combines distribution loading (log-normal price distribution)
with a payoff comparator to estimate E[max(S_T - K, 0)] using QAE.

References
----------
Stamatopoulos et al., Quantum 4:291 (2020), arXiv:1905.02666.
Woerner & Egger, npj Quantum Information 5:15 (2019), arXiv:1806.06893.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qufin.options.amplitude_estimation.estimation_problem import EstimationProblem
from qufin.options.distributions import log_normal_distribution, state_prep_circuit


@dataclass
class EuropeanQAESpec:
    """European option specification for QAE pricing.

    Parameters
    ----------
    s0 : float
        Spot price.
    k : float
        Strike price.
    r : float
        Risk-free rate.
    sigma : float
        Volatility.
    T : float
        Time to expiry.
    is_call : bool
        True for call, False for put.
    n_qubits : int
        Number of qubits for price discretization.
    """

    s0: float = 100.0
    k: float = 100.0
    r: float = 0.05
    sigma: float = 0.2
    T: float = 1.0
    is_call: bool = True
    n_qubits: int = 3


def build_european_estimation_problem(
    spec: EuropeanQAESpec,
) -> tuple[EstimationProblem, float]:
    """Build the QAE estimation problem for European option pricing.

    Constructs:
    1. Distribution loading circuit: prepares log-normal S_T distribution
    2. Payoff comparator: marks states where payoff > 0
    3. Payoff rotation: encodes payoff value into an ancilla amplitude

    Returns
    -------
    tuple of (EstimationProblem, rescale_factor)
        The estimation problem and a factor to rescale the QAE estimate
        back to the option price.
    """
    from qiskit.circuit import QuantumCircuit

    # Step 1: Load log-normal distribution
    dist = log_normal_distribution(
        n_qubits=spec.n_qubits,
        s0=spec.s0,
        mu=spec.r,  # risk-neutral drift
        sigma=spec.sigma,
        T=spec.T,
    )

    n_price = spec.n_qubits
    n_total = n_price + 1  # +1 ancilla for payoff

    # Step 2: Build the full state preparation circuit
    # A|0> = sum_i sqrt(p_i) |i> |f(S_i)>
    # where f(S_i) encodes whether payoff > 0 and its magnitude
    qc = QuantumCircuit(n_total)

    # Load distribution into price register using an invertible state
    # preparation so the Grover operator (which needs A^dagger) can be built.
    qc.compose(state_prep_circuit(dist.amplitudes(), n_price), range(n_price), inplace=True)

    # Step 3: Payoff comparator + rotation
    # For each price state |i>, if S_i > K (call) or S_i < K (put),
    # rotate ancilla proportional to payoff
    values = dist.values
    payoffs = np.zeros(len(values))
    for i, s in enumerate(values):
        if spec.is_call:
            payoffs[i] = max(s - spec.k, 0)
        else:
            payoffs[i] = max(spec.k - s, 0)

    # Normalize payoffs to [0, 1] range for rotation angle
    max_payoff = np.max(payoffs)
    if max_payoff == 0:
        max_payoff = 1.0  # avoid division by zero
    rescale_factor = max_payoff

    # For each basis state |i>, apply controlled Ry rotation on ancilla
    # angle = 2 * arcsin(sqrt(payoff_i / max_payoff))
    for i in range(2**n_price):
        if payoffs[i] > 0:
            normalized_payoff = payoffs[i] / max_payoff
            angle = 2 * np.arcsin(np.sqrt(min(normalized_payoff, 1.0)))

            # Control on basis state |i> in qiskit's little-endian convention:
            # qubit q holds bit (i >> q) & 1, matching the StatePreparation that
            # loaded amplitude sqrt(p_i) onto basis integer i. X the zero-bit
            # qubits so the all-ones multi-control fires exactly on |i>.
            zero_qubits = [q for q in range(n_price) if not ((i >> q) & 1)]
            for q in zero_qubits:
                qc.x(q)

            # Multi-controlled Ry onto the payoff ancilla
            if n_price == 1:
                qc.cry(angle, 0, n_price)
            else:
                from qiskit.circuit.library import RYGate
                qc.append(
                    RYGate(angle).control(n_price),
                    [*list(range(n_price)), n_price],
                )

            for q in zero_qubits:
                qc.x(q)

    # The ancilla qubit is the objective
    # Measuring |1> on ancilla occurs with probability
    # sum_i p_i * sin^2(theta_i) where sin^2(theta_i) ~ payoff_i/max_payoff
    objective_qubits = [n_price]  # ancilla qubit

    # Discount factor for present value
    discount = np.exp(-spec.r * spec.T)

    # Total rescale: QAE gives P(ancilla=1) = sum_i p_i * payoff_i/max_payoff
    # Actual price = discount * max_payoff * estimate
    total_rescale = discount * rescale_factor

    problem = EstimationProblem(
        state_preparation=qc,
        objective_qubits=objective_qubits,
        n_qubits=n_total,
    )

    return problem, total_rescale
