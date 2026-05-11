"""Asian option pricing (geometric + arithmetic) via QAE.

Constructs QAE estimation problems for Asian options where the payoff
depends on the average price over a monitoring window.

For geometric Asian options, closed-form BS-like solutions exist.
For arithmetic, QAE provides an alternative to Monte Carlo.

References
----------
Stamatopoulos et al., Quantum 4:291 (2020), arXiv:1905.02666.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from qufin.options.amplitude_estimation.estimation_problem import EstimationProblem
from qufin.options.distributions import log_normal_distribution


@dataclass
class AsianOptionSpec:
    """Asian option specification.

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
    n_monitoring : int
        Number of monitoring dates (equally spaced).
    is_call : bool
        True for call, False for put.
    average_type : str
        "arithmetic" or "geometric".
    n_qubits_per_step : int
        Qubits per monitoring date for QAE discretization.
    """

    s0: float = 100.0
    k: float = 100.0
    r: float = 0.05
    sigma: float = 0.2
    T: float = 1.0
    n_monitoring: int = 4
    is_call: bool = True
    average_type: Literal["arithmetic", "geometric"] = "arithmetic"
    n_qubits_per_step: int = 3


def geometric_asian_closed_form(
    s0: float,
    k: float,
    r: float,
    sigma: float,
    T: float,
    n_monitoring: int,
    is_call: bool = True,
) -> float:
    """Closed-form price for geometric Asian option (Kemna-Vorst).

    The geometric average follows a log-normal distribution, so
    Black-Scholes-like formulas apply.

    Parameters
    ----------
    s0, k, r, sigma, T : float
        Standard option parameters.
    n_monitoring : int
        Number of equally spaced monitoring dates.
    is_call : bool
        Call or put.

    Returns
    -------
    float
        Option price.
    """
    from scipy.stats import norm

    n = n_monitoring
    # Adjusted volatility and drift for geometric average
    sigma_g = sigma * np.sqrt((2 * n + 1) / (6 * (n + 1)))
    mu_g = (r - 0.5 * sigma**2) * (n + 1) / (2 * n) + 0.5 * sigma_g**2

    d1 = (np.log(s0 / k) + (mu_g + 0.5 * sigma_g**2) * T) / (sigma_g * np.sqrt(T))
    d2 = d1 - sigma_g * np.sqrt(T)

    discount = np.exp(-r * T)
    if is_call:
        price = discount * (s0 * np.exp(mu_g * T) * norm.cdf(d1) - k * norm.cdf(d2))
    else:
        price = discount * (k * norm.cdf(-d2) - s0 * np.exp(mu_g * T) * norm.cdf(-d1))

    return float(price)


def build_asian_estimation_problem(
    spec: AsianOptionSpec,
) -> tuple[EstimationProblem, float]:
    """Build QAE estimation problem for Asian option pricing.

    Uses a simplified single-step approximation: loads the distribution
    of the average price and applies payoff comparator.

    For multi-step paths, the average distribution is approximated
    by a log-normal with adjusted parameters.

    Returns
    -------
    tuple of (EstimationProblem, rescale_factor)
    """
    from qiskit.circuit import QuantumCircuit

    n = spec.n_monitoring

    # Approximate the average price distribution
    # For geometric: exact log-normal with adjusted params
    # For arithmetic: approximate as log-normal (moment matching)
    sigma_avg = spec.sigma * np.sqrt((2 * n + 1) / (6 * (n + 1)))
    mu_avg = (spec.r - 0.5 * spec.sigma**2) * (n + 1) / (2 * n) + 0.5 * sigma_avg**2

    n_q = spec.n_qubits_per_step
    dist = log_normal_distribution(
        n_qubits=n_q,
        s0=spec.s0,
        mu=mu_avg,
        sigma=sigma_avg,
        T=spec.T,
    )

    n_total = n_q + 1  # +1 ancilla
    qc = QuantumCircuit(n_total)

    # Load average price distribution
    amplitudes = dist.amplitudes()
    norm_val = np.linalg.norm(amplitudes)
    if norm_val > 0:
        amplitudes = amplitudes / norm_val
    qc.initialize(amplitudes, range(n_q))

    # Payoff rotation on ancilla
    values = dist.values
    payoffs = np.zeros(len(values))
    for i, s in enumerate(values):
        if spec.is_call:
            payoffs[i] = max(s - spec.k, 0)
        else:
            payoffs[i] = max(spec.k - s, 0)

    max_payoff = float(np.max(payoffs))
    if max_payoff == 0:
        max_payoff = 1.0

    for i in range(2**n_q):
        if payoffs[i] > 0:
            normalized = payoffs[i] / max_payoff
            angle = 2 * np.arcsin(np.sqrt(min(normalized, 1.0)))

            bits = format(i, f"0{n_q}b")
            for b_idx, b in enumerate(bits):
                if b == "0":
                    qc.x(b_idx)

            if n_q == 1:
                qc.cry(angle, 0, n_q)
            else:
                from qiskit.circuit.library import RYGate
                qc.append(
                    RYGate(angle).control(n_q),
                    [*list(range(n_q)), n_q],
                )

            for b_idx, b in enumerate(bits):
                if b == "0":
                    qc.x(b_idx)

    discount = np.exp(-spec.r * spec.T)
    rescale = discount * max_payoff

    problem = EstimationProblem(
        state_preparation=qc,
        objective_qubits=[n_q],
        n_qubits=n_total,
    )

    return problem, rescale
