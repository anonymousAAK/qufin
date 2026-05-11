"""Barrier option pricing via QAE.

Constructs QAE estimation problems for barrier options with
knock-in / knock-out features.

References
----------
Stamatopoulos et al., Quantum 4:291 (2020), arXiv:1905.02666, Fig. 8.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from qufin.options.amplitude_estimation.estimation_problem import EstimationProblem
from qufin.options.distributions import log_normal_distribution


@dataclass
class BarrierOptionSpec:
    """Barrier option specification.

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
    barrier : float
        Barrier level.
    barrier_type : str
        One of "up-and-out", "down-and-out", "up-and-in", "down-and-in".
    is_call : bool
        True for call, False for put.
    n_qubits : int
        Qubits for price discretization.
    """

    s0: float = 100.0
    k: float = 100.0
    r: float = 0.05
    sigma: float = 0.2
    T: float = 1.0
    barrier: float = 120.0
    barrier_type: Literal["up-and-out", "down-and-out", "up-and-in", "down-and-in"] = "up-and-out"
    is_call: bool = True
    n_qubits: int = 3


def barrier_closed_form(
    s0: float,
    k: float,
    r: float,
    sigma: float,
    T: float,
    barrier: float,
    barrier_type: str = "up-and-out",
    is_call: bool = True,
) -> float:
    """Closed-form barrier option price (Merton 1973, Reiner-Rubinstein 1991).

    Only supports single-barrier European options.
    Uses in-out parity: knock-in + knock-out = vanilla.
    """
    from scipy.stats import norm

    d1 = (np.log(s0 / k) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Vanilla BS price
    if is_call:
        vanilla = float(s0 * norm.cdf(d1) - k * np.exp(-r * T) * norm.cdf(d2))
    else:
        vanilla = float(k * np.exp(-r * T) * norm.cdf(-d2) - s0 * norm.cdf(-d1))

    H = barrier

    # When barrier is very far from spot, knock-out ~ vanilla
    # (probability of hitting barrier is negligible)
    barrier_distance = abs(np.log(H / s0)) / (sigma * np.sqrt(T))
    if barrier_distance > 6:
        if "out" in barrier_type:
            return vanilla
        else:  # knock-in
            return 0.0

    lam = (r + 0.5 * sigma**2) / sigma**2
    y = np.log(H**2 / (s0 * k)) / (sigma * np.sqrt(T)) + lam * sigma * np.sqrt(T)

    ratio = H / s0

    if barrier_type == "up-and-out" and s0 >= H:
        # Already above barrier — knocked out immediately
        return 0.0

    if barrier_type == "down-and-out" and s0 <= H:
        # Already below barrier — knocked out immediately
        return 0.0

    if barrier_type == "up-and-out" and is_call and s0 < H:
        c_up_in = (
            s0 * ratio ** (2 * lam) * norm.cdf(y)
            - k * np.exp(-r * T) * ratio ** (2 * lam - 2) * norm.cdf(y - sigma * np.sqrt(T))
        )
        return max(vanilla - float(c_up_in), 0.0)

    elif barrier_type == "up-and-out" and not is_call and s0 < H:
        # Up-and-out put
        x1 = np.log(s0 / H) / (sigma * np.sqrt(T)) + lam * sigma * np.sqrt(T)
        p_up_in = (
            -s0 * ratio ** (2 * lam) * norm.cdf(-y)
            + k * np.exp(-r * T) * ratio ** (2 * lam - 2) * norm.cdf(-y + sigma * np.sqrt(T))
        )
        return max(vanilla - float(p_up_in), 0.0)

    elif barrier_type == "down-and-out" and is_call and s0 > H:
        x1 = np.log(s0 / H) / (sigma * np.sqrt(T)) + lam * sigma * np.sqrt(T)
        c_do = (
            s0 * norm.cdf(x1)
            - k * np.exp(-r * T) * norm.cdf(x1 - sigma * np.sqrt(T))
            - s0 * (H / s0) ** (2 * lam) * norm.cdf(y)
            + k * np.exp(-r * T) * (H / s0) ** (2 * lam - 2) * norm.cdf(y - sigma * np.sqrt(T))
        )
        return max(float(c_do), 0.0)

    elif barrier_type == "down-and-out" and not is_call and s0 > H:
        # Down-and-out put
        x1 = np.log(s0 / H) / (sigma * np.sqrt(T)) + lam * sigma * np.sqrt(T)
        p_do = (
            -s0 * norm.cdf(-x1)
            + k * np.exp(-r * T) * norm.cdf(-x1 + sigma * np.sqrt(T))
            + s0 * (H / s0) ** (2 * lam) * norm.cdf(-y)
            - k * np.exp(-r * T) * (H / s0) ** (2 * lam - 2) * norm.cdf(-y + sigma * np.sqrt(T))
        )
        return max(float(p_do), 0.0)

    elif "in" in barrier_type:
        # In-out parity: knock-in = vanilla - knock-out
        out_type = barrier_type.replace("-in", "-out")
        out_price = barrier_closed_form(s0, k, r, sigma, T, barrier, out_type, is_call)
        return max(vanilla - out_price, 0.0)

    return vanilla


def build_barrier_estimation_problem(
    spec: BarrierOptionSpec,
) -> tuple[EstimationProblem, float]:
    """Build QAE estimation problem for barrier option pricing.

    Uses terminal price distribution with barrier condition applied
    as a constraint on the payoff function.

    For single-step (European-style barrier checked only at expiry):
    payoff = max(S_T - K, 0) * I(barrier not breached at T)

    Returns
    -------
    tuple of (EstimationProblem, rescale_factor)
    """
    from qiskit.circuit import QuantumCircuit

    n_q = spec.n_qubits
    dist = log_normal_distribution(
        n_qubits=n_q,
        s0=spec.s0,
        mu=spec.r,
        sigma=spec.sigma,
        T=spec.T,
    )

    n_total = n_q + 1  # +1 ancilla
    qc = QuantumCircuit(n_total)

    # Load price distribution
    amplitudes = dist.amplitudes()
    norm_val = np.linalg.norm(amplitudes)
    if norm_val > 0:
        amplitudes = amplitudes / norm_val
    qc.initialize(amplitudes, range(n_q))

    # Compute payoffs with barrier condition
    values = dist.values
    payoffs = np.zeros(len(values))
    for i, s in enumerate(values):
        # Basic payoff
        raw_payoff = max(s - spec.k, 0) if spec.is_call else max(spec.k - s, 0)

        # Apply barrier condition (European-style, checked at expiry)
        barrier_active = True
        if spec.barrier_type == "up-and-out":
            barrier_active = s < spec.barrier
        elif spec.barrier_type == "down-and-out":
            barrier_active = s > spec.barrier
        elif spec.barrier_type == "up-and-in":
            barrier_active = s >= spec.barrier
        elif spec.barrier_type == "down-and-in":
            barrier_active = s <= spec.barrier

        payoffs[i] = raw_payoff if barrier_active else 0.0

    max_payoff = float(np.max(payoffs))
    if max_payoff == 0:
        max_payoff = 1.0

    # Payoff rotation
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
