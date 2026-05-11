"""Heston stochastic-volatility option pricing via QAE.

Implements weak-Euler discretization of the Heston model for
loading into quantum circuits, following Wang & Kan (2024).

The Heston model:
    dS = r*S*dt + sqrt(V)*S*dW_1
    dV = kappa*(theta - V)*dt + xi*sqrt(V)*dW_2
    corr(dW_1, dW_2) = rho

References
----------
Wang & Kan, Quantum 8:1504 (2024), arXiv:2312.15871.
Heston, Review of Financial Studies 6(2):327-343 (1993).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from qufin.options.distributions import DistributionSpec


@dataclass
class HestonParams:
    """Heston model parameters.

    Parameters
    ----------
    s0 : float
        Initial stock price.
    v0 : float
        Initial variance.
    r : float
        Risk-free rate.
    kappa : float
        Mean reversion speed.
    theta : float
        Long-run variance.
    xi : float
        Vol-of-vol (volatility of variance).
    rho : float
        Correlation between stock and variance Brownian motions.
    T : float
        Time to expiry.
    """

    s0: float = 100.0
    v0: float = 0.04
    r: float = 0.05
    kappa: float = 2.0
    theta: float = 0.04
    xi: float = 0.3
    rho: float = -0.7
    T: float = 1.0


def heston_weak_euler_terminal(
    params: HestonParams,
    n_steps: int = 100,
    n_paths: int = 100_000,
    seed: int | None = 42,
) -> NDArray[np.float64]:
    """Simulate Heston terminal prices using weak-Euler scheme.

    Wang & Kan (2024) show that weak-Euler matches strong-Euler accuracy
    while eliminating the need for Gaussian state preparation on quantum
    hardware. The key insight: weak convergence only requires matching
    moments, not pathwise accuracy.

    Uses full truncation to ensure V >= 0.

    Returns
    -------
    NDArray of shape (n_paths,)
        Terminal stock prices.
    """
    rng = np.random.default_rng(seed)
    dt = params.T / n_steps

    s = np.full(n_paths, params.s0)
    v = np.full(n_paths, params.v0)

    for _ in range(n_steps):
        z1 = rng.standard_normal(n_paths)
        z2 = params.rho * z1 + np.sqrt(1 - params.rho**2) * rng.standard_normal(n_paths)

        v_pos = np.maximum(v, 0)
        sqrt_v = np.sqrt(v_pos)

        # Weak Euler: use sqrt(dt) * z instead of dW
        s = s * np.exp((params.r - 0.5 * v_pos) * dt + sqrt_v * np.sqrt(dt) * z1)
        v = v + params.kappa * (params.theta - v_pos) * dt + params.xi * sqrt_v * np.sqrt(dt) * z2

        # Full truncation
        v = np.maximum(v, 0)

    return s


def heston_strong_euler_terminal(
    params: HestonParams,
    n_steps: int = 100,
    n_paths: int = 100_000,
    seed: int | None = 42,
) -> NDArray[np.float64]:
    """Simulate Heston terminal prices using strong-Euler scheme.

    Standard Euler-Maruyama with full truncation.

    Returns
    -------
    NDArray of shape (n_paths,)
        Terminal stock prices.
    """
    rng = np.random.default_rng(seed)
    dt = params.T / n_steps

    s = np.full(n_paths, params.s0)
    v = np.full(n_paths, params.v0)

    for _ in range(n_steps):
        z1 = rng.standard_normal(n_paths)
        z2 = params.rho * z1 + np.sqrt(1 - params.rho**2) * rng.standard_normal(n_paths)

        v_pos = np.maximum(v, 0)
        sqrt_v = np.sqrt(v_pos)

        s = s + params.r * s * dt + sqrt_v * s * np.sqrt(dt) * z1
        v = v + params.kappa * (params.theta - v_pos) * dt + params.xi * sqrt_v * np.sqrt(dt) * z2

        s = np.maximum(s, 0)
        v = np.maximum(v, 0)

    return s


def heston_european_price(
    params: HestonParams,
    k: float = 100.0,
    is_call: bool = True,
    n_steps: int = 100,
    n_paths: int = 200_000,
    method: Literal["weak_euler", "strong_euler"] = "weak_euler",
    seed: int | None = 42,
) -> tuple[float, float]:
    """Price a European option under Heston via Monte Carlo.

    Returns
    -------
    tuple of (price, std_err)
    """
    if method == "weak_euler":
        s_T = heston_weak_euler_terminal(params, n_steps, n_paths, seed)
    else:
        s_T = heston_strong_euler_terminal(params, n_steps, n_paths, seed)

    payoffs = np.maximum(s_T - k, 0) if is_call else np.maximum(k - s_T, 0)

    discounted = np.exp(-params.r * params.T) * payoffs
    price = float(np.mean(discounted))
    std_err = float(np.std(discounted, ddof=1) / np.sqrt(n_paths))

    return price, std_err


def heston_terminal_distribution(
    params: HestonParams,
    n_qubits: int = 4,
    n_paths: int = 200_000,
    n_sigma: float = 3.0,
    method: Literal["weak_euler", "strong_euler"] = "weak_euler",
    seed: int | None = 42,
) -> DistributionSpec:
    """Generate discretized terminal price distribution under Heston.

    Simulates many paths and fits a histogram to create a
    DistributionSpec suitable for QAE loading.

    Parameters
    ----------
    params : HestonParams
        Heston model parameters.
    n_qubits : int
        Qubits for discretization (2^n_qubits bins).
    n_paths : int
        Paths for Monte Carlo estimation.
    n_sigma : float
        Domain range in standard deviations from mean.
    method : str
        "weak_euler" or "strong_euler".
    seed : int | None
        Random seed.

    Returns
    -------
    DistributionSpec
    """
    if method == "weak_euler":
        s_T = heston_weak_euler_terminal(params, seed=seed, n_paths=n_paths)
    else:
        s_T = heston_strong_euler_terminal(params, seed=seed, n_paths=n_paths)

    n_states = 2**n_qubits
    mean_s = float(np.mean(s_T))
    std_s = float(np.std(s_T))

    low = max(0.01, mean_s - n_sigma * std_s)
    high = mean_s + n_sigma * std_s

    # Build histogram-based distribution
    values = np.linspace(low, high, n_states)
    bin_edges = np.zeros(n_states + 1)
    bin_edges[0] = low
    bin_edges[-1] = high
    dx = (high - low) / n_states
    for i in range(1, n_states):
        bin_edges[i] = low + i * dx

    counts, _ = np.histogram(s_T, bins=bin_edges)
    probs = counts.astype(np.float64) / counts.sum()

    return DistributionSpec(
        n_qubits=n_qubits,
        low=low,
        high=high,
        probabilities=probs,
        values=values,
    )


def resource_estimates(
    n_qubits_price: int,
    n_qubits_vol: int = 0,
    n_steps: int = 1,
) -> dict[str, int]:
    """Estimate T-count and T-depth for Heston QAE circuit.

    Based on Table III of Wang & Kan (2312.15871).
    These are approximate resource estimates for fault-tolerant
    implementations.

    Parameters
    ----------
    n_qubits_price : int
        Qubits for price register.
    n_qubits_vol : int
        Qubits for volatility register (0 = BS, >0 = Heston).
    n_steps : int
        Number of time steps.

    Returns
    -------
    dict with 'total_qubits', 'T_count', 'T_depth', 'cnot_count'.
    """
    n_p = n_qubits_price
    n_v = n_qubits_vol if n_qubits_vol > 0 else n_p

    # Rough estimates following Wang & Kan Table III scaling
    total_qubits = n_p + n_v + n_p + 1  # price + vol + ancilla + objective

    # T-count scales as O(n^2 * n_steps) for arithmetic circuits
    t_count = 8 * (n_p**2 + n_v**2) * n_steps

    # T-depth is parallelizable
    t_depth = 4 * (n_p + n_v) * n_steps

    # CNOT count
    cnot_count = 12 * (n_p**2 + n_v**2) * n_steps

    return {
        "total_qubits": total_qubits * n_steps,
        "T_count": t_count,
        "T_depth": t_depth,
        "cnot_count": cnot_count,
    }
