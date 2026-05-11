"""Distribution loading for QAE: log-normal, normal, GARCH-implied.

Prepares quantum circuits that load classical probability distributions
into qubit amplitudes for use in quantum amplitude estimation.

References
----------
Stamatopoulos et al., Quantum 4:291 (2020), arXiv:1905.02666, Section III-B.
Grover & Rudolph, arXiv:quant-ph/0208112 — efficient state preparation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class DistributionSpec:
    """Specification for a discretized probability distribution.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (2^n_qubits grid points).
    low : float
        Lower bound of the domain.
    high : float
        Upper bound of the domain.
    probabilities : NDArray[np.float64]
        Discrete probability vector, shape (2^n_qubits,).
    values : NDArray[np.float64]
        Grid values corresponding to each probability, shape (2^n_qubits,).
    """

    n_qubits: int
    low: float
    high: float
    probabilities: NDArray[np.float64]
    values: NDArray[np.float64]

    @property
    def n_states(self) -> int:
        return 2**self.n_qubits

    def amplitudes(self) -> NDArray[np.float64]:
        """Square root of probabilities for state preparation."""
        return np.sqrt(self.probabilities)


def log_normal_distribution(
    n_qubits: int = 3,
    s0: float = 100.0,
    mu: float = 0.05,
    sigma: float = 0.2,
    T: float = 1.0,
    n_sigma: float = 3.0,
) -> DistributionSpec:
    """Log-normal distribution for GBM stock prices.

    At time T, the stock price follows:
    S_T = S_0 * exp((mu - sigma^2/2)*T + sigma*sqrt(T)*Z)

    where Z ~ N(0,1).

    Parameters
    ----------
    n_qubits : int
        Number of qubits for discretization.
    s0 : float
        Initial stock price.
    mu : float
        Drift (risk-free rate for risk-neutral pricing).
    sigma : float
        Volatility.
    T : float
        Time to expiry.
    n_sigma : float
        Number of standard deviations for domain bounds.
    """
    n_states = 2**n_qubits

    # Log-normal parameters
    ln_mean = np.log(s0) + (mu - 0.5 * sigma**2) * T
    ln_std = sigma * np.sqrt(T)

    # Domain in log-space, then transform
    low_log = ln_mean - n_sigma * ln_std
    high_log = ln_mean + n_sigma * ln_std
    low = float(np.exp(low_log))
    high = float(np.exp(high_log))

    # Grid values (uniform in price space)
    values = np.linspace(low, high, n_states)

    # Compute log-normal PDF at grid points
    probs = np.zeros(n_states)
    for i, s in enumerate(values):
        if s > 0:
            log_s = np.log(s)
            probs[i] = np.exp(-0.5 * ((log_s - ln_mean) / ln_std) ** 2) / (
                s * ln_std * np.sqrt(2 * np.pi)
            )

    # Normalize to form a proper discrete distribution
    dx = (high - low) / (n_states - 1) if n_states > 1 else 1.0
    probs = probs * dx
    total = probs.sum()
    if total > 0:
        probs = probs / total

    return DistributionSpec(
        n_qubits=n_qubits,
        low=low,
        high=high,
        probabilities=probs,
        values=values,
    )


def normal_distribution(
    n_qubits: int = 3,
    mean: float = 0.0,
    std: float = 1.0,
    n_sigma: float = 3.0,
) -> DistributionSpec:
    """Normal (Gaussian) distribution.

    Parameters
    ----------
    n_qubits : int
        Number of qubits for discretization.
    mean : float
        Mean of the distribution.
    std : float
        Standard deviation.
    n_sigma : float
        Number of standard deviations for domain bounds.
    """
    n_states = 2**n_qubits
    low = mean - n_sigma * std
    high = mean + n_sigma * std
    values = np.linspace(low, high, n_states)

    probs = np.exp(-0.5 * ((values - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))
    dx = (high - low) / (n_states - 1) if n_states > 1 else 1.0
    probs = probs * dx
    total = probs.sum()
    if total > 0:
        probs = probs / total

    return DistributionSpec(
        n_qubits=n_qubits,
        low=low,
        high=high,
        probabilities=probs,
        values=values,
    )


def uniform_distribution(
    n_qubits: int = 3,
    low: float = 0.0,
    high: float = 1.0,
) -> DistributionSpec:
    """Uniform distribution on [low, high]."""
    n_states = 2**n_qubits
    values = np.linspace(low, high, n_states)
    probs = np.ones(n_states) / n_states

    return DistributionSpec(
        n_qubits=n_qubits,
        low=low,
        high=high,
        probabilities=probs,
        values=values,
    )


def build_loading_circuit(dist: DistributionSpec) -> object:
    """Build a quantum circuit that loads a distribution into amplitudes.

    Uses Qiskit's Initialize instruction to prepare the state
    |psi> = sum_i sqrt(p_i) |i>.

    Parameters
    ----------
    dist : DistributionSpec
        The distribution to load.

    Returns
    -------
    QuantumCircuit
        Circuit that prepares the distribution state.
    """
    from qiskit.circuit import QuantumCircuit

    amplitudes = dist.amplitudes()

    # Ensure normalization (numerical safety)
    norm = np.linalg.norm(amplitudes)
    if norm > 0:
        amplitudes = amplitudes / norm

    qc = QuantumCircuit(dist.n_qubits)
    qc.initialize(amplitudes, range(dist.n_qubits))
    return qc
