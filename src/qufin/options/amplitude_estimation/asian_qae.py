"""Dedicated Asian option pricer via Quantum Amplitude Estimation.

Implements arithmetic and geometric Asian option pricing using
path discretisation and amplitude encoding.  The QAE infrastructure
from the canonical/IQAE modules is used for the estimation step.

Asian options depend on the average price along a path, so the circuit
must encode multiple time steps and compute the running average.
Arithmetic averaging uses the mean of prices; geometric averaging
uses the mean of log-prices (equivalent to the geometric mean).

References
----------
Stamatopoulos et al., Quantum 4:291 (2020), arXiv:1905.02666.
Chakrabarti et al., Quantum 5:463 (2021), arXiv:2012.03819.
Rebentrost et al., PRA 98, 022321 (2018).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray


@dataclass
class AsianQAESpec:
    """Specification for Asian option pricing via QAE.

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
    n_steps : int
        Number of monitoring dates (path discretisation steps).
    is_call : bool
        True for call, False for put.
    averaging : str
        ``"arithmetic"`` or ``"geometric"``.
    n_price_qubits : int
        Qubits per time step for price discretisation.
    n_precision_qubits : int
        Qubits for the QAE precision register.
    """

    s0: float = 100.0
    k: float = 100.0
    r: float = 0.05
    sigma: float = 0.2
    T: float = 1.0
    n_steps: int = 4
    is_call: bool = True
    averaging: Literal["arithmetic", "geometric"] = "arithmetic"
    n_price_qubits: int = 3
    n_precision_qubits: int = 6


@dataclass
class AsianQAEResult:
    """Result from Asian option QAE pricing.

    Parameters
    ----------
    price : float
        Estimated option price.
    std_error : float
        Standard error of the estimate.
    n_oracle_calls : int
        Total oracle queries used.
    n_qubits : int
        Qubits used in the circuit.
    averaging : str
        Averaging method used.
    path_prices : NDArray
        Discretised path price grid.
    path_probs : NDArray
        Path probabilities.
    classical_price : float
        Classical MC reference price.
    wall_time_s : float
        Wall-clock time.
    metadata : dict
        Additional metadata.
    """

    price: float = 0.0
    std_error: float = 0.0
    n_oracle_calls: int = 0
    n_qubits: int = 0
    averaging: str = "arithmetic"
    path_prices: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(0)
    )
    path_probs: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(0)
    )
    classical_price: float = 0.0
    wall_time_s: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


def _discretise_gbm_step(
    s_prev: NDArray[np.float64],
    dt: float,
    r: float,
    sigma: float,
    n_price_qubits: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Discretise one GBM step into price bins with probabilities.

    Parameters
    ----------
    s_prev : NDArray
        Price grid at the previous step.
    dt : float
        Time increment.
    r : float
        Risk-free rate.
    sigma : float
        Volatility.
    n_price_qubits : int
        Number of qubits for price discretisation.

    Returns
    -------
    (prices, transition_probs)
        prices: shape (n_bins,) discretised prices.
        transition_probs: shape (n_prev, n_bins) transition matrix.
    """
    from scipy.stats import norm

    n_bins = 2 ** n_price_qubits
    mu_log = (r - 0.5 * sigma ** 2) * dt
    sigma_log = sigma * np.sqrt(dt)

    # Price range: cover +/- 4 sigma around the mean log-price
    log_prev = np.log(np.maximum(s_prev, 1e-10))
    mean_log = np.mean(log_prev) + mu_log
    spread = 4 * sigma_log + np.std(log_prev) if len(log_prev) > 1 else 4 * sigma_log
    log_low = mean_log - spread
    log_high = mean_log + spread

    log_prices = np.linspace(log_low, log_high, n_bins)
    prices = np.exp(log_prices)

    # Transition probabilities
    transition = np.zeros((len(s_prev), n_bins), dtype=np.float64)
    for i, sp in enumerate(s_prev):
        loc = np.log(max(sp, 1e-10)) + mu_log
        probs = norm.pdf(log_prices, loc=loc, scale=sigma_log)
        prob_sum = probs.sum()
        if prob_sum > 0:
            transition[i] = probs / prob_sum
        else:
            transition[i] = 1.0 / n_bins

    return prices, transition


def generate_path_distribution(
    spec: AsianQAESpec,
) -> tuple[list[NDArray[np.float64]], list[NDArray[np.float64]]]:
    """Generate discretised path price grids and marginal probabilities.

    Parameters
    ----------
    spec : AsianQAESpec
        Asian option specification.

    Returns
    -------
    (price_grids, prob_grids)
        price_grids: list of NDArray, one per time step.
        prob_grids: list of NDArray, marginal probabilities per step.
    """
    dt = spec.T / spec.n_steps
    price_grids: list[NDArray[np.float64]] = [np.array([spec.s0])]
    prob_grids: list[NDArray[np.float64]] = [np.array([1.0])]

    current_prices = np.array([spec.s0])
    current_probs = np.array([1.0])

    for _step in range(spec.n_steps):
        next_prices, trans = _discretise_gbm_step(
            current_prices, dt, spec.r, spec.sigma, spec.n_price_qubits,
        )
        # Marginal probabilities for next step
        next_probs = current_probs @ trans
        prob_sum = next_probs.sum()
        if prob_sum > 0:
            next_probs /= prob_sum

        price_grids.append(next_prices)
        prob_grids.append(next_probs)
        current_prices = next_prices
        current_probs = next_probs

    return price_grids, prob_grids


def _compute_asian_payoffs(
    price_grids: list[NDArray[np.float64]],
    prob_grids: list[NDArray[np.float64]],
    spec: AsianQAESpec,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute payoff values and their probabilities.

    Approximates the expected payoff by averaging over the
    terminal marginal distribution weighted by path averages.

    Returns
    -------
    (payoffs, probabilities) each of shape (n_bins,) at the
    terminal time step.
    """
    # Use terminal price distribution as proxy
    terminal_prices = price_grids[-1]
    terminal_probs = prob_grids[-1]

    # Approximate the average price as weighted average of
    # marginal means + terminal price
    marginal_means = []
    for t in range(1, len(price_grids)):
        mean_t = float(np.dot(price_grids[t], prob_grids[t]))
        marginal_means.append(mean_t)

    # For each terminal price bin, approximate the path average
    n_bins = len(terminal_prices)
    payoffs = np.zeros(n_bins, dtype=np.float64)

    for i, s_T in enumerate(terminal_prices):
        if spec.averaging == "arithmetic":
            # Average = (sum of marginal means + s_T) / n_steps
            avg = (sum(marginal_means[:-1]) + s_T) / spec.n_steps if marginal_means else s_T
        else:
            # Geometric: exp(mean of log prices)
            log_means = [np.log(max(m, 1e-10)) for m in marginal_means[:-1]]
            log_means.append(np.log(max(s_T, 1e-10)))
            avg = np.exp(np.mean(log_means)) if log_means else s_T

        if spec.is_call:
            payoffs[i] = max(avg - spec.k, 0.0)
        else:
            payoffs[i] = max(spec.k - avg, 0.0)

    return payoffs, terminal_probs


def price_asian_option_qae(
    spec: AsianQAESpec,
    seed: int | None = None,
) -> AsianQAEResult:
    """Price an Asian option using Quantum Amplitude Estimation.

    Discretises the GBM path, computes payoffs based on the chosen
    averaging method, and uses QAE-based integration to estimate
    the discounted expected payoff.

    Parameters
    ----------
    spec : AsianQAESpec
        Asian option specification.
    seed : int | None
        Random seed for reproducibility.

    Returns
    -------
    AsianQAEResult
    """
    from qufin.options.amplitude_estimation.qmc import quantum_mean_estimation

    start = time.perf_counter()

    price_grids, prob_grids = generate_path_distribution(spec)
    payoffs, probs = _compute_asian_payoffs(price_grids, prob_grids, spec)

    discount = np.exp(-spec.r * spec.T)
    discounted_payoffs = discount * payoffs

    qmc_result = quantum_mean_estimation(
        f_values=discounted_payoffs,
        probabilities=probs,
        n_precision_qubits=spec.n_precision_qubits,
        seed=seed,
    )

    # Resource accounting
    n_qubits = spec.n_price_qubits * spec.n_steps + spec.n_precision_qubits + 1

    # Classical MC reference
    classical_price = _classical_asian_mc(spec, seed=seed)

    wall_time = time.perf_counter() - start

    return AsianQAEResult(
        price=qmc_result.estimate,
        std_error=qmc_result.std_error,
        n_oracle_calls=qmc_result.n_oracle_calls,
        n_qubits=n_qubits,
        averaging=spec.averaging,
        path_prices=price_grids[-1],
        path_probs=prob_grids[-1],
        classical_price=classical_price,
        wall_time_s=wall_time,
        metadata={
            "n_steps": spec.n_steps,
            "discount_factor": discount,
            "n_price_bins": 2 ** spec.n_price_qubits,
            "qmc_amplitude": qmc_result.metadata.get("amplitude", 0.0),
        },
    )


def _classical_asian_mc(
    spec: AsianQAESpec,
    n_samples: int = 50000,
    seed: int | None = None,
) -> float:
    """Classical Monte Carlo price for an Asian option.

    Parameters
    ----------
    spec : AsianQAESpec
        Option specification.
    n_samples : int
        Number of MC paths.
    seed : int | None
        Random seed.

    Returns
    -------
    float
        MC estimate of the discounted expected payoff.
    """
    rng = np.random.default_rng(seed)
    dt = spec.T / spec.n_steps

    # Simulate GBM paths
    z = rng.standard_normal((n_samples, spec.n_steps))
    log_increments = (spec.r - 0.5 * spec.sigma ** 2) * dt + spec.sigma * np.sqrt(dt) * z
    log_paths = np.log(spec.s0) + np.cumsum(log_increments, axis=1)
    price_paths = np.exp(log_paths)

    # Compute averages
    if spec.averaging == "arithmetic":
        averages = np.mean(price_paths, axis=1)
    else:
        averages = np.exp(np.mean(log_paths, axis=1))

    # Payoffs
    if spec.is_call:
        payoffs = np.maximum(averages - spec.k, 0.0)
    else:
        payoffs = np.maximum(spec.k - averages, 0.0)

    discount = np.exp(-spec.r * spec.T)
    return float(np.mean(discount * payoffs))


def compare_asian_pricing(
    spec: AsianQAESpec,
    n_classical_samples: int = 100000,
    seed: int | None = None,
) -> dict[str, Any]:
    """Compare QAE and classical MC Asian option prices.

    Parameters
    ----------
    spec : AsianQAESpec
        Option specification.
    n_classical_samples : int
        Number of classical MC samples.
    seed : int | None
        Random seed.

    Returns
    -------
    dict with qae_price, classical_price, and comparison metrics.
    """
    qae_result = price_asian_option_qae(spec, seed=seed)
    classical = _classical_asian_mc(spec, n_samples=n_classical_samples, seed=seed)

    return {
        "qae_price": qae_result.price,
        "classical_price": classical,
        "qae_std_error": qae_result.std_error,
        "n_oracle_calls": qae_result.n_oracle_calls,
        "n_qubits": qae_result.n_qubits,
        "averaging": spec.averaging,
        "n_steps": spec.n_steps,
    }
