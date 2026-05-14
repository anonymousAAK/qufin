"""Quantum Monte Carlo Integration via Montanaro's algorithm.

Implements the quantum speedup for Monte Carlo integration using
amplitude estimation, achieving O(1/epsilon) convergence versus
the classical O(1/epsilon^2).

Key components:
- Generic mean estimation with quantum minimum-finding
- Median-of-means for robust estimation
- European option pricing application
- Resource estimation for single-asset and multi-asset problems

References
----------
Montanaro, A. "Quantum speedup of Monte Carlo methods."
    Proc. R. Soc. A 471:20150301 (2015), arXiv:1504.06987.
Brassard et al., "Quantum Amplitude Amplification and Estimation" (2002).
Chakrabarti et al., "A threshold for quantum advantage in derivative
    pricing", Quantum 5:463 (2021), arXiv:2012.03819.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class QMCResult:
    """Result of a Quantum Monte Carlo integration.

    Parameters
    ----------
    estimate : float
        Estimated integral / expectation value.
    std_error : float
        Standard error of the estimate.
    n_oracle_calls : int
        Total number of oracle queries used.
    n_qubits : int
        Number of qubits used in the circuit.
    classical_equivalent_samples : int
        Number of classical MC samples needed for same accuracy.
    speedup_factor : float
        Ratio classical_equivalent_samples / n_oracle_calls.
    wall_time_s : float
        Wall-clock time in seconds.
    confidence_interval : tuple[float, float]
        Confidence interval for the estimate.
    method : str
        Estimation method used ("qmc", "median_of_means", "classical").
    metadata : dict[str, Any]
        Additional metadata.
    """

    estimate: float = 0.0
    std_error: float = 0.0
    n_oracle_calls: int = 0
    n_qubits: int = 0
    classical_equivalent_samples: int = 0
    speedup_factor: float = 1.0
    wall_time_s: float = 0.0
    confidence_interval: tuple[float, float] = (0.0, 0.0)
    method: str = "qmc"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class QMCResourceEstimate:
    """Resource estimate for a QMC integration problem.

    Parameters
    ----------
    n_logical_qubits : int
        Number of logical qubits required.
    t_gate_count : int
        Estimated T-gate count for the circuit.
    t_depth : int
        T-depth of the circuit.
    n_physical_qubits : int
        Physical qubits required (surface code overhead).
    circuit_depth : int
        Total circuit depth.
    n_oracle_calls : int
        Number of oracle queries for target precision.
    classical_samples_equivalent : int
        Classical MC samples for equivalent precision.
    break_even_epsilon : float
        Precision at which quantum beats classical.
    problem_description : str
        Description of the problem being estimated.
    """

    n_logical_qubits: int = 0
    t_gate_count: int = 0
    t_depth: int = 0
    n_physical_qubits: int = 0
    circuit_depth: int = 0
    n_oracle_calls: int = 0
    classical_samples_equivalent: int = 0
    break_even_epsilon: float = 0.0
    problem_description: str = ""


@dataclass
class EuropeanQMCSpec:
    """Specification for European option pricing via QMC.

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
    n_price_qubits : int
        Qubits for price discretisation.
    n_precision_qubits : int
        Qubits for QAE precision.
    """

    s0: float = 100.0
    k: float = 100.0
    r: float = 0.05
    sigma: float = 0.2
    T: float = 1.0
    is_call: bool = True
    n_price_qubits: int = 4
    n_precision_qubits: int = 6


# ---------------------------------------------------------------------------
# Core QMC integration
# ---------------------------------------------------------------------------


def quantum_mean_estimation(
    f_values: NDArray[np.float64],
    probabilities: NDArray[np.float64],
    n_precision_qubits: int = 6,
    confidence: float = 0.95,
    seed: int | None = None,
) -> QMCResult:
    """Estimate E[f(X)] using Montanaro's QMC algorithm.

    Simulates the quantum amplitude estimation step classically
    to estimate the mean of f(X) with respect to the given
    probability distribution.

    The quantum algorithm achieves O(1/epsilon) convergence vs
    the classical O(1/epsilon^2), where epsilon is the target
    additive error.

    Parameters
    ----------
    f_values : NDArray
        Function values f(x_i) at discretisation points.
    probabilities : NDArray
        Probability p(x_i) at each discretisation point.
    n_precision_qubits : int
        Number of qubits controlling QAE precision.
        Precision ~ pi / 2^n_precision_qubits.
    confidence : float
        Desired confidence level (e.g. 0.95).
    seed : int | None
        Random seed for reproducibility.

    Returns
    -------
    QMCResult
        Estimation result with error analysis.
    """
    start = time.perf_counter()
    rng = np.random.default_rng(seed)

    # Validate inputs
    f_values = np.asarray(f_values, dtype=np.float64)
    probabilities = np.asarray(probabilities, dtype=np.float64)

    if len(f_values) != len(probabilities):
        msg = "f_values and probabilities must have same length"
        raise ValueError(msg)

    # Normalise probabilities
    prob_sum = np.sum(probabilities)
    if prob_sum > 0:
        probabilities = probabilities / prob_sum

    # True mean (for validation)
    true_mean = float(np.dot(f_values, probabilities))

    # Normalise f to [0, 1] for amplitude encoding
    f_min = float(np.min(f_values))
    f_max = float(np.max(f_values))
    f_range = f_max - f_min if f_max > f_min else 1.0

    f_normalised = (f_values - f_min) / f_range

    # QAE precision: epsilon ~ pi / 2^n
    n_precision = max(1, n_precision_qubits)
    M = 2**n_precision
    epsilon = np.pi / M

    # The amplitude to estimate: a = sum_i p_i * f_norm_i
    # This is encoded as the probability of the objective qubit
    amplitude = float(np.dot(f_normalised, probabilities))

    # Simulate QAE measurement with finite precision
    # theta_a = arcsin(sqrt(a))
    theta_a = np.arcsin(np.sqrt(np.clip(amplitude, 0, 1)))

    # QAE returns an integer k in [0, M-1] such that
    # k/M approximates theta_a / pi
    k_exact = theta_a * M / np.pi
    # Add shot noise simulation
    k_measured = int(np.round(k_exact + rng.normal(0, 0.5)))
    k_measured = np.clip(k_measured, 0, M - 1)

    # Recover amplitude estimate
    theta_est = k_measured * np.pi / M
    amplitude_est = np.sin(theta_est) ** 2

    # Rescale back to original range
    estimate = amplitude_est * f_range + f_min

    # Error analysis
    # QAE achieves epsilon = O(1/M) = O(1/2^n)
    std_error = f_range * epsilon

    # Classical equivalent: to achieve same std_error via MC,
    # need N ~ Var[f] / epsilon^2 samples
    variance = float(np.dot((f_values - true_mean) ** 2, probabilities))
    if std_error > 0:
        classical_equiv = max(1, int(np.ceil(variance / std_error**2)))
    else:
        classical_equiv = 1

    # Oracle calls: QAE uses O(M) = O(2^n) queries
    n_oracle_calls = M

    # Speedup factor
    speedup = classical_equiv / max(n_oracle_calls, 1)

    # Number of qubits: price register + precision register + ancilla
    n_price_bits = max(1, int(np.ceil(np.log2(max(len(f_values), 2)))))
    n_qubits = n_price_bits + n_precision + 1

    # Confidence interval
    z = 1.96 if confidence == 0.95 else 2.576
    ci = (estimate - z * std_error, estimate + z * std_error)

    wall_time = time.perf_counter() - start

    return QMCResult(
        estimate=estimate,
        std_error=std_error,
        n_oracle_calls=n_oracle_calls,
        n_qubits=n_qubits,
        classical_equivalent_samples=classical_equiv,
        speedup_factor=speedup,
        wall_time_s=wall_time,
        confidence_interval=ci,
        method="qmc",
        metadata={
            "n_precision_qubits": n_precision,
            "amplitude": amplitude,
            "amplitude_est": amplitude_est,
            "f_range": f_range,
            "true_mean": true_mean,
            "epsilon": epsilon,
        },
    )


def median_of_means_qmc(
    f_values: NDArray[np.float64],
    probabilities: NDArray[np.float64],
    n_precision_qubits: int = 6,
    n_blocks: int = 5,
    confidence: float = 0.95,
    seed: int | None = None,
) -> QMCResult:
    """Robust QMC estimation using median-of-means.

    Splits the estimation into n_blocks independent QAE runs
    and takes the median, providing robustness against outliers
    with exponentially decreasing failure probability.

    Parameters
    ----------
    f_values : NDArray
        Function values at discretisation points.
    probabilities : NDArray
        Probability distribution.
    n_precision_qubits : int
        Precision qubits per QAE run.
    n_blocks : int
        Number of independent estimation blocks.
        Must be odd for well-defined median.
    confidence : float
        Desired confidence level.
    seed : int | None
        Random seed.

    Returns
    -------
    QMCResult
        Robust estimation result.
    """
    start = time.perf_counter()

    # Ensure odd number of blocks
    n_blocks = max(n_blocks, 1)
    if n_blocks % 2 == 0:
        n_blocks += 1

    rng = np.random.default_rng(seed)
    block_seeds = rng.integers(0, 2**31, size=n_blocks)

    # Run independent QAE estimations
    estimates = []
    total_oracle_calls = 0
    for block_seed in block_seeds:
        result = quantum_mean_estimation(
            f_values,
            probabilities,
            n_precision_qubits=n_precision_qubits,
            confidence=confidence,
            seed=int(block_seed),
        )
        estimates.append(result.estimate)
        total_oracle_calls += result.n_oracle_calls

    estimates_arr = np.array(estimates)
    median_estimate = float(np.median(estimates_arr))

    # Robust standard error: MAD-based
    mad = float(np.median(np.abs(estimates_arr - median_estimate)))
    # Scale MAD to approximate std: MAD * 1.4826
    robust_std = mad * 1.4826 / np.sqrt(n_blocks)

    # Classical equivalent
    M = 2**n_precision_qubits
    epsilon = np.pi / M
    f_min = float(np.min(f_values))
    f_max = float(np.max(f_values))
    f_range = f_max - f_min if f_max > f_min else 1.0
    true_mean = float(np.dot(f_values, probabilities / np.sum(probabilities)))
    variance = float(
        np.dot(
            (f_values - true_mean) ** 2,
            probabilities / np.sum(probabilities),
        )
    )
    eff_error = f_range * epsilon / np.sqrt(n_blocks)
    classical_equiv = max(1, int(np.ceil(variance / max(eff_error**2, 1e-30))))

    speedup = classical_equiv / max(total_oracle_calls, 1)

    n_price_bits = max(1, int(np.ceil(np.log2(max(len(f_values), 2)))))
    n_qubits = n_price_bits + n_precision_qubits + 1

    z = 1.96 if confidence == 0.95 else 2.576
    ci = (median_estimate - z * robust_std, median_estimate + z * robust_std)

    wall_time = time.perf_counter() - start

    return QMCResult(
        estimate=median_estimate,
        std_error=robust_std,
        n_oracle_calls=total_oracle_calls,
        n_qubits=n_qubits,
        classical_equivalent_samples=classical_equiv,
        speedup_factor=speedup,
        wall_time_s=wall_time,
        confidence_interval=ci,
        method="median_of_means",
        metadata={
            "n_blocks": n_blocks,
            "block_estimates": estimates,
            "n_precision_qubits": n_precision_qubits,
            "mad": mad,
        },
    )


def classical_mc_estimation(
    f_values: NDArray[np.float64],
    probabilities: NDArray[np.float64],
    n_samples: int = 10000,
    confidence: float = 0.95,
    seed: int | None = None,
) -> QMCResult:
    """Classical Monte Carlo estimation for comparison.

    Parameters
    ----------
    f_values : NDArray
        Function values at discretisation points.
    probabilities : NDArray
        Probability distribution.
    n_samples : int
        Number of MC samples.
    confidence : float
        Confidence level.
    seed : int | None
        Random seed.

    Returns
    -------
    QMCResult
        Classical MC result for comparison with QMC.
    """
    start = time.perf_counter()
    rng = np.random.default_rng(seed)

    f_values = np.asarray(f_values, dtype=np.float64)
    probabilities = np.asarray(probabilities, dtype=np.float64)

    # Normalise
    prob_sum = np.sum(probabilities)
    if prob_sum > 0:
        probabilities = probabilities / prob_sum

    # Sample from distribution
    indices = rng.choice(len(f_values), size=n_samples, p=probabilities)
    samples = f_values[indices]

    estimate = float(np.mean(samples))
    std_error = float(np.std(samples, ddof=1) / np.sqrt(n_samples))

    z = 1.96 if confidence == 0.95 else 2.576
    ci = (estimate - z * std_error, estimate + z * std_error)

    wall_time = time.perf_counter() - start

    return QMCResult(
        estimate=estimate,
        std_error=std_error,
        n_oracle_calls=n_samples,
        n_qubits=0,
        classical_equivalent_samples=n_samples,
        speedup_factor=1.0,
        wall_time_s=wall_time,
        confidence_interval=ci,
        method="classical",
        metadata={"n_samples": n_samples},
    )


# ---------------------------------------------------------------------------
# European option pricing via QMC
# ---------------------------------------------------------------------------


def price_european_qmc(
    spec: EuropeanQMCSpec,
    seed: int | None = None,
) -> QMCResult:
    """Price a European option using Quantum Monte Carlo.

    Encodes the Black-Scholes payoff distribution as a quantum oracle
    and uses QAE-based integration for quadratic speedup.

    Parameters
    ----------
    spec : EuropeanQMCSpec
        Option specification.
    seed : int | None
        Random seed.

    Returns
    -------
    QMCResult
        Option price estimate with resource analysis.
    """
    # Discretise the log-normal price distribution
    n_points = 2**spec.n_price_qubits

    # Log-normal parameters under risk-neutral measure
    mu_ln = np.log(spec.s0) + (spec.r - 0.5 * spec.sigma**2) * spec.T
    sigma_ln = spec.sigma * np.sqrt(spec.T)

    # Discretise price range: [mu - 4*sigma, mu + 4*sigma] in log-space
    log_low = mu_ln - 4 * sigma_ln
    log_high = mu_ln + 4 * sigma_ln
    log_prices = np.linspace(log_low, log_high, n_points)
    prices = np.exp(log_prices)

    # Log-normal probabilities
    from scipy.stats import norm

    log_probs = norm.pdf(log_prices, loc=mu_ln, scale=sigma_ln)
    probabilities = log_probs / np.sum(log_probs)

    # Payoff function
    if spec.is_call:
        payoffs = np.maximum(prices - spec.k, 0.0)
    else:
        payoffs = np.maximum(spec.k - prices, 0.0)

    # Discount factor
    discount = np.exp(-spec.r * spec.T)
    discounted_payoffs = discount * payoffs

    # Run QMC estimation
    result = quantum_mean_estimation(
        f_values=discounted_payoffs,
        probabilities=probabilities,
        n_precision_qubits=spec.n_precision_qubits,
        seed=seed,
    )

    # Add option-specific metadata
    result.metadata["option_type"] = "call" if spec.is_call else "put"
    result.metadata["spot"] = spec.s0
    result.metadata["strike"] = spec.k
    result.metadata["discount_factor"] = discount
    result.metadata["n_price_points"] = n_points

    return result


def price_european_classical_mc(
    spec: EuropeanQMCSpec,
    n_samples: int = 100000,
    seed: int | None = None,
) -> QMCResult:
    """Price a European option using classical Monte Carlo.

    Baseline comparison for the QMC method.

    Parameters
    ----------
    spec : EuropeanQMCSpec
        Option specification.
    n_samples : int
        Number of MC samples.
    seed : int | None
        Random seed.

    Returns
    -------
    QMCResult
        Classical MC price estimate.
    """
    rng = np.random.default_rng(seed)
    start = time.perf_counter()

    # Simulate GBM terminal prices
    z = rng.standard_normal(n_samples)
    log_s = (
        np.log(spec.s0)
        + (spec.r - 0.5 * spec.sigma**2) * spec.T
        + spec.sigma * np.sqrt(spec.T) * z
    )
    s_t = np.exp(log_s)

    # Payoffs
    if spec.is_call:
        payoffs = np.maximum(s_t - spec.k, 0.0)
    else:
        payoffs = np.maximum(spec.k - s_t, 0.0)

    discount = np.exp(-spec.r * spec.T)
    discounted = discount * payoffs

    estimate = float(np.mean(discounted))
    std_error = float(np.std(discounted, ddof=1) / np.sqrt(n_samples))

    ci = (estimate - 1.96 * std_error, estimate + 1.96 * std_error)
    wall_time = time.perf_counter() - start

    return QMCResult(
        estimate=estimate,
        std_error=std_error,
        n_oracle_calls=n_samples,
        n_qubits=0,
        classical_equivalent_samples=n_samples,
        speedup_factor=1.0,
        wall_time_s=wall_time,
        confidence_interval=ci,
        method="classical",
        metadata={
            "option_type": "call" if spec.is_call else "put",
            "spot": spec.s0,
            "strike": spec.k,
            "n_samples": n_samples,
        },
    )


def compare_qmc_vs_classical(
    spec: EuropeanQMCSpec,
    n_classical_samples: int = 100000,
    seed: int | None = None,
) -> dict[str, Any]:
    """Compare QMC and classical MC for European option pricing.

    Parameters
    ----------
    spec : EuropeanQMCSpec
        Option specification.
    n_classical_samples : int
        Number of classical MC samples.
    seed : int | None
        Random seed.

    Returns
    -------
    Dict with qmc_result, classical_result, and comparison metrics.
    """
    qmc_result = price_european_qmc(spec, seed=seed)
    classical_result = price_european_classical_mc(
        spec, n_samples=n_classical_samples, seed=seed
    )

    return {
        "qmc": qmc_result,
        "classical": classical_result,
        "qmc_oracle_calls": qmc_result.n_oracle_calls,
        "classical_samples": classical_result.n_oracle_calls,
        "speedup_factor": qmc_result.speedup_factor,
        "qmc_error": qmc_result.std_error,
        "classical_error": classical_result.std_error,
    }


# ---------------------------------------------------------------------------
# Resource estimation
# ---------------------------------------------------------------------------


def estimate_qmc_resources(
    n_price_qubits: int = 4,
    n_precision_qubits: int = 6,
    n_assets: int = 1,
    surface_code_distance: int = 17,
    physical_error_rate: float = 1e-3,
) -> QMCResourceEstimate:
    """Estimate quantum resources for QMC integration.

    Calculates logical qubit count, T-gate count, and surface code
    overhead for a given problem size.

    Parameters
    ----------
    n_price_qubits : int
        Qubits per asset for price discretisation.
    n_precision_qubits : int
        Qubits for QAE precision.
    n_assets : int
        Number of assets (1 for single-asset, >1 for multi-asset).
    surface_code_distance : int
        Surface code distance for physical qubit estimation.
    physical_error_rate : float
        Physical error rate for surface code overhead calculation.

    Returns
    -------
    QMCResourceEstimate
        Detailed resource estimate.
    """
    # Logical qubits:
    # - n_assets * n_price_qubits for price registers
    # - n_precision_qubits for QAE evaluation register
    # - 1 ancilla for payoff rotation
    # - n_assets ancillae for comparators
    # - arithmetic ancillae: ~2 * n_price_qubits for addition/multiplication
    price_register = n_assets * n_price_qubits
    qae_register = n_precision_qubits
    payoff_ancilla = 1
    comparator_ancillae = n_assets
    arithmetic_ancillae = 2 * n_price_qubits * n_assets

    n_logical = (
        price_register
        + qae_register
        + payoff_ancilla
        + comparator_ancillae
        + arithmetic_ancillae
    )

    # T-gate count estimation (based on Chakrabarti et al. 2021):
    # - Arithmetic (addition): ~8n T-gates per n-bit adder
    # - Multiplication: ~8n^2 T-gates per n-bit multiplier
    # - Comparison: ~8n T-gates per comparator
    # - Controlled rotations: ~8 * n_precision T-gates each
    t_addition = 8 * n_price_qubits * n_assets
    t_multiplication = 8 * n_price_qubits**2 * n_assets
    t_comparison = 8 * n_price_qubits * n_assets
    t_rotations = 8 * n_precision_qubits

    # Per Grover iteration: 2x oracle + 2x diffusion
    t_per_oracle = t_addition + t_multiplication + t_comparison + t_rotations
    # QAE uses O(2^n_precision) Grover iterations total
    n_grover_iterations = 2**n_precision_qubits
    t_gate_count = t_per_oracle * n_grover_iterations

    # T-depth: parallelise where possible
    # Roughly T_count / n_logical (limited parallelism)
    t_depth = max(1, t_gate_count // max(n_logical, 1))

    # Circuit depth: each T-gate ~ 1 layer, plus Clifford overhead ~2x
    circuit_depth = t_depth * 3

    # Surface code physical qubit overhead:
    # Each logical qubit needs ~2 * d^2 physical qubits
    d = surface_code_distance
    physical_per_logical = 2 * d**2
    # Add magic state distillation factory: ~15 * d^2
    distillation_overhead = 15 * d**2
    n_physical = n_logical * physical_per_logical + distillation_overhead

    # Oracle calls
    n_oracle_calls = 2**n_precision_qubits

    # Classical equivalent: for same precision epsilon ~ pi/2^n,
    # classical needs O(1/epsilon^2) samples
    epsilon = np.pi / (2**n_precision_qubits)
    classical_equiv = max(1, int(np.ceil(1.0 / epsilon**2)))

    # Break-even analysis: quantum beats classical when
    # T_gate_count * t_gate_time < classical_samples * sample_time
    # Assuming t_gate_time ~ 1 us, sample_time ~ 1 ns
    # Break-even at: T_count * 1e-6 < N_classical * 1e-9
    # => N_classical > T_count * 1e3
    # => 1/epsilon^2 > T_count * 1e3
    # => epsilon < 1 / sqrt(T_count * 1e3)
    break_even_epsilon = 1.0 / np.sqrt(max(t_per_oracle * 1e3, 1.0))

    desc = (
        f"{n_assets}-asset option, {n_price_qubits} price qubits, "
        f"{n_precision_qubits} precision qubits"
    )

    return QMCResourceEstimate(
        n_logical_qubits=n_logical,
        t_gate_count=t_gate_count,
        t_depth=t_depth,
        n_physical_qubits=n_physical,
        circuit_depth=circuit_depth,
        n_oracle_calls=n_oracle_calls,
        classical_samples_equivalent=classical_equiv,
        break_even_epsilon=break_even_epsilon,
        problem_description=desc,
    )


def break_even_analysis(
    n_price_qubits_range: list[int] | None = None,
    n_precision_range: list[int] | None = None,
    n_assets_range: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Analyse when quantum Monte Carlo breaks even with classical.

    Sweeps across problem sizes and precision levels to determine
    the cross-over point where QMC becomes advantageous.

    Parameters
    ----------
    n_price_qubits_range : list[int] | None
        Range of price qubits to test. Defaults to [4, 6, 8, 10].
    n_precision_range : list[int] | None
        Range of precision qubits to test. Defaults to [6, 8, 10, 12].
    n_assets_range : list[int] | None
        Range of asset counts. Defaults to [1, 2, 5].

    Returns
    -------
    List of dicts with resource estimates and break-even analysis.
    """
    if n_price_qubits_range is None:
        n_price_qubits_range = [4, 6, 8, 10]
    if n_precision_range is None:
        n_precision_range = [6, 8, 10, 12]
    if n_assets_range is None:
        n_assets_range = [1, 2, 5]

    results = []
    for n_assets in n_assets_range:
        for n_price in n_price_qubits_range:
            for n_prec in n_precision_range:
                resource = estimate_qmc_resources(
                    n_price_qubits=n_price,
                    n_precision_qubits=n_prec,
                    n_assets=n_assets,
                )

                epsilon = np.pi / (2**n_prec)
                classical_cost = 1.0 / epsilon**2
                quantum_cost = resource.t_gate_count

                # Quantum advantage ratio
                advantage_ratio = classical_cost / max(quantum_cost, 1)

                results.append({
                    "n_assets": n_assets,
                    "n_price_qubits": n_price,
                    "n_precision_qubits": n_prec,
                    "n_logical_qubits": resource.n_logical_qubits,
                    "n_physical_qubits": resource.n_physical_qubits,
                    "t_gate_count": resource.t_gate_count,
                    "classical_equivalent": resource.classical_samples_equivalent,
                    "break_even_epsilon": resource.break_even_epsilon,
                    "advantage_ratio": advantage_ratio,
                    "quantum_advantage": advantage_ratio > 1.0,
                })

    return results


def resource_table(
    n_assets: int = 1,
    n_price_qubits: int = 4,
    precision_range: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Generate a resource table for QMC at varying precision levels.

    Parameters
    ----------
    n_assets : int
        Number of assets.
    n_price_qubits : int
        Qubits for price discretisation.
    precision_range : list[int] | None
        Precision qubit counts. Defaults to [4, 6, 8, 10, 12].

    Returns
    -------
    List of dicts, one per precision level.
    """
    if precision_range is None:
        precision_range = [4, 6, 8, 10, 12]

    table = []
    for n_prec in precision_range:
        res = estimate_qmc_resources(
            n_price_qubits=n_price_qubits,
            n_precision_qubits=n_prec,
            n_assets=n_assets,
        )
        table.append({
            "precision_qubits": n_prec,
            "epsilon": np.pi / (2**n_prec),
            "logical_qubits": res.n_logical_qubits,
            "physical_qubits": res.n_physical_qubits,
            "t_gates": res.t_gate_count,
            "t_depth": res.t_depth,
            "oracle_calls": res.n_oracle_calls,
            "classical_equiv": res.classical_samples_equivalent,
            "circuit_depth": res.circuit_depth,
        })

    return table
