"""Multi-asset option pricing via Quantum Amplitude Estimation.

Supports basket options (weighted average of assets) and rainbow options
(best-of, worst-of) using correlated log-normal price distributions
loaded into a quantum register and priced via QAE.

References
----------
Stamatopoulos et al., Quantum 4:291 (2020), arXiv:1905.02666.
Woerner & Egger, npj Quantum Information 5:15 (2019), arXiv:1806.06893.
Kaneko et al., "Quantum pricing with a smile" (2020), arXiv:2007.01467.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from qufin.backends.base import Backend
from qufin.options.amplitude_estimation.estimation_problem import EstimationProblem


@dataclass
class MultiAssetSpec:
    """Multi-asset option specification for QAE pricing.

    Parameters
    ----------
    spots : list[float]
        Initial spot prices per asset.
    strikes : float
        Strike price.
    r : float
        Risk-free rate.
    sigma : list[float]
        Volatilities per asset.
    correlation : NDArray[np.float64]
        Correlation matrix of asset returns, shape (N, N).
    T : float
        Time to maturity in years.
    weights : list[float] | None
        Basket weights per asset. If None, equal weights are used.
    payoff_type : str
        One of ``"basket_call"``, ``"basket_put"``,
        ``"best_of_call"``, ``"worst_of_call"``.
    n_qubits_per_asset : int
        Number of qubits used to discretize each asset's price range.
    """

    spots: list[float] = field(default_factory=lambda: [100.0, 100.0])
    strikes: float = 100.0
    r: float = 0.05
    sigma: list[float] = field(default_factory=lambda: [0.2, 0.2])
    correlation: NDArray[np.float64] = field(
        default_factory=lambda: np.eye(2)
    )
    T: float = 1.0
    weights: list[float] | None = None
    payoff_type: str = "basket_call"
    n_qubits_per_asset: int = 3

    def __post_init__(self) -> None:
        n = len(self.spots)
        if len(self.sigma) != n:
            raise ValueError(
                f"Length of sigma ({len(self.sigma)}) must match "
                f"number of assets ({n})."
            )
        if self.correlation.shape != (n, n):
            raise ValueError(
                f"Correlation matrix shape {self.correlation.shape} "
                f"must be ({n}, {n})."
            )
        if self.weights is not None and len(self.weights) != n:
            raise ValueError(
                f"Length of weights ({len(self.weights)}) must match "
                f"number of assets ({n})."
            )
        valid_payoffs = {
            "basket_call", "basket_put", "best_of_call", "worst_of_call",
        }
        if self.payoff_type not in valid_payoffs:
            raise ValueError(
                f"payoff_type must be one of {valid_payoffs}, "
                f"got '{self.payoff_type}'."
            )

    @property
    def n_assets(self) -> int:
        """Number of assets."""
        return len(self.spots)

    @property
    def effective_weights(self) -> list[float]:
        """Basket weights, defaulting to equal weights."""
        if self.weights is not None:
            return self.weights
        n = self.n_assets
        return [1.0 / n] * n


@dataclass
class MultiAssetQAEResult:
    """Result from multi-asset QAE pricing.

    Parameters
    ----------
    price : float
        Estimated option price.
    confidence_interval : tuple[float, float]
        Confidence interval for the price estimate.
    n_assets : int
        Number of underlying assets.
    n_qubits_total : int
        Total number of qubits in the circuit.
    circuit_depth : int | None
        Depth of the state preparation circuit, if available.
    n_oracle_calls : int
        Total number of oracle (Grover operator) calls.
    payoff_type : str
        The payoff type used.
    """

    price: float = 0.0
    confidence_interval: tuple[float, float] = (0.0, 0.0)
    n_assets: int = 0
    n_qubits_total: int = 0
    circuit_depth: int | None = None
    n_oracle_calls: int = 0
    payoff_type: str = "basket_call"


def _discretize_asset_prices(
    s0: float,
    mu: float,
    sigma: float,
    T: float,
    n_qubits: int,
    n_sigma_range: float = 3.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Discretize a log-normal price distribution for one asset.

    Returns
    -------
    values : NDArray[np.float64]
        Grid of asset prices, shape ``(2**n_qubits,)``.
    probabilities : NDArray[np.float64]
        Probability at each grid point, shape ``(2**n_qubits,)``.
    """
    from scipy.stats import lognorm

    n_states = 2**n_qubits
    # Log-normal parameters for S_T = S_0 * exp((mu - 0.5*sig^2)*T + sig*sqrt(T)*Z)
    log_mean = np.log(s0) + (mu - 0.5 * sigma**2) * T
    log_std = sigma * np.sqrt(T)

    low = np.exp(log_mean - n_sigma_range * log_std)
    high = np.exp(log_mean + n_sigma_range * log_std)
    values = np.linspace(low, high, n_states)

    # Probability mass at each grid point
    dx = values[1] - values[0] if n_states > 1 else 1.0
    probs = lognorm.pdf(values, s=log_std, scale=np.exp(log_mean)) * dx
    total = probs.sum()
    if total > 0:
        probs = probs / total
    return values, probs


def build_multi_asset_distribution(
    spec: MultiAssetSpec,
) -> QuantumCircuit:
    """Build a quantum circuit encoding correlated log-normal distributions.

    Uses Cholesky decomposition of the correlation matrix to introduce
    correlations between the asset price registers.

    Each asset is represented by ``n_qubits_per_asset`` qubits, giving a
    total of ``N * n_qubits_per_asset`` qubits for the distribution.

    Parameters
    ----------
    spec : MultiAssetSpec
        Multi-asset option specification.

    Returns
    -------
    QuantumCircuit
        Circuit that prepares the correlated multi-asset distribution.
    """
    from qiskit.circuit import QuantumCircuit, QuantumRegister

    n = spec.n_assets
    nq = spec.n_qubits_per_asset
    n_states = 2**nq

    # Cholesky decomposition for correlations
    L = np.linalg.cholesky(spec.correlation)

    # Compute marginal distributions for each asset
    all_values: list[NDArray[np.float64]] = []
    all_probs: list[NDArray[np.float64]] = []
    for i in range(n):
        vals, probs = _discretize_asset_prices(
            s0=spec.spots[i],
            mu=spec.r,
            sigma=spec.sigma[i],
            T=spec.T,
            n_qubits=nq,
        )
        all_values.append(vals)
        all_probs.append(probs)

    # Build joint probability tensor incorporating correlations.
    # For a simplified NISQ-friendly approach, we construct the full
    # joint amplitude vector over the tensor-product Hilbert space.
    total_qubits = n * nq
    total_states = 2**total_qubits

    # Build correlated joint probability distribution using Cholesky.
    # Map each basis state to a multi-index, compute correlated
    # probability as product of conditionals.
    joint_probs = np.zeros(total_states)

    for state_idx in range(total_states):
        # Decode multi-index: which grid point for each asset
        asset_indices = []
        remainder = state_idx
        for i in range(n - 1, -1, -1):
            asset_indices.insert(0, remainder // (n_states**i))
            remainder = remainder % (n_states**i)

        # Compute correlated log-probability using Cholesky factors.
        # z_i = sum_j L_{ij} * u_j where u_j are uniform quantiles
        # mapped from the grid indices.
        prob = 1.0
        for i in range(n):
            idx = asset_indices[i]
            # Weight the marginal by correlation contribution
            # For the diagonal-dominant case, this is a good approx.
            marginal_prob = all_probs[i][idx]
            # Apply correlation adjustment via Cholesky off-diag terms
            corr_factor = 1.0
            for j in range(i):
                idx_j = asset_indices[j]
                # Shift probability mass based on off-diagonal L entries
                center_j = (n_states - 1) / 2.0
                deviation = (idx_j - center_j) / max(center_j, 1.0)
                center_i = (n_states - 1) / 2.0
                deviation_i = (idx - center_i) / max(center_i, 1.0)
                # Increase probability when both deviate in same direction
                # (positive correlation) or opposite (negative correlation)
                corr_factor *= 1.0 + L[i, j] * deviation * deviation_i
            prob = marginal_prob * max(corr_factor, 0.0)

        joint_probs[state_idx] = prob

    # Normalize
    total = joint_probs.sum()
    if total > 0:
        joint_probs = joint_probs / total

    # Prepare amplitudes (sqrt of probabilities)
    amplitudes = np.sqrt(joint_probs)
    norm = np.linalg.norm(amplitudes)
    if norm > 0:
        amplitudes = amplitudes / norm

    # Build circuit
    registers = []
    for i in range(n):
        registers.append(QuantumRegister(nq, name=f"asset{i}"))

    qc = QuantumCircuit(*registers)
    qc.initialize(amplitudes, range(total_qubits))

    # Store distribution metadata for payoff computation
    qc.metadata = {
        "all_values": all_values,
        "all_probs": all_probs,
        "joint_probs": joint_probs,
        "n_assets": n,
        "n_qubits_per_asset": nq,
    }

    return qc


def _compute_payoff(
    asset_prices: list[float],
    spec: MultiAssetSpec,
) -> float:
    """Compute the option payoff for given asset prices.

    Parameters
    ----------
    asset_prices : list[float]
        Terminal price of each asset.
    spec : MultiAssetSpec
        Multi-asset specification.

    Returns
    -------
    float
        Non-negative payoff.
    """
    weights = spec.effective_weights

    if spec.payoff_type == "basket_call":
        basket_val = sum(w * s for w, s in zip(weights, asset_prices, strict=True))
        return max(basket_val - spec.strikes, 0.0)
    elif spec.payoff_type == "basket_put":
        basket_val = sum(w * s for w, s in zip(weights, asset_prices, strict=True))
        return max(spec.strikes - basket_val, 0.0)
    elif spec.payoff_type == "best_of_call":
        return max(max(asset_prices) - spec.strikes, 0.0)
    elif spec.payoff_type == "worst_of_call":
        return max(min(asset_prices) - spec.strikes, 0.0)
    else:
        raise ValueError(f"Unknown payoff_type: {spec.payoff_type}")


def build_basket_payoff_oracle(
    spec: MultiAssetSpec,
    dist_circuit: QuantumCircuit,
) -> QuantumCircuit:
    """Build the payoff oracle circuit for a multi-asset option.

    Adds an ancilla qubit and applies controlled rotations so that
    the ancilla amplitude encodes the payoff value for each basis state.

    Parameters
    ----------
    spec : MultiAssetSpec
        Multi-asset option specification.
    dist_circuit : QuantumCircuit
        Distribution loading circuit returned by
        :func:`build_multi_asset_distribution`.

    Returns
    -------
    QuantumCircuit
        Full circuit (distribution + payoff oracle) with an ancilla qubit.
    """
    from qiskit.circuit import QuantumCircuit
    from qiskit.circuit.library import RYGate

    metadata = dist_circuit.metadata
    all_values: list[NDArray[np.float64]] = metadata["all_values"]
    n = metadata["n_assets"]
    nq = metadata["n_qubits_per_asset"]
    n_states = 2**nq

    n_price_qubits = n * nq
    n_total = n_price_qubits + 1  # +1 ancilla

    qc = QuantumCircuit(n_total)

    # Compose distribution circuit into the price register
    qc.compose(dist_circuit, qubits=range(n_price_qubits), inplace=True)

    # Compute payoffs for all basis states
    total_states = 2**n_price_qubits
    payoffs = np.zeros(total_states)

    for state_idx in range(total_states):
        # Decode multi-index
        asset_indices = []
        remainder = state_idx
        for i in range(n - 1, -1, -1):
            asset_indices.insert(0, remainder // (n_states**i))
            remainder = remainder % (n_states**i)

        asset_prices = [all_values[i][asset_indices[i]] for i in range(n)]
        payoffs[state_idx] = _compute_payoff(asset_prices, spec)

    max_payoff = np.max(payoffs)
    if max_payoff == 0:
        max_payoff = 1.0

    # For each basis state, apply controlled RY rotation on ancilla
    ancilla = n_price_qubits  # ancilla qubit index
    for state_idx in range(total_states):
        if payoffs[state_idx] <= 0:
            continue

        normalized = payoffs[state_idx] / max_payoff
        angle = 2 * np.arcsin(np.sqrt(min(normalized, 1.0)))

        bits = format(state_idx, f"0{n_price_qubits}b")

        # Apply X gates to control on this specific basis state
        for b_idx, b in enumerate(bits):
            if b == "0":
                qc.x(b_idx)

        # Multi-controlled RY on ancilla
        if n_price_qubits == 1:
            qc.cry(angle, 0, ancilla)
        else:
            qc.append(
                RYGate(angle).control(n_price_qubits),
                [*list(range(n_price_qubits)), ancilla],
            )

        # Undo X gates
        for b_idx, b in enumerate(bits):
            if b == "0":
                qc.x(b_idx)

    # Store max_payoff for rescaling
    qc.metadata = {"max_payoff": max_payoff, **metadata}

    return qc


def build_multi_asset_estimation_problem(
    spec: MultiAssetSpec,
) -> tuple[EstimationProblem, float]:
    """Build the QAE estimation problem for multi-asset option pricing.

    Constructs the full state preparation circuit (distribution loading
    + payoff oracle) and wraps it in an ``EstimationProblem``.

    Parameters
    ----------
    spec : MultiAssetSpec
        Multi-asset option specification.

    Returns
    -------
    tuple[EstimationProblem, float]
        The estimation problem and a rescale factor
        ``discount * max_payoff`` used to convert the QAE amplitude
        estimate back to an option price.
    """
    dist_circuit = build_multi_asset_distribution(spec)
    full_circuit = build_basket_payoff_oracle(spec, dist_circuit)

    max_payoff = full_circuit.metadata["max_payoff"]
    discount = np.exp(-spec.r * spec.T)
    rescale = discount * max_payoff

    n_price_qubits = spec.n_assets * spec.n_qubits_per_asset
    n_total = n_price_qubits + 1
    objective_qubits = [n_price_qubits]  # ancilla

    problem = EstimationProblem(
        state_preparation=full_circuit,
        objective_qubits=objective_qubits,
        n_qubits=n_total,
    )

    return problem, rescale


def price_multi_asset_qae(
    spec: MultiAssetSpec,
    backend: Backend,
    qae_method: str = "iqae",
    qae_config: Any = None,
) -> MultiAssetQAEResult:
    """Price a multi-asset option using Quantum Amplitude Estimation.

    Parameters
    ----------
    spec : MultiAssetSpec
        Multi-asset option specification.
    backend : Backend
        Quantum backend to execute circuits on.
    qae_method : str
        QAE variant: ``"iqae"``, ``"canonical"``, or ``"mlae"``.
    qae_config : Any
        Configuration object for the chosen QAE method. If ``None``,
        default configuration is used.

    Returns
    -------
    MultiAssetQAEResult
        Pricing result with price, confidence interval, and metadata.
    """
    from qufin.options.amplitude_estimation.canonical import (
        CanonicalAmplitudeEstimation,
        CanonicalQAEConfig,
    )
    from qufin.options.amplitude_estimation.iqae import (
        IQAEConfig,
        IterativeAmplitudeEstimation,
    )
    from qufin.options.amplitude_estimation.mlae import (
        MaximumLikelihoodAmplitudeEstimation,
        MLAEConfig,
    )

    problem, rescale = build_multi_asset_estimation_problem(spec)

    n_price_qubits = spec.n_assets * spec.n_qubits_per_asset
    n_total = n_price_qubits + 1

    if qae_method == "iqae":
        config = qae_config if qae_config is not None else IQAEConfig()
        estimator = IterativeAmplitudeEstimation(problem, config, backend)
        result = estimator.estimate()
        estimate = result.estimate
        ci = result.confidence_interval
        n_oracle_calls = result.n_oracle_calls
    elif qae_method == "canonical":
        config = qae_config if qae_config is not None else CanonicalQAEConfig()
        estimator = CanonicalAmplitudeEstimation(problem, config, backend)
        result = estimator.estimate()
        estimate = result.estimate
        ci = result.confidence_interval
        n_oracle_calls = result.n_oracle_calls
    elif qae_method == "mlae":
        config = qae_config if qae_config is not None else MLAEConfig()
        estimator = MaximumLikelihoodAmplitudeEstimation(problem, config, backend)
        result = estimator.estimate()
        estimate = result.estimate
        ci = result.confidence_interval
        n_oracle_calls = result.n_oracle_calls
    else:
        raise ValueError(
            f"Unknown qae_method '{qae_method}'. "
            f"Must be 'iqae', 'canonical', or 'mlae'."
        )

    price = rescale * estimate
    ci_price = (rescale * ci[0], rescale * ci[1])

    # Attempt to get circuit depth
    circuit_depth: int | None = None
    import contextlib
    with contextlib.suppress(Exception):
        circuit_depth = problem.state_preparation.depth()

    return MultiAssetQAEResult(
        price=price,
        confidence_interval=ci_price,
        n_assets=spec.n_assets,
        n_qubits_total=n_total,
        circuit_depth=circuit_depth,
        n_oracle_calls=n_oracle_calls,
        payoff_type=spec.payoff_type,
    )


def price_multi_asset_mc(
    spec: MultiAssetSpec,
    n_paths: int = 100_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Price a multi-asset option via classical Monte Carlo.

    Uses Cholesky decomposition to generate correlated Brownian
    motions for the underlying assets.

    Parameters
    ----------
    spec : MultiAssetSpec
        Multi-asset option specification.
    n_paths : int
        Number of Monte Carlo simulation paths.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Dictionary with keys ``"price"``, ``"std_error"``,
        ``"ci_low"``, ``"ci_high"``, and ``"n_paths"``.
    """
    rng = np.random.default_rng(seed)
    n = spec.n_assets
    weights = spec.effective_weights

    # Cholesky decomposition for correlated normals
    L = np.linalg.cholesky(spec.correlation)

    # Generate independent standard normals: shape (n_paths, n_assets)
    Z = rng.standard_normal((n_paths, n))

    # Correlate: Z_corr = Z @ L^T
    Z_corr = Z @ L.T

    # Simulate terminal prices for each asset under risk-neutral measure
    # S_T = S_0 * exp((r - 0.5*sig^2)*T + sig*sqrt(T)*Z)
    S_T = np.zeros((n_paths, n))
    for i in range(n):
        drift = (spec.r - 0.5 * spec.sigma[i] ** 2) * spec.T
        diffusion = spec.sigma[i] * np.sqrt(spec.T) * Z_corr[:, i]
        S_T[:, i] = spec.spots[i] * np.exp(drift + diffusion)

    # Compute payoffs
    payoffs = np.zeros(n_paths)
    if spec.payoff_type == "basket_call":
        basket = np.zeros(n_paths)
        for i in range(n):
            basket += weights[i] * S_T[:, i]
        payoffs = np.maximum(basket - spec.strikes, 0.0)
    elif spec.payoff_type == "basket_put":
        basket = np.zeros(n_paths)
        for i in range(n):
            basket += weights[i] * S_T[:, i]
        payoffs = np.maximum(spec.strikes - basket, 0.0)
    elif spec.payoff_type == "best_of_call":
        best = np.max(S_T, axis=1)
        payoffs = np.maximum(best - spec.strikes, 0.0)
    elif spec.payoff_type == "worst_of_call":
        worst = np.min(S_T, axis=1)
        payoffs = np.maximum(worst - spec.strikes, 0.0)

    # Discount to present value
    discount = np.exp(-spec.r * spec.T)
    discounted = discount * payoffs

    price = float(np.mean(discounted))
    std_error = float(np.std(discounted, ddof=1) / np.sqrt(n_paths))
    ci_low = price - 1.96 * std_error
    ci_high = price + 1.96 * std_error

    return {
        "price": price,
        "std_error": std_error,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_paths": n_paths,
    }
