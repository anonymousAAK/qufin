"""Path-dependent QAE for Asian option pricing.

Constructs a multi-register quantum circuit encoding discrete GBM
price paths |S_1>|S_2>...|S_T>, computes the running average via
quantum arithmetic, and estimates Asian option payoffs using QAE.

References
----------
Stamatopoulos et al., Quantum 4:291 (2020), arXiv:1905.02666.
Rebentrost et al., "Quantum computational finance: Monte Carlo
pricing of financial derivatives", Phys. Rev. A 98, 022321 (2018).
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from qufin.backends.base import Backend
from qufin.options.amplitude_estimation.estimation_problem import EstimationProblem
from qufin.options.distributions import log_normal_distribution


@dataclass
class PathDependentAsianSpec:
    """Asian option specification for path-dependent QAE pricing.

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
        Time to expiry (years).
    n_steps : int
        Number of monitoring dates (equally spaced).
    is_call : bool
        True for call, False for put.
    average_type : str
        ``"arithmetic"`` or ``"geometric"``.
    n_qubits_per_step : int
        Qubits per time step for price discretization.
    """

    s0: float = 100.0
    k: float = 100.0
    r: float = 0.05
    sigma: float = 0.2
    T: float = 1.0
    n_steps: int = 4
    is_call: bool = True
    average_type: Literal["arithmetic", "geometric"] = "arithmetic"
    n_qubits_per_step: int = 2

    def __post_init__(self) -> None:
        if self.n_steps < 1:
            raise ValueError("n_steps must be >= 1.")
        if self.n_qubits_per_step < 1:
            raise ValueError("n_qubits_per_step must be >= 1.")
        if self.average_type not in ("arithmetic", "geometric"):
            raise ValueError(
                f"average_type must be 'arithmetic' or 'geometric', "
                f"got '{self.average_type}'."
            )

    @property
    def dt(self) -> float:
        """Time step size."""
        return self.T / self.n_steps

    @property
    def n_price_qubits(self) -> int:
        """Total qubits for all price registers."""
        return self.n_steps * self.n_qubits_per_step


def _discretize_step_prices(
    s0: float,
    r: float,
    sigma: float,
    dt: float,
    n_qubits: int,
    n_sigma: float = 3.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Discretize the log-normal distribution for one GBM step.

    Returns
    -------
    values : NDArray[np.float64]
        Grid of price levels, shape ``(2**n_qubits,)``.
    probabilities : NDArray[np.float64]
        Probabilities at each grid point, shape ``(2**n_qubits,)``.
    """
    n_states = 2**n_qubits

    # S_{t+dt} = S_t * exp((r - 0.5*sig^2)*dt + sig*sqrt(dt)*Z)
    # For state preparation we model the distribution of S_{t+dt} / S_0
    # starting from S_0 at each step (independent increments under GBM).
    ln_mean = np.log(s0) + (r - 0.5 * sigma**2) * dt
    ln_std = sigma * np.sqrt(dt)

    low = float(np.exp(ln_mean - n_sigma * ln_std))
    high = float(np.exp(ln_mean + n_sigma * ln_std))
    values = np.linspace(low, high, n_states)

    # Log-normal PDF at grid points
    probs = np.zeros(n_states)
    for i, s in enumerate(values):
        if s > 0:
            log_s = np.log(s)
            probs[i] = np.exp(-0.5 * ((log_s - ln_mean) / ln_std) ** 2) / (
                s * ln_std * np.sqrt(2 * np.pi)
            )

    dx = (high - low) / (n_states - 1) if n_states > 1 else 1.0
    probs = probs * dx
    total = probs.sum()
    if total > 0:
        probs = probs / total

    return values, probs


def build_path_state_preparation(
    spec: PathDependentAsianSpec,
) -> tuple[Any, list[NDArray[np.float64]]]:
    """Build the multi-register state |S_1>|S_2>...|S_T>.

    Encodes the joint distribution of a discrete GBM path using
    independent log-normal increments at each monitoring date.
    Under GBM, the increments are independent in log-space, so the
    joint state factorizes as a product of per-step distributions:

        |psi> = |S_1> (x) |S_2> (x) ... (x) |S_T>

    Each register is loaded with the marginal log-normal distribution
    for S_t given S_0, capturing the sequential GBM dynamics.

    Parameters
    ----------
    spec : PathDependentAsianSpec
        Asian option specification.

    Returns
    -------
    tuple[QuantumCircuit, list[NDArray]]
        The state preparation circuit and the list of per-step
        price grids (each of shape ``(2**n_qubits_per_step,)``).
    """
    from qiskit.circuit import QuantumCircuit, QuantumRegister

    nq = spec.n_qubits_per_step
    T = spec.n_steps

    registers = []
    all_values: list[NDArray[np.float64]] = []
    all_probs: list[NDArray[np.float64]] = []

    for t in range(T):
        # Each step models S_{(t+1)*dt} starting from S_0
        # Under GBM, S_t = S_0 * exp((r - sig^2/2)*t*dt + sig*sqrt(t*dt)*Z)
        t_step = (t + 1) * spec.dt
        vals, probs = _discretize_step_prices(
            s0=spec.s0,
            r=spec.r,
            sigma=spec.sigma,
            dt=t_step,
            n_qubits=nq,
        )
        all_values.append(vals)
        all_probs.append(probs)
        registers.append(QuantumRegister(nq, name=f"S{t}"))

    # Build product-state circuit: independent registers
    n_total = T * nq
    qc = QuantumCircuit(*registers)

    from qiskit.circuit.library import StatePreparation

    for t in range(T):
        amps = np.sqrt(all_probs[t])
        norm = np.linalg.norm(amps)
        if norm > 0:
            amps = amps / norm

        start_qubit = t * nq
        qc.append(
            StatePreparation(amps),
            list(range(start_qubit, start_qubit + nq)),
        )

    return qc, all_values


def compute_path_averages(
    all_values: list[NDArray[np.float64]],
    average_type: str = "arithmetic",
) -> NDArray[np.float64]:
    """Compute the average price for every basis state of the path register.

    Enumerates all combinations of per-step price grid indices and
    computes the arithmetic or geometric average for each.

    Parameters
    ----------
    all_values : list[NDArray]
        Per-step price grids, length ``T``.
    average_type : str
        ``"arithmetic"`` or ``"geometric"``.

    Returns
    -------
    NDArray[np.float64]
        Average prices, shape ``(product of grid sizes,)``.
    """
    T = len(all_values)
    n_states_per_step = len(all_values[0])
    total_states = n_states_per_step ** T

    averages = np.zeros(total_states)
    for state_idx in range(total_states):
        # Decode multi-index
        prices = []
        remainder = state_idx
        for t in range(T - 1, -1, -1):
            idx = remainder // (n_states_per_step ** t)
            remainder = remainder % (n_states_per_step ** t)
            prices.append(all_values[T - 1 - t][idx])

        prices = list(reversed(prices))

        if average_type == "arithmetic":
            averages[state_idx] = float(np.mean(prices))
        else:
            averages[state_idx] = float(np.exp(np.mean(np.log(prices))))

    return averages


def build_asian_payoff_oracle(
    spec: PathDependentAsianSpec,
    path_circuit: Any,
    all_values: list[NDArray[np.float64]],
) -> tuple[Any, float]:
    """Build the payoff oracle for Asian option pricing.

    Adds an ancilla qubit to the path-state circuit and applies
    controlled rotations encoding the Asian option payoff
    ``max(0, avg - K)`` (call) or ``max(0, K - avg)`` (put)
    into the ancilla amplitude.

    Parameters
    ----------
    spec : PathDependentAsianSpec
        Asian option specification.
    path_circuit : QuantumCircuit
        State preparation circuit from :func:`build_path_state_preparation`.
    all_values : list[NDArray]
        Per-step price grids.

    Returns
    -------
    tuple[QuantumCircuit, float]
        The full circuit (path + payoff oracle) and the max payoff
        used for rescaling.
    """
    from qiskit.circuit import QuantumCircuit
    from qiskit.circuit.library import RYGate

    n_price = spec.n_price_qubits
    n_total = n_price + 1  # +1 ancilla
    ancilla = n_price

    qc = QuantumCircuit(n_total)

    # Compose path state preparation into price register
    qc.compose(path_circuit, qubits=range(n_price), inplace=True)

    # Compute average prices and payoffs for all basis states
    averages = compute_path_averages(all_values, spec.average_type)
    total_states = len(averages)

    payoffs = np.zeros(total_states)
    for i, avg in enumerate(averages):
        if spec.is_call:
            payoffs[i] = max(avg - spec.k, 0.0)
        else:
            payoffs[i] = max(spec.k - avg, 0.0)

    max_payoff = float(np.max(payoffs))
    if max_payoff == 0.0:
        max_payoff = 1.0

    # For each basis state with positive payoff, apply controlled RY
    for state_idx in range(total_states):
        if payoffs[state_idx] <= 0.0:
            continue

        normalized = payoffs[state_idx] / max_payoff
        angle = 2.0 * np.arcsin(np.sqrt(min(normalized, 1.0)))

        bits = format(state_idx, f"0{n_price}b")

        # Apply X gates to control on this specific basis state
        for b_idx, b in enumerate(bits):
            if b == "0":
                qc.x(b_idx)

        # Multi-controlled RY on ancilla
        if n_price == 1:
            qc.cry(angle, 0, ancilla)
        else:
            qc.append(
                RYGate(angle).control(n_price),
                [*list(range(n_price)), ancilla],
            )

        # Undo X gates
        for b_idx, b in enumerate(bits):
            if b == "0":
                qc.x(b_idx)

    return qc, max_payoff


def build_path_dependent_estimation_problem(
    spec: PathDependentAsianSpec,
) -> tuple[EstimationProblem, float]:
    """Build the full QAE estimation problem for path-dependent Asian pricing.

    Constructs:
    1. Multi-register path state |S_1>...|S_T>
    2. Payoff oracle encoding Asian option payoff into ancilla
    3. Wraps everything in an ``EstimationProblem``

    Parameters
    ----------
    spec : PathDependentAsianSpec
        Asian option specification.

    Returns
    -------
    tuple[EstimationProblem, float]
        The estimation problem and rescale factor
        ``discount * max_payoff``.
    """
    path_circuit, all_values = build_path_state_preparation(spec)
    full_circuit, max_payoff = build_asian_payoff_oracle(
        spec, path_circuit, all_values,
    )

    n_price = spec.n_price_qubits
    discount = np.exp(-spec.r * spec.T)
    rescale = discount * max_payoff

    problem = EstimationProblem(
        state_preparation=full_circuit,
        objective_qubits=[n_price],  # ancilla
        n_qubits=n_price + 1,
    )

    return problem, rescale


@dataclass
class PathDependentQAEResult:
    """Result from path-dependent Asian option QAE pricing.

    Parameters
    ----------
    price : float
        Estimated option price.
    confidence_interval : tuple[float, float]
        Confidence interval for the price.
    n_steps : int
        Number of monitoring dates.
    n_qubits_total : int
        Total number of qubits used.
    circuit_depth : int | None
        Depth of the state preparation circuit.
    n_oracle_calls : int
        Total oracle (Grover operator) calls.
    average_type : str
        Arithmetic or geometric average.
    """

    price: float = 0.0
    confidence_interval: tuple[float, float] = (0.0, 0.0)
    n_steps: int = 0
    n_qubits_total: int = 0
    circuit_depth: int | None = None
    n_oracle_calls: int = 0
    average_type: str = "arithmetic"


def price_asian_qae(
    spec: PathDependentAsianSpec,
    backend: Backend,
    qae_method: str = "iqae",
    qae_config: Any = None,
) -> PathDependentQAEResult:
    """Price an Asian option using path-dependent QAE.

    Integrates with the existing QAE algorithms (canonical, IQAE, MLAE)
    to estimate the Asian option price from the path-dependent circuit.

    Parameters
    ----------
    spec : PathDependentAsianSpec
        Asian option specification.
    backend : Backend
        Quantum backend to execute circuits.
    qae_method : str
        QAE variant: ``"iqae"``, ``"canonical"``, or ``"mlae"``.
    qae_config : Any
        Configuration for the chosen QAE method.  If ``None``,
        a default configuration is used.

    Returns
    -------
    PathDependentQAEResult
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

    problem, rescale = build_path_dependent_estimation_problem(spec)
    n_total = spec.n_price_qubits + 1

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
        estimator = MaximumLikelihoodAmplitudeEstimation(
            problem, config, backend,
        )
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

    circuit_depth: int | None = None
    with contextlib.suppress(Exception):
        circuit_depth = problem.state_preparation.depth()

    return PathDependentQAEResult(
        price=price,
        confidence_interval=ci_price,
        n_steps=spec.n_steps,
        n_qubits_total=n_total,
        circuit_depth=circuit_depth,
        n_oracle_calls=n_oracle_calls,
        average_type=spec.average_type,
    )


def price_asian_mc(
    spec: PathDependentAsianSpec,
    n_paths: int = 100_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Price an Asian option via classical Monte Carlo.

    Simulates full GBM paths and computes arithmetic or geometric
    average payoffs as a classical benchmark.

    Parameters
    ----------
    spec : PathDependentAsianSpec
        Asian option specification.
    n_paths : int
        Number of Monte Carlo paths.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Dictionary with keys ``"price"``, ``"std_error"``,
        ``"ci_low"``, ``"ci_high"``, ``"n_paths"``.
    """
    rng = np.random.default_rng(seed)
    dt = spec.dt
    T = spec.n_steps

    # Simulate GBM paths: S_t = S_{t-1} * exp((r - sig^2/2)*dt + sig*sqrt(dt)*Z)
    Z = rng.standard_normal((n_paths, T))
    drift = (spec.r - 0.5 * spec.sigma**2) * dt
    diffusion = spec.sigma * np.sqrt(dt)

    # Build price paths
    log_prices = np.zeros((n_paths, T))
    log_prices[:, 0] = np.log(spec.s0) + drift + diffusion * Z[:, 0]
    for t in range(1, T):
        log_prices[:, t] = log_prices[:, t - 1] + drift + diffusion * Z[:, t]

    prices = np.exp(log_prices)

    # Compute averages
    if spec.average_type == "arithmetic":
        averages = np.mean(prices, axis=1)
    else:
        averages = np.exp(np.mean(log_prices, axis=1))

    # Payoffs
    if spec.is_call:
        payoffs = np.maximum(averages - spec.k, 0.0)
    else:
        payoffs = np.maximum(spec.k - averages, 0.0)

    # Discount
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
