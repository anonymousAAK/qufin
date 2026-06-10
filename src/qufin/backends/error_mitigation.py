"""Error mitigation strategies for noisy quantum backends.

Implements Zero-Noise Extrapolation (ZNE), Twirled Readout Error
eXtinction (TREX), measurement error mitigation via matrix inversion,
Probabilistic Error Cancellation (PEC), and Clifford Data Regression (CDR).

These strategies improve the accuracy of expectation values and
probability distributions from noisy quantum hardware without
requiring additional qubits.

References
----------
Temme, Bravyi, Gambetta, "Error Mitigation for Short-Depth Quantum
  Circuits", PRL 119:180509 (2017) — ZNE & PEC.
van den Berg et al., "Probabilistic Error Cancellation with Sparse
  Pauli-Lindblad Models on Noisy Quantum Processors" (2023) — TREX.
Bravyi, Gambetta, et al., "Mitigating measurement errors in multiqubit
  experiments", PRA 103:042605 (2021) — M3/matrix mitigation.
Czarnik, Arrasmith, Coles, Cincio, "Error mitigation with Clifford
  quantum-circuit data", Quantum 5, 592 (2021) — CDR.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from qufin.backends.base import Backend


@dataclass
class MitigationResult:
    """Result after applying error mitigation.

    Parameters
    ----------
    raw_counts : dict[str, int]
        Original noisy measurement counts.
    mitigated_counts : dict[str, int]
        Counts after mitigation (may contain non-integer values rounded).
    mitigated_probs : dict[str, float]
        Mitigated probability distribution.
    method : str
        Mitigation method used.
    metadata : dict
        Additional info (e.g., calibration data, extrapolation coefficients).
    """

    raw_counts: dict[str, int]
    mitigated_counts: dict[str, int]
    mitigated_probs: dict[str, float]
    method: str = ""
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


# ---------------------------------------------------------------------------
# Zero-Noise Extrapolation (ZNE)
# ---------------------------------------------------------------------------


def zne_extrapolate(
    circuit: Any,
    backend: Backend,
    scale_factors: list[float] | None = None,
    shots: int = 4096,
    observable_fn: Any = None,
) -> dict[str, Any]:
    """Zero-Noise Extrapolation via unitary folding.

    Amplifies noise by repeating (folding) the circuit at multiple
    scale factors, measures the observable at each, and extrapolates
    to the zero-noise limit using Richardson extrapolation.

    Parameters
    ----------
    circuit : QuantumCircuit
        The circuit to mitigate. Must NOT include measurements (added here).
    backend : Backend
        Noisy backend to execute on.
    scale_factors : list[float]
        Noise amplification factors (e.g., [1, 3, 5]). Each must be an
        odd integer. Default: [1, 3, 5].
    shots : int
        Shots per execution.
    observable_fn : callable
        Function(counts_dict, shots) -> float that computes the observable
        from measurement counts. Default: probability of the all-zeros state.

    Returns
    -------
    Dict with keys: mitigated_value, raw_values, scale_factors, coefficients.
    """

    if scale_factors is None:
        scale_factors = [1, 3, 5]

    if observable_fn is None:

        def observable_fn(counts: dict[str, int], n_shots: int) -> float:
            n_qubits = len(next(iter(counts)))
            zero_state = "0" * n_qubits
            return counts.get(zero_state, 0) / n_shots

    raw_values = []
    for sf in scale_factors:
        folded = _fold_circuit(circuit, sf)
        # Add measurements
        meas_circ = folded.copy()
        meas_circ.measure_all()
        result = backend.run(meas_circ, shots=shots)
        val = observable_fn(result.counts, shots)
        raw_values.append(val)

    # Richardson extrapolation: fit polynomial of degree len(scale_factors)-1
    # and evaluate at scale_factor = 0
    coeffs = _richardson_coefficients(scale_factors)
    mitigated = float(np.dot(coeffs, raw_values))

    return {
        "mitigated_value": mitigated,
        "raw_values": raw_values,
        "scale_factors": scale_factors,
        "coefficients": coeffs.tolist(),
    }


def _fold_circuit(circuit: Any, scale_factor: int) -> Any:
    """Fold a circuit to amplify noise by scale_factor.

    For odd integer scale_factor n, the folded circuit is:
        C (C^dag C)^{(n-1)/2}
    which is logically equivalent to C but has n times the depth.
    """
    if scale_factor == 1:
        return circuit.copy()

    assert scale_factor % 2 == 1 and scale_factor >= 1, (
        f"Scale factor must be a positive odd integer, got {scale_factor}"
    )

    folded = circuit.copy()
    n_folds = (scale_factor - 1) // 2
    for _ in range(n_folds):
        folded.compose(circuit.inverse(), inplace=True)
        folded.compose(circuit, inplace=True)

    return folded


def _richardson_coefficients(scale_factors: list[float]) -> NDArray[np.float64]:
    """Compute Richardson extrapolation coefficients.

    Find weights w_i such that sum(w_i * f(c_i)) ≈ f(0),
    where c_i are the scale factors. Uses Lagrange interpolation
    evaluated at 0.
    """
    n = len(scale_factors)
    c = np.array(scale_factors, dtype=np.float64)
    w = np.zeros(n)
    for i in range(n):
        num = 1.0
        den = 1.0
        for j in range(n):
            if j != i:
                num *= -c[j]
                den *= c[i] - c[j]
        w[i] = num / den
    return w


# ---------------------------------------------------------------------------
# Measurement Error Mitigation (matrix inversion)
# ---------------------------------------------------------------------------


def calibrate_readout(
    n_qubits: int,
    backend: Backend,
    shots: int = 8192,
) -> NDArray[np.float64]:
    """Calibrate the readout error matrix for n_qubits.

    Prepares each computational basis state and measures to build
    the 2^n x 2^n calibration matrix M where M[i,j] = P(measure i | prepared j).

    Parameters
    ----------
    n_qubits : int
        Number of qubits (max ~6 for full matrix; beyond that use tensored).
    backend : Backend
        Backend to calibrate on.
    shots : int
        Shots per calibration circuit.

    Returns
    -------
    Calibration matrix M of shape (2^n, 2^n).
    """
    from qiskit.circuit import QuantumCircuit

    dim = 2**n_qubits
    cal_matrix = np.zeros((dim, dim), dtype=np.float64)

    for state_idx in range(dim):
        qc = QuantumCircuit(n_qubits, n_qubits)
        # Prepare |state_idx> by flipping qubits that should be 1
        bits = format(state_idx, f"0{n_qubits}b")
        for q, bit in enumerate(reversed(bits)):
            if bit == "1":
                qc.x(q)
        qc.measure(range(n_qubits), range(n_qubits))

        result = backend.run(qc, shots=shots)
        for bitstring, count in result.counts.items():
            measured_idx = int(bitstring, 2)
            cal_matrix[measured_idx, state_idx] = count / shots

    return cal_matrix


def mitigate_readout(
    counts: dict[str, int],
    cal_matrix: NDArray[np.float64],
    shots: int,
    method: str = "pseudo_inverse",
) -> MitigationResult:
    """Apply readout error mitigation using the calibration matrix.

    Parameters
    ----------
    counts : dict[str, int]
        Raw measurement counts.
    cal_matrix : NDArray
        Calibration matrix from calibrate_readout().
    shots : int
        Total number of shots.
    method : str
        "pseudo_inverse" (default) or "least_squares".

    Returns
    -------
    MitigationResult with corrected counts.
    """
    n_qubits = int(np.log2(cal_matrix.shape[0]))
    dim = cal_matrix.shape[0]

    # Build raw probability vector
    raw_probs = np.zeros(dim)
    for bitstring, count in counts.items():
        idx = int(bitstring, 2)
        raw_probs[idx] = count / shots

    # Invert the calibration matrix
    if method == "pseudo_inverse":
        cal_inv = np.linalg.pinv(cal_matrix)
        corrected = cal_inv @ raw_probs
    else:
        # Constrained least squares: minimize ||M*x - raw||^2 s.t. x>=0, sum=1
        from scipy.optimize import minimize as scipy_minimize

        def objective(x):
            return np.sum((cal_matrix @ x - raw_probs) ** 2)

        x0 = raw_probs.copy()
        bounds = [(0, 1)] * dim
        constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1.0}]
        res = scipy_minimize(objective, x0, bounds=bounds, constraints=constraints)
        corrected = res.x

    # Clip negative probabilities and renormalize
    corrected = np.maximum(corrected, 0)
    total = corrected.sum()
    if total > 0:
        corrected /= total

    # Convert back to counts
    mitigated_counts = {}
    mitigated_probs = {}
    for idx in range(dim):
        if corrected[idx] > 1e-10:
            bs = format(idx, f"0{n_qubits}b")
            mitigated_counts[bs] = round(corrected[idx] * shots)
            mitigated_probs[bs] = float(corrected[idx])

    return MitigationResult(
        raw_counts=counts,
        mitigated_counts=mitigated_counts,
        mitigated_probs=mitigated_probs,
        method=f"readout_{method}",
        metadata={"cal_matrix_cond": float(np.linalg.cond(cal_matrix))},
    )


# ---------------------------------------------------------------------------
# TREX (Twirled Readout Error eXtinction)
# ---------------------------------------------------------------------------


def trex_mitigate(
    circuit: Any,
    backend: Backend,
    n_twirls: int = 10,
    shots_per_twirl: int = 1024,
) -> MitigationResult:
    """Twirled Readout Error eXtinction (TREX).

    Randomizes measurement errors by applying random X gates before
    measurement and flipping the corresponding classical bits in
    post-processing. Averages over multiple twirls to suppress
    correlated readout errors.

    Parameters
    ----------
    circuit : QuantumCircuit
        Circuit WITHOUT final measurements.
    backend : Backend
        Backend to execute on.
    n_twirls : int
        Number of random twirl patterns to average over.
    shots_per_twirl : int
        Shots per twirl execution.

    Returns
    -------
    MitigationResult with TREX-corrected counts.
    """

    n = circuit.num_qubits
    rng = np.random.default_rng(42)
    aggregated_counts: dict[str, int] = {}

    for _ in range(n_twirls):
        # Random bit-flip pattern
        flip_pattern = rng.integers(0, 2, size=n)

        # Build twirled circuit
        twirled = circuit.copy()
        for q in range(n):
            if flip_pattern[q]:
                twirled.x(q)
        twirled.measure_all()

        result = backend.run(twirled, shots=shots_per_twirl)

        # Undo the classical bit flips
        for bitstring, count in result.counts.items():
            bits = list(bitstring)
            # Qiskit bitstring is big-endian: bit[0] = qubit[n-1]
            for q in range(n):
                bit_idx = n - 1 - q
                if flip_pattern[q]:
                    bits[bit_idx] = "0" if bits[bit_idx] == "1" else "1"
            corrected_bs = "".join(bits)
            aggregated_counts[corrected_bs] = aggregated_counts.get(corrected_bs, 0) + count

    total_shots = n_twirls * shots_per_twirl
    mitigated_probs = {k: v / total_shots for k, v in aggregated_counts.items()}

    return MitigationResult(
        raw_counts=aggregated_counts,
        mitigated_counts=aggregated_counts,
        mitigated_probs=mitigated_probs,
        method="trex",
        metadata={"n_twirls": n_twirls, "shots_per_twirl": shots_per_twirl},
    )


# ---------------------------------------------------------------------------
# Probabilistic Error Cancellation (PEC)
# ---------------------------------------------------------------------------


@dataclass
class PECConfig:
    """Configuration for Probabilistic Error Cancellation.

    Parameters
    ----------
    n_samples : int
        Number of Monte Carlo PEC samples.
    noise_model_type : str
        Type of noise model: "depolarizing", "pauli_twirl", or "custom".
    max_overhead : float
        Maximum allowed sampling overhead (gamma). If the estimated
        overhead exceeds this, a warning is issued.
    seed : int | None
        Random seed for reproducibility.
    """

    n_samples: int = 1000
    noise_model_type: str = "depolarizing"
    max_overhead: float = 100.0
    seed: int | None = 42


def characterize_noise_channel(
    gate_name: str,
    backend: Backend,
    n_qubits: int = 1,
    shots: int = 8192,
) -> NDArray[np.float64]:
    """Characterize a noisy gate channel via Pauli twirling tomography.

    Approximates the Pauli transfer matrix (PTM) of the noisy
    implementation of a gate by preparing Pauli eigenstates, applying
    the gate, and measuring in the Pauli basis.

    For a single-qubit channel, the PTM is a 4x4 real matrix acting
    on the Pauli vector (I, X, Y, Z).

    Parameters
    ----------
    gate_name : str
        Name of the gate to characterize (e.g. "x", "h", "cx").
    backend : Backend
        Noisy backend to run characterization circuits on.
    n_qubits : int
        Number of qubits the gate acts on (1 or 2).
    shots : int
        Shots per tomography circuit.

    Returns
    -------
    NDArray of shape (4^n, 4^n) — the Pauli transfer matrix.
    """
    from qiskit.circuit import QuantumCircuit

    dim = 4**n_qubits
    ptm = np.eye(dim, dtype=np.float64)

    # For single-qubit: prepare +X, +Y, +Z eigenstates, apply gate,
    # measure in each Pauli basis to reconstruct the channel
    if n_qubits == 1:
        # Preparation circuits for |+>, |+i>, |0> (eigenstates of X, Y, Z)
        prep_labels = ["X", "Y", "Z"]
        for i, prep in enumerate(prep_labels):
            for j, meas in enumerate(prep_labels):
                qc = QuantumCircuit(1, 1)
                # Prepare eigenstate
                if prep == "X":
                    qc.h(0)
                elif prep == "Y":
                    qc.h(0)
                    qc.s(0)
                # Z: no prep needed (|0> is +1 eigenstate)

                # Apply the gate
                if gate_name == "x":
                    qc.x(0)
                elif gate_name == "h":
                    qc.h(0)
                elif gate_name == "z":
                    qc.z(0)
                elif gate_name == "s":
                    qc.s(0)
                elif gate_name == "t":
                    qc.t(0)
                elif gate_name == "id":
                    qc.id(0)
                else:
                    qc.id(0)

                # Measure in Pauli basis
                if meas == "X":
                    qc.h(0)
                elif meas == "Y":
                    qc.sdg(0)
                    qc.h(0)
                # Z: measure in computational basis

                qc.measure(0, 0)
                result = backend.run(qc, shots=shots)
                p0 = result.counts.get("0", 0) / shots
                # Expectation value: <P> = p(0) - p(1)
                exp_val = 2 * p0 - 1
                # PTM entry: row=(meas Pauli idx+1), col=(prep Pauli idx+1)
                ptm[j + 1, i + 1] = exp_val
    else:
        # For multi-qubit gates, return identity PTM as placeholder
        pass

    return ptm


def quasi_probability_decomposition(
    ideal_channel: NDArray[np.float64],
    noisy_channel: NDArray[np.float64],
) -> dict[str, Any]:
    """Compute quasi-probability decomposition of an ideal channel.

    Given the Pauli transfer matrices (PTMs) of the ideal and noisy
    channels, finds coefficients {eta_i} such that:
        ideal = sum_i eta_i * B_i
    where B_i are implementable (noisy) basis operations, and eta_i
    may be negative (quasi-probabilities).

    Parameters
    ----------
    ideal_channel : NDArray
        PTM of the ideal gate, shape (d, d).
    noisy_channel : NDArray
        PTM of the noisy gate, shape (d, d).

    Returns
    -------
    Dict with keys:
        - coefficients: list of quasi-probability weights
        - gamma: sampling overhead (1-norm of coefficients)
        - basis_labels: list of basis operation labels
    """
    dim = ideal_channel.shape[0]

    # Compute correction matrix: ideal = correction @ noisy
    # correction = ideal @ noisy^{-1}
    try:
        noisy_inv = np.linalg.inv(noisy_channel)
    except np.linalg.LinAlgError:
        noisy_inv = np.linalg.pinv(noisy_channel)

    correction = ideal_channel @ noisy_inv

    # Decompose correction into Pauli basis for single-qubit case
    # For a 4x4 PTM, decompose into {I, X, Y, Z} Pauli channels
    pauli_labels = ["I", "X", "Y", "Z"]
    if dim == 4:
        # Extract diagonal of correction as Pauli channel coefficients
        coeffs = np.diag(correction).tolist()
    else:
        # For larger channels, use the diagonal as approximation
        coeffs = np.diag(correction).tolist()

    gamma = float(np.sum(np.abs(coeffs)))

    return {
        "coefficients": coeffs,
        "gamma": gamma,
        "basis_labels": pauli_labels[:dim],
    }


def pec_overhead_estimate(
    circuit: Any,
    noise_params: dict[str, float],
) -> float:
    """Estimate the PEC sampling overhead gamma for a circuit.

    The overhead grows exponentially with circuit depth and
    noise strength: gamma ~ (1 + epsilon)^{n_gates} where
    epsilon is the per-gate noise strength.

    Parameters
    ----------
    circuit : QuantumCircuit
        The quantum circuit.
    noise_params : dict
        Noise parameters with keys:
        - "gate_error": per-gate depolarizing error rate
        - "n_gates": (optional) override gate count

    Returns
    -------
    float: estimated sampling overhead gamma >= 1.
    """
    gate_error = noise_params.get("gate_error", 0.01)

    if "n_gates" in noise_params:
        n_gates = noise_params["n_gates"]
    else:
        # Count non-barrier, non-measurement instructions
        n_gates = 0
        if hasattr(circuit, "data"):
            for instruction in circuit.data:
                name = instruction.operation.name
                if name not in ("barrier", "measure"):
                    n_gates += 1
        else:
            n_gates = 10  # fallback

    # For depolarizing noise, gamma per gate = 1 + 2*epsilon
    gamma_per_gate = 1.0 + 2.0 * gate_error
    gamma = gamma_per_gate**n_gates

    return float(gamma)


def pec_mitigate(
    circuit: Any,
    backend: Backend,
    config: PECConfig | None = None,
    observable_fn: Any = None,
    noise_params: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Approximate Probabilistic Error Cancellation (heuristic sign sampling).

    .. warning::
       This is an *approximate, heuristic* routine, **not** a faithful
       implementation of the Temme et al. (2017) PEC protocol. A correct PEC
       implementation requires (a) gate-set tomography to characterise each
       gate's noise channel, and (b) a per-gate quasi-probability decomposition
       whose basis operations are *actually inserted* into sampled circuits.
       This function instead re-executes the same noisy circuit and applies a
       random global sign weighted by the overhead ``gamma``; it reproduces the
       PEC sampling *overhead* and variance behaviour but does **not** invert
       the noise channel, so its estimates are biased. Use it for overhead
       studies and pedagogy, not for production error cancellation. Tracked for
       a real QPD-based implementation.

    Parameters
    ----------
    circuit : QuantumCircuit
        Circuit WITHOUT final measurements.
    backend : Backend
        Noisy backend to execute on.
    config : PECConfig | None
        PEC configuration. Uses defaults if None.
    observable_fn : callable | None
        Function(counts, shots) -> float. Default: P(all-zeros).
    noise_params : dict | None
        Noise parameters for overhead estimation.
        Keys: "gate_error" (float). Default: {"gate_error": 0.01}.

    Returns
    -------
    Dict with keys: mitigated_value, raw_value, gamma, n_samples,
                     sample_values, overhead_estimate.
    """
    import warnings

    warnings.warn(
        "pec_mitigate is an approximate heuristic (random sign weighting), not a "
        "faithful Temme et al. quasi-probability PEC implementation; its estimates "
        "are biased. See the function docstring.",
        UserWarning,
        stacklevel=2,
    )
    if config is None:
        config = PECConfig()
    if noise_params is None:
        noise_params = {"gate_error": 0.01}

    rng = np.random.default_rng(config.seed)

    if observable_fn is None:

        def observable_fn(counts: dict[str, int], n_shots: int) -> float:
            n_qubits = len(next(iter(counts)))
            zero_state = "0" * n_qubits
            return counts.get(zero_state, 0) / n_shots

    # Estimate overhead
    gamma = pec_overhead_estimate(circuit, noise_params)

    # Get raw (unmitigated) result
    meas_circ = circuit.copy()
    meas_circ.measure_all()
    raw_result = backend.run(meas_circ, shots=config.n_samples)
    raw_value = observable_fn(raw_result.counts, config.n_samples)

    # Monte Carlo PEC estimation
    # For each sample, apply a random Pauli sign flip and accumulate
    gate_error = noise_params.get("gate_error", 0.01)
    sample_values = []
    signs = []

    for _ in range(config.n_samples):
        # Each gate independently contributes a sign flip with
        # probability proportional to the quasi-probability weight
        sign = 1.0

        # Count gates in circuit
        n_gates = 0
        if hasattr(circuit, "data"):
            for instruction in circuit.data:
                name = instruction.operation.name
                if name not in ("barrier", "measure"):
                    n_gates += 1

        # For each gate, flip sign with probability p_flip
        # p_flip = epsilon / (1 + 2*epsilon) for depolarizing noise
        p_flip = gate_error / (1.0 + 2.0 * gate_error)
        n_flips = rng.binomial(n_gates, p_flip)
        if n_flips % 2 == 1:
            sign = -1.0

        signs.append(sign)

        # Run the circuit (with noise from the backend)
        result = backend.run(meas_circ, shots=1)
        val = observable_fn(result.counts, 1)
        sample_values.append(sign * gamma * val)

    mitigated_value = float(np.mean(sample_values))

    return {
        "mitigated_value": mitigated_value,
        "raw_value": raw_value,
        "gamma": gamma,
        "n_samples": config.n_samples,
        "sample_values": sample_values,
        "overhead_estimate": gamma,
    }


# ---------------------------------------------------------------------------
# Clifford Data Regression (CDR)
# ---------------------------------------------------------------------------


@dataclass
class CDRConfig:
    """Configuration for Clifford Data Regression.

    Parameters
    ----------
    n_training_circuits : int
        Number of near-Clifford training circuits to generate.
    regression_type : str
        Regression model: "linear" or "ridge".
    ridge_alpha : float
        Regularization strength for ridge regression.
    seed : int | None
        Random seed for reproducibility.
    """

    n_training_circuits: int = 20
    regression_type: str = "linear"
    ridge_alpha: float = 1.0
    seed: int | None = 42


# Map of non-Clifford gates to their nearest Clifford equivalents
_CLIFFORD_MAP: dict[str, str] = {
    "t": "s",
    "tdg": "sdg",
    "rx": "x",
    "ry": "y",
    "rz": "z",
    "u1": "z",
    "u2": "h",
    "u3": "h",
    "p": "z",
}

# Set of Clifford gate names
_CLIFFORD_GATES: set[str] = {
    "id",
    "x",
    "y",
    "z",
    "h",
    "s",
    "sdg",
    "cx",
    "cz",
    "cy",
    "swap",
    "ecr",
    "measure",
    "barrier",
    "reset",
}


def nearest_clifford_gate(gate_name: str) -> str:
    """Map a non-Clifford gate to its nearest Clifford equivalent.

    Parameters
    ----------
    gate_name : str
        Name of the gate (lowercase).

    Returns
    -------
    str: name of the nearest Clifford gate. If the gate is already
         Clifford, returns it unchanged.
    """
    lower = gate_name.lower()
    if lower in _CLIFFORD_GATES:
        return lower
    return _CLIFFORD_MAP.get(lower, "id")


def generate_clifford_circuits(
    circuit: Any,
    n_circuits: int,
    seed: int | None = 42,
) -> list[Any]:
    """Generate near-Clifford training circuits from a target circuit.

    Replaces a random subset of non-Clifford gates with their nearest
    Clifford equivalents to create circuits that are classically
    simulable (or close to it) while remaining structurally similar
    to the target.

    Parameters
    ----------
    circuit : QuantumCircuit
        The target circuit.
    n_circuits : int
        Number of near-Clifford variants to generate.
    seed : int | None
        Random seed for reproducibility.

    Returns
    -------
    List of near-Clifford QuantumCircuit instances.
    """
    from qiskit.circuit import QuantumCircuit

    rng = np.random.default_rng(seed)

    # Find indices of non-Clifford gates
    non_clifford_indices = []
    for idx, instruction in enumerate(circuit.data):
        name = instruction.operation.name.lower()
        if name not in _CLIFFORD_GATES:
            non_clifford_indices.append(idx)

    circuits = []
    for _ in range(n_circuits):
        qc = QuantumCircuit(circuit.num_qubits, circuit.num_clbits)

        # Decide which non-Clifford gates to replace (random subset)
        if non_clifford_indices:
            n_replace = rng.integers(1, max(2, len(non_clifford_indices) + 1))
            replace_set = set(
                rng.choice(
                    non_clifford_indices,
                    size=min(n_replace, len(non_clifford_indices)),
                    replace=False,
                )
            )
        else:
            replace_set = set()

        for idx, instruction in enumerate(circuit.data):
            gate = instruction.operation
            qubits = instruction.qubits
            clbits = instruction.clbits

            qubit_indices = [circuit.qubits.index(q) for q in qubits]
            clbit_indices = [circuit.clbits.index(c) for c in clbits]

            if idx in replace_set:
                # Replace with nearest Clifford
                cliff_name = nearest_clifford_gate(gate.name)
                if cliff_name == "id":
                    qc.id(qubit_indices[0])
                elif cliff_name == "x":
                    qc.x(qubit_indices[0])
                elif cliff_name == "y":
                    qc.y(qubit_indices[0])
                elif cliff_name == "z":
                    qc.z(qubit_indices[0])
                elif cliff_name == "h":
                    qc.h(qubit_indices[0])
                elif cliff_name == "s":
                    qc.s(qubit_indices[0])
                elif cliff_name == "sdg":
                    qc.sdg(qubit_indices[0])
                else:
                    qc.id(qubit_indices[0])
            elif gate.name == "measure":
                qc.measure(qubit_indices, clbit_indices)
            elif gate.name == "barrier":
                qc.barrier(qubit_indices)
            else:
                qc.append(gate, qubit_indices, clbit_indices)

        circuits.append(qc)

    return circuits


def _simulate_ideal(
    circuit: Any,
    backend: Backend,
    shots: int,
    observable_fn: Any,
) -> float:
    """Simulate a circuit on the backend and compute an observable.

    Uses the backend's statevector method if available, otherwise
    runs with shots.

    Parameters
    ----------
    circuit : QuantumCircuit
        Circuit to simulate (without measurements).
    backend : Backend
        Backend (ideally a simulator).
    shots : int
        Number of shots if statevector is not available.
    observable_fn : callable
        Function(counts, shots) -> float.

    Returns
    -------
    float: observable value.
    """
    meas_circ = circuit.copy()
    meas_circ.measure_all()
    result = backend.run(meas_circ, shots=shots)
    return observable_fn(result.counts, shots)


def cdr_mitigate(
    circuit: Any,
    backend: Backend,
    config: CDRConfig | None = None,
    observable_fn: Any = None,
    ideal_backend: Backend | None = None,
    shots: int = 4096,
) -> dict[str, Any]:
    """Clifford Data Regression error mitigation.

    Implements the CDR protocol from Czarnik et al. (2021):
    1. Generate near-Clifford training circuits
    2. Run training circuits on noisy backend and ideal simulator
    3. Fit linear regression: ideal_value = f(noisy_value)
    4. Apply learned correction to the target circuit result

    Parameters
    ----------
    circuit : QuantumCircuit
        Target circuit WITHOUT final measurements.
    backend : Backend
        Noisy backend.
    config : CDRConfig | None
        CDR configuration. Uses defaults if None.
    observable_fn : callable | None
        Function(counts, shots) -> float. Default: P(all-zeros).
    ideal_backend : Backend | None
        Ideal (noiseless) backend for training. If None, uses the
        same backend (useful for MockBackend in testing).
    shots : int
        Shots per circuit execution.

    Returns
    -------
    Dict with keys: mitigated_value, raw_value, slope, intercept,
                     training_noisy, training_ideal, n_training.
    """
    if config is None:
        config = CDRConfig()
    if ideal_backend is None:
        ideal_backend = backend

    if observable_fn is None:

        def observable_fn(counts: dict[str, int], n_shots: int) -> float:
            n_qubits = len(next(iter(counts)))
            zero_state = "0" * n_qubits
            return counts.get(zero_state, 0) / n_shots

    # Generate near-Clifford training circuits
    training_circuits = generate_clifford_circuits(
        circuit,
        config.n_training_circuits,
        seed=config.seed,
    )

    # Collect training data
    training_noisy = []
    training_ideal = []

    for tc in training_circuits:
        # Noisy result
        noisy_val = _simulate_ideal(tc, backend, shots, observable_fn)
        training_noisy.append(noisy_val)

        # Ideal result
        ideal_val = _simulate_ideal(
            tc,
            ideal_backend,
            shots,
            observable_fn,
        )
        training_ideal.append(ideal_val)

    x_train = np.array(training_noisy, dtype=np.float64)
    y_train = np.array(training_ideal, dtype=np.float64)

    # Fit regression model
    if config.regression_type == "ridge":
        slope, intercept = _ridge_regression(
            x_train,
            y_train,
            config.ridge_alpha,
        )
    else:
        slope, intercept = _linear_regression(x_train, y_train)

    # Get raw noisy result for the target circuit
    raw_value = _simulate_ideal(circuit, backend, shots, observable_fn)

    # Apply learned correction
    mitigated_value = float(slope * raw_value + intercept)

    return {
        "mitigated_value": mitigated_value,
        "raw_value": raw_value,
        "slope": float(slope),
        "intercept": float(intercept),
        "training_noisy": training_noisy,
        "training_ideal": training_ideal,
        "n_training": config.n_training_circuits,
    }


def _linear_regression(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
) -> tuple[float, float]:
    """Simple linear regression: y = slope * x + intercept.

    Parameters
    ----------
    x : NDArray
        Input features (1D).
    y : NDArray
        Target values (1D).

    Returns
    -------
    Tuple of (slope, intercept).
    """
    n = len(x)
    if n == 0:
        return 1.0, 0.0

    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x_var = np.sum((x - x_mean) ** 2)

    if x_var < 1e-15:
        # All x values are the same; slope is undefined
        return 1.0, float(y_mean - x_mean)

    slope = float(np.sum((x - x_mean) * (y - y_mean)) / x_var)
    intercept = float(y_mean - slope * x_mean)

    return slope, intercept


def _ridge_regression(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    alpha: float = 1.0,
) -> tuple[float, float]:
    """Ridge regression: y = slope * x + intercept with L2 penalty.

    Parameters
    ----------
    x : NDArray
        Input features (1D).
    y : NDArray
        Target values (1D).
    alpha : float
        Regularization strength.

    Returns
    -------
    Tuple of (slope, intercept).
    """
    n = len(x)
    if n == 0:
        return 1.0, 0.0

    # Add intercept column: X = [[x_1, 1], [x_2, 1], ...]
    x_mat = np.column_stack([x, np.ones(n)])
    # Ridge solution: (X^T X + alpha I)^{-1} X^T y
    # Don't penalize the intercept
    penalty = alpha * np.eye(2)
    penalty[1, 1] = 0.0  # no penalty on intercept

    xtx = x_mat.T @ x_mat + penalty
    xty = x_mat.T @ y

    try:
        beta = np.linalg.solve(xtx, xty)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(xtx, xty, rcond=None)[0]

    return float(beta[0]), float(beta[1])
