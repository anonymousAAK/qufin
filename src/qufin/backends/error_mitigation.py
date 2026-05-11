"""Error mitigation strategies for noisy quantum backends.

Implements Zero-Noise Extrapolation (ZNE), Twirled Readout Error
eXtinction (TREX), and measurement error mitigation via matrix inversion.

These strategies improve the accuracy of expectation values and
probability distributions from noisy quantum hardware without
requiring additional qubits.

References
----------
Temme, Bravyi, Gambetta, "Error Mitigation for Short-Depth Quantum
  Circuits", PRL 119:180509 (2017) — ZNE.
van den Berg et al., "Probabilistic Error Cancellation with Sparse
  Pauli-Lindblad Models on Noisy Quantum Processors" (2023) — TREX.
Bravyi, Gambetta, et al., "Mitigating measurement errors in multiqubit
  experiments", PRA 103:042605 (2021) — M3/matrix mitigation.
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
            aggregated_counts[corrected_bs] = (
                aggregated_counts.get(corrected_bs, 0) + count
            )

    total_shots = n_twirls * shots_per_twirl
    mitigated_probs = {k: v / total_shots for k, v in aggregated_counts.items()}

    return MitigationResult(
        raw_counts=aggregated_counts,
        mitigated_counts=aggregated_counts,
        mitigated_probs=mitigated_probs,
        method="trex",
        metadata={"n_twirls": n_twirls, "shots_per_twirl": shots_per_twirl},
    )
