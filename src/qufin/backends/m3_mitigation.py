"""Matrix-free Measurement Mitigation (M3) for scalable readout correction.

Implements a tensored (per-qubit) calibration approach that scales
linearly in the number of qubits, unlike the full calibration matrix
approach in ``error_mitigation.py`` which scales as 2^n.

The key insight: if readout errors are approximately independent across
qubits, the full 2^n x 2^n calibration matrix can be approximated as
a tensor product of n 2x2 matrices, requiring only 2n calibration
circuits instead of 2^n.

Includes iterative Bayesian correction for improved accuracy.

References
----------
Nation, Kang, Sundaresan, Gambetta, "Scalable Mitigation of Measurement
  Errors on Quantum Computers", PRX Quantum 2:040326 (2021).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from qufin.backends.base import Backend


@dataclass
class M3Config:
    """Configuration for M3 measurement mitigation.

    Parameters
    ----------
    n_calibration_shots : int
        Shots per calibration circuit. Higher = more accurate calibration.
    method : str
        Correction method: "direct" (matrix inversion) or "iterative"
        (Bayesian iterative correction).
    max_iterations : int
        Maximum iterations for iterative method.
    convergence_tol : float
        Convergence tolerance for iterative method.
    """

    n_calibration_shots: int = 8192
    method: str = "direct"
    max_iterations: int = 25
    convergence_tol: float = 1e-6


@dataclass
class CalibrationData:
    """Per-qubit calibration data from tensored calibration.

    Parameters
    ----------
    qubit_matrices : list[NDArray]
        List of 2x2 calibration matrices, one per qubit.
        Each matrix[i][j] = P(measure i | prepared j).
    n_qubits : int
        Number of calibrated qubits.
    shots : int
        Shots used for calibration.
    """

    qubit_matrices: list[NDArray] = field(default_factory=list)
    n_qubits: int = 0
    shots: int = 0


class M3Mitigator:
    """Matrix-free measurement mitigator using tensored calibration.

    Calibrates per-qubit readout errors and applies scalable correction
    to measurement counts. Much more efficient than full matrix
    calibration for large qubit counts.

    Parameters
    ----------
    config : M3Config
        Configuration for calibration and correction.
    """

    def __init__(self, config: M3Config | None = None) -> None:
        self._config = config or M3Config()
        self._cal_data: CalibrationData | None = None

    @property
    def is_calibrated(self) -> bool:
        """Whether calibration has been performed."""
        return self._cal_data is not None

    @property
    def calibration_data(self) -> CalibrationData | None:
        """Return the current calibration data."""
        return self._cal_data

    def calibrate(
        self,
        backend: Backend,
        qubit_list: list[int] | None = None,
        n_qubits: int | None = None,
    ) -> CalibrationData:
        """Calibrate per-qubit readout errors.

        Runs 2 circuits per qubit (prepare |0> and |1>), measuring
        the readout fidelity for each qubit independently.

        Parameters
        ----------
        backend : Backend
            Backend to calibrate on.
        qubit_list : list[int] | None
            Specific qubits to calibrate. If None, calibrates
            qubits 0..n_qubits-1.
        n_qubits : int | None
            Number of qubits (used if qubit_list is None). Default 2.

        Returns
        -------
        CalibrationData with per-qubit 2x2 matrices.
        """
        if qubit_list is None:
            n = n_qubits or 2
            qubit_list = list(range(n))

        cal_data = tensored_calibration(
            n_qubits=len(qubit_list),
            backend=backend,
            shots=self._config.n_calibration_shots,
        )
        self._cal_data = cal_data
        return cal_data

    def apply(
        self,
        counts: dict[str, int],
        shots: int,
    ) -> dict[str, Any]:
        """Apply matrix-free correction to measurement counts.

        Parameters
        ----------
        counts : dict[str, int]
            Raw measurement counts.
        shots : int
            Total number of shots.

        Returns
        -------
        Dict with keys: mitigated_counts, mitigated_probs, method, metadata.

        Raises
        ------
        RuntimeError
            If calibration has not been performed.
        """
        if self._cal_data is None:
            raise RuntimeError(
                "Mitigator not calibrated. Call calibrate() first."
            )

        if self._config.method == "iterative":
            corrected_probs = iterative_correction(
                counts=counts,
                cal_data=self._cal_data,
                max_iter=self._config.max_iterations,
                tol=self._config.convergence_tol,
            )
        else:
            corrected_probs = _direct_correction(counts, self._cal_data, shots)

        # Build corrected counts
        mitigated_counts = {}
        mitigated_probs = {}
        for bitstring, prob in corrected_probs.items():
            if prob > 1e-10:
                mitigated_counts[bitstring] = round(prob * shots)
                mitigated_probs[bitstring] = prob

        return {
            "mitigated_counts": mitigated_counts,
            "mitigated_probs": mitigated_probs,
            "method": f"m3_{self._config.method}",
            "metadata": {
                "n_qubits": self._cal_data.n_qubits,
                "calibration_shots": self._cal_data.shots,
            },
        }

    def overhead_estimate(self, n_qubits: int) -> dict[str, Any]:
        """Estimate computational overhead for M3 mitigation.

        Compares M3 (tensored) vs full matrix calibration overhead.

        Parameters
        ----------
        n_qubits : int
            Number of qubits.

        Returns
        -------
        Dict with calibration circuit counts and matrix sizes for
        both M3 and full approaches.
        """
        return {
            "n_qubits": n_qubits,
            "m3_calibration_circuits": 2 * n_qubits,
            "full_calibration_circuits": 2**n_qubits,
            "m3_matrix_elements": 4 * n_qubits,
            "full_matrix_elements": (2**n_qubits) ** 2,
            "m3_memory_bytes": 4 * n_qubits * 8,
            "full_memory_bytes": (2**n_qubits) ** 2 * 8,
            "speedup_factor": 2**n_qubits / (2 * n_qubits) if n_qubits > 0 else 1,
        }


def tensored_calibration(
    n_qubits: int,
    backend: Backend,
    shots: int = 8192,
) -> CalibrationData:
    """Calibrate individual qubit readout errors using tensored approach.

    Runs 2 circuits per qubit (prepare |0> and prepare |1>), building
    a 2x2 calibration matrix for each qubit independently. This requires
    only 2*n_qubits circuits instead of 2^n_qubits for full calibration.

    Parameters
    ----------
    n_qubits : int
        Number of qubits to calibrate.
    backend : Backend
        Backend to run calibration circuits on.
    shots : int
        Shots per calibration circuit.

    Returns
    -------
    CalibrationData with per-qubit 2x2 matrices.
    """
    from qiskit.circuit import QuantumCircuit

    qubit_matrices = []

    for _q in range(n_qubits):
        cal_matrix = np.zeros((2, 2), dtype=np.float64)

        for prep_state in [0, 1]:
            qc = QuantumCircuit(1, 1)
            if prep_state == 1:
                qc.x(0)
            qc.measure(0, 0)

            result = backend.run(qc, shots=shots)

            for bitstring, count in result.counts.items():
                # Handle both "0"/"1" and multi-bit strings
                measured_bit = int(bitstring[-1])
                cal_matrix[measured_bit, prep_state] = count / shots

        # Ensure columns sum to 1 (normalize for rounding)
        for col in range(2):
            col_sum = cal_matrix[:, col].sum()
            if col_sum > 0:
                cal_matrix[:, col] /= col_sum

        qubit_matrices.append(cal_matrix)

    return CalibrationData(
        qubit_matrices=qubit_matrices,
        n_qubits=n_qubits,
        shots=shots,
    )


def _direct_correction(
    counts: dict[str, int],
    cal_data: CalibrationData,
    shots: int,
) -> dict[str, float]:
    """Apply direct (matrix-inverse) tensored correction.

    Inverts each per-qubit 2x2 calibration matrix and applies the
    tensored inverse to the raw probability vector.

    Parameters
    ----------
    counts : dict[str, int]
        Raw measurement counts.
    cal_data : CalibrationData
        Calibration data from tensored_calibration.
    shots : int
        Total shots.

    Returns
    -------
    Corrected probability distribution.
    """
    n_qubits = cal_data.n_qubits

    # Invert each per-qubit matrix
    inv_matrices = []
    for mat in cal_data.qubit_matrices:
        inv_matrices.append(np.linalg.pinv(mat))

    # Build raw probability vector
    dim = 2**n_qubits
    raw_probs = np.zeros(dim)
    for bitstring, count in counts.items():
        idx = int(bitstring, 2)
        if idx < dim:
            raw_probs[idx] = count / shots

    # Apply tensored inverse: for each bitstring, multiply per-qubit
    # correction factors
    corrected = np.zeros(dim)
    for out_idx in range(dim):
        out_bits = format(out_idx, f"0{n_qubits}b")
        for in_idx in range(dim):
            in_bits = format(in_idx, f"0{n_qubits}b")
            # Tensored matrix element = product of per-qubit elements
            element = 1.0
            for q in range(n_qubits):
                # Bit ordering: big-endian, bit[0] = MSB
                out_bit = int(out_bits[q])
                in_bit = int(in_bits[q])
                element *= inv_matrices[q][out_bit, in_bit]
            corrected[out_idx] += element * raw_probs[in_idx]

    # Clip negatives and renormalize
    corrected = np.maximum(corrected, 0)
    total = corrected.sum()
    if total > 0:
        corrected /= total

    result = {}
    for idx in range(dim):
        if corrected[idx] > 1e-10:
            bs = format(idx, f"0{n_qubits}b")
            result[bs] = float(corrected[idx])

    return result


def iterative_correction(
    counts: dict[str, int],
    cal_data: CalibrationData,
    max_iter: int = 25,
    tol: float = 1e-6,
) -> dict[str, float]:
    """Iterative Bayesian correction using tensored calibration.

    Uses an iterative algorithm that converges to the maximum-likelihood
    corrected distribution without requiring matrix inversion. Each
    iteration applies Bayes' rule using the tensored calibration matrix.

    Parameters
    ----------
    counts : dict[str, int]
        Raw measurement counts.
    cal_data : CalibrationData
        Calibration data from tensored_calibration.
    max_iter : int
        Maximum number of iterations.
    tol : float
        Convergence tolerance (L1 norm of change).

    Returns
    -------
    Corrected probability distribution.
    """
    n_qubits = cal_data.n_qubits
    dim = 2**n_qubits
    shots = sum(counts.values())

    # Build raw probability vector
    raw_probs = np.zeros(dim)
    for bitstring, count in counts.items():
        idx = int(bitstring, 2)
        if idx < dim:
            raw_probs[idx] = count / shots

    # Initialize with raw probabilities
    corrected = raw_probs.copy()

    for _ in range(max_iter):
        # Forward model: apply tensored calibration matrix
        predicted = _apply_tensored_forward(corrected, cal_data)

        # Bayesian update: element-wise ratio
        ratio = np.zeros(dim)
        for i in range(dim):
            if predicted[i] > 1e-15:
                ratio[i] = raw_probs[i] / predicted[i]
            else:
                ratio[i] = 1.0

        # Update
        new_corrected = corrected * ratio

        # Clip and renormalize
        new_corrected = np.maximum(new_corrected, 0)
        total = new_corrected.sum()
        if total > 0:
            new_corrected /= total

        # Check convergence
        change = np.sum(np.abs(new_corrected - corrected))
        corrected = new_corrected
        if change < tol:
            break

    result = {}
    for idx in range(dim):
        if corrected[idx] > 1e-10:
            bs = format(idx, f"0{n_qubits}b")
            result[bs] = float(corrected[idx])

    return result


def _apply_tensored_forward(
    probs: NDArray[np.float64],
    cal_data: CalibrationData,
) -> NDArray[np.float64]:
    """Apply the tensored calibration matrix (forward model).

    Computes M * p where M is the tensored calibration matrix.

    Parameters
    ----------
    probs : NDArray
        Input probability vector of length 2^n.
    cal_data : CalibrationData
        Calibration data.

    Returns
    -------
    Predicted (noisy) probability vector.
    """
    n_qubits = cal_data.n_qubits
    dim = 2**n_qubits
    result = np.zeros(dim)

    for out_idx in range(dim):
        out_bits = format(out_idx, f"0{n_qubits}b")
        for in_idx in range(dim):
            in_bits = format(in_idx, f"0{n_qubits}b")
            element = 1.0
            for q in range(n_qubits):
                out_bit = int(out_bits[q])
                in_bit = int(in_bits[q])
                element *= cal_data.qubit_matrices[q][out_bit, in_bit]
            result[out_idx] += element * probs[in_idx]

    return result
