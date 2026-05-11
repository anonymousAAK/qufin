"""Quantum kernels (ZZ-feature-map, Havlicek et al. Nature 567, 2019).

Implements the ZZ feature map circuit and a quantum kernel SVM classifier
for financial data classification tasks such as credit scoring and fraud
detection.

References
----------
Havlicek et al., Nature 567, 209-212 (2019), arXiv:1804.11326.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from qufin.backends.base import Backend


class ZZFeatureMap:
    """ZZ feature map circuit builder.

    Encodes classical data *x* into a quantum state using single-qubit Z
    rotations and two-qubit ZZ entangling gates, repeated *reps* times.
    """

    @staticmethod
    def build_circuit(
        x: NDArray[np.float64], n_qubits: int, reps: int = 2
    ) -> Any:
        """Build the ZZ feature map circuit for data vector *x*.

        Parameters
        ----------
        x : array of shape (n_qubits,)
            Classical feature vector (values should be in [0, 2*pi]).
        n_qubits : int
            Number of qubits (must match ``len(x)``).
        reps : int
            Number of repetitions of the feature map layer.

        Returns
        -------
        QuantumCircuit
            Qiskit circuit encoding *x*.
        """
        from qiskit.circuit import QuantumCircuit

        qc = QuantumCircuit(n_qubits)

        for _ in range(reps):
            # Layer of Hadamards
            for i in range(n_qubits):
                qc.h(i)

            # Single-qubit Z rotations: R_Z(2 * x_i)
            for i in range(n_qubits):
                qc.rz(2.0 * x[i], i)

            # Two-qubit ZZ entangling: for each pair (i, j)
            for i in range(n_qubits):
                for j in range(i + 1, n_qubits):
                    phi = 2.0 * x[i] * x[j]
                    qc.cx(i, j)
                    qc.rz(phi, j)
                    qc.cx(i, j)

        return qc


def quantum_kernel(
    x1: NDArray[np.float64],
    x2: NDArray[np.float64],
    n_qubits: int,
    backend: Backend,
    reps: int = 2,
) -> float:
    """Compute the quantum kernel value between two data points.

    Uses statevector overlap: k(x1, x2) = |<0|U^dag(x2) U(x1)|0>|^2.
    """
    from qiskit.circuit import QuantumCircuit

    # Build U(x1)
    circ_x1 = ZZFeatureMap.build_circuit(x1, n_qubits, reps)
    # Build U(x2)^dag
    circ_x2_dag = ZZFeatureMap.build_circuit(x2, n_qubits, reps).inverse()

    # Compose: U^dag(x2) @ U(x1)
    combined = QuantumCircuit(n_qubits)
    combined.compose(circ_x1, inplace=True)
    combined.compose(circ_x2_dag, inplace=True)

    sv = backend.statevector(combined)
    # Kernel value is probability of measuring |0...0>
    return float(np.abs(sv[0]) ** 2)


def quantum_kernel_matrix(
    X: NDArray[np.float64],
    n_qubits: int,
    backend: Backend,
    reps: int = 2,
) -> NDArray[np.float64]:
    """Compute the full kernel matrix for dataset *X*.

    Parameters
    ----------
    X : array of shape (n_samples, n_qubits)
    n_qubits : int
    backend : Backend
    reps : int

    Returns
    -------
    K : array of shape (n_samples, n_samples)
    """
    n = X.shape[0]
    K = np.eye(n, dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            k_val = quantum_kernel(X[i], X[j], n_qubits, backend, reps)
            K[i, j] = k_val
            K[j, i] = k_val
    return K


@dataclass
class QuantumKernelClassifier:
    """SVM classifier backed by a quantum kernel.

    Wraps ``sklearn.svm.SVC`` with a precomputed quantum kernel matrix.
    """

    n_qubits: int
    backend: Backend
    reps: int = 2
    C: float = 1.0

    def __post_init__(self) -> None:
        from sklearn.svm import SVC

        self._svc = SVC(kernel="precomputed", C=self.C)
        self._X_train: NDArray[np.float64] | None = None

    def fit(self, X: NDArray[np.float64], y: NDArray) -> QuantumKernelClassifier:
        """Fit the classifier on training data."""
        self._X_train = X
        K_train = quantum_kernel_matrix(X, self.n_qubits, self.backend, self.reps)
        self._svc.fit(K_train, y)
        return self

    def predict(self, X: NDArray[np.float64]) -> NDArray:
        """Predict class labels for *X*."""
        assert self._X_train is not None, "Must call fit() first."
        n_test = X.shape[0]
        n_train = self._X_train.shape[0]
        K_test = np.zeros((n_test, n_train), dtype=np.float64)
        for i in range(n_test):
            for j in range(n_train):
                K_test[i, j] = quantum_kernel(
                    X[i], self._X_train[j], self.n_qubits, self.backend, self.reps
                )
        return self._svc.predict(K_test)
