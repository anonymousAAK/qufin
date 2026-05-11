"""Variational quantum classifiers for fraud detection.

Implements a parameterized variational quantum classifier (VQC) with
angle encoding and a TwoLocal ansatz, trained via gradient-free
optimization (COBYLA).

References
----------
Schuld, Bocharov, Svore, Killoran, PRA 101, 032308 (2020).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from qufin.backends.base import Backend


@dataclass
class VQCConfig:
    """Configuration for a variational quantum classifier."""

    n_qubits: int = 4
    n_layers: int = 3
    optimizer: str = "COBYLA"
    n_epochs: int = 200
    lr: float = 0.01
    seed: int | None = 42


class VariationalQuantumClassifier:
    """Variational quantum classifier with angle encoding and TwoLocal ansatz.

    Features are encoded via R_Y rotations.  The ansatz alternates layers
    of R_Y / R_Z single-qubit rotations with a ladder of CNOT gates.
    The model is trained by minimising cross-entropy loss using a
    gradient-free optimizer (COBYLA by default).
    """

    def __init__(self, config: VQCConfig, backend: Backend) -> None:
        self.config = config
        self.backend = backend
        self._params: NDArray[np.float64] | None = None
        self._rng = np.random.default_rng(config.seed)

    def _n_params(self) -> int:
        """Total number of variational parameters."""
        # Each layer: n_qubits * 2 (RY + RZ), plus one final RY layer
        return 2 * self.config.n_qubits * self.config.n_layers + self.config.n_qubits

    def build_circuit(
        self,
        x: NDArray[np.float64],
        params: NDArray[np.float64],
    ) -> Any:
        """Build the full VQC circuit (encoding + ansatz).

        Parameters
        ----------
        x : array of shape (n_qubits,) or smaller
            Feature vector for angle encoding.
        params : array
            Variational parameters.

        Returns
        -------
        QuantumCircuit
        """
        from qiskit.circuit import QuantumCircuit

        n = self.config.n_qubits
        qc = QuantumCircuit(n)

        # Angle encoding: R_Y(x_i)
        for i in range(min(len(x), n)):
            qc.ry(float(x[i]), i)

        # TwoLocal ansatz
        idx = 0
        for _layer in range(self.config.n_layers):
            # Parameterized rotations
            for i in range(n):
                qc.ry(float(params[idx]), i)
                idx += 1
                qc.rz(float(params[idx]), i)
                idx += 1
            # CNOT ladder
            for i in range(n - 1):
                qc.cx(i, i + 1)
            if n > 2:
                qc.cx(n - 1, 0)

        # Final rotation layer
        for i in range(n):
            qc.ry(float(params[idx]), i)
            idx += 1

        return qc

    def _circuit_probability(
        self, x: NDArray[np.float64], params: NDArray[np.float64]
    ) -> float:
        """Probability of measuring |0...0> (class-0 probability)."""
        qc = self.build_circuit(x, params)
        sv = self.backend.statevector(qc)
        return float(np.abs(sv[0]) ** 2)

    def _loss(
        self,
        params: NDArray[np.float64],
        X: NDArray[np.float64],
        y: NDArray,
    ) -> float:
        """Binary cross-entropy loss over the dataset."""
        eps = 1e-10
        total = 0.0
        for i in range(X.shape[0]):
            p0 = np.clip(self._circuit_probability(X[i], params), eps, 1 - eps)
            if y[i] == 0:
                total -= np.log(p0)
            else:
                total -= np.log(1.0 - p0)
        return float(total / X.shape[0])

    def fit(
        self, X: NDArray[np.float64], y: NDArray
    ) -> VariationalQuantumClassifier:
        """Train the VQC on labelled data.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
        y : array of shape (n_samples,) with labels in {0, 1}

        Returns
        -------
        self
        """
        init_params = self._rng.uniform(0, 2 * np.pi, self._n_params())

        result = minimize(
            self._loss,
            init_params,
            args=(X, y),
            method=self.config.optimizer,
            options={"maxiter": self.config.n_epochs},
        )
        self._params = result.x
        return self

    def predict_proba(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return class probabilities for each sample.

        Returns array of shape ``(n_samples, 2)`` with columns [P(0), P(1)].
        """
        assert self._params is not None, "Must call fit() first."
        n = X.shape[0]
        probs = np.zeros((n, 2), dtype=np.float64)
        for i in range(n):
            p0 = self._circuit_probability(X[i], self._params)
            probs[i, 0] = p0
            probs[i, 1] = 1.0 - p0
        return probs

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.int64]:
        """Predict class labels (0 or 1) for each sample."""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1).astype(np.int64)
