"""Quantum reservoir computing for volatility forecasting (arXiv:2505.13933).

Implements a quantum reservoir based on a transverse-field Ising Hamiltonian.
Input data is encoded via single-qubit rotations, the reservoir dynamics
evolve the state, and expectation values are extracted as features for a
classical ridge-regression readout layer.

References
----------
Li et al., arXiv:2505.13933 (2025).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from qufin.backends.base import Backend


@dataclass
class QuantumReservoirConfig:
    """Configuration for quantum reservoir computing."""

    n_qubits: int = 4
    n_layers: int = 3
    coupling_strength: float = 0.5
    measurement_basis: str = "Z"
    ridge_alpha: float = 1.0
    seed: int | None = 42


class QuantumReservoir:
    """Quantum reservoir for time-series regression.

    Uses a transverse-field Ising Hamiltonian as reservoir dynamics:
    H = -J sum_{<i,j>} Z_i Z_j  -  h sum_i X_i

    Input data is encoded via R_Y rotations, then reservoir layers apply
    ZZ couplings and transverse-field X rotations.  Expectation values of
    Pauli-Z operators serve as the feature vector for a linear readout.
    """

    def __init__(self, config: QuantumReservoirConfig, backend: Backend) -> None:
        self.config = config
        self.backend = backend
        self._readout: Any = None
        self._rng = np.random.default_rng(config.seed)

    def build_reservoir_circuit(
        self, input_data: NDArray[np.float64]
    ) -> Any:
        """Build the reservoir circuit for a single input vector.

        Parameters
        ----------
        input_data : array of shape (n_qubits,) or smaller
            Values are mapped to R_Y rotation angles.

        Returns
        -------
        QuantumCircuit
        """
        from qiskit.circuit import QuantumCircuit

        n = self.config.n_qubits
        qc = QuantumCircuit(n)
        J = self.config.coupling_strength

        # Encode input via R_Y rotations
        for i in range(min(len(input_data), n)):
            qc.ry(float(input_data[i]), i)

        # Reservoir layers: transverse-field Ising dynamics
        for _ in range(self.config.n_layers):
            # ZZ coupling between nearest neighbours
            for i in range(n - 1):
                qc.cx(i, i + 1)
                qc.rz(2.0 * J, i + 1)
                qc.cx(i, i + 1)

            # Transverse field: R_X on every qubit
            for i in range(n):
                qc.rx(J, i)

        return qc

    def extract_features(
        self, input_data: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Extract feature vector from a single input via Z expectation values.

        Returns array of shape ``(n_qubits,)`` with <Z_i> for each qubit.
        """
        circ = self.build_reservoir_circuit(input_data)
        sv = self.backend.statevector(circ)
        n = self.config.n_qubits

        # Compute <Z_i> from statevector
        features = np.zeros(n, dtype=np.float64)
        n_states = 2**n
        probs = np.abs(sv[:n_states]) ** 2
        for i in range(n):
            # Z_i eigenvalue is +1 if bit i is 0, -1 if bit i is 1
            plus = 0.0
            minus = 0.0
            for s in range(n_states):
                bit = (s >> (n - 1 - i)) & 1
                if bit == 0:
                    plus += probs[s]
                else:
                    minus += probs[s]
            features[i] = plus - minus

        return features

    def _build_feature_matrix(
        self, X: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Build feature matrix for the full dataset."""
        n_samples = X.shape[0]
        n_feat = self.config.n_qubits
        F = np.zeros((n_samples, n_feat), dtype=np.float64)
        for i in range(n_samples):
            F[i] = self.extract_features(X[i])
        return F

    def fit(
        self, X_train: NDArray[np.float64], y_train: NDArray[np.float64]
    ) -> QuantumReservoir:
        """Fit the linear readout layer on quantum reservoir features.

        Parameters
        ----------
        X_train : array of shape (n_samples, n_features)
        y_train : array of shape (n_samples,)

        Returns
        -------
        self
        """
        from sklearn.linear_model import Ridge

        F = self._build_feature_matrix(X_train)
        self._readout = Ridge(alpha=self.config.ridge_alpha)
        self._readout.fit(F, y_train)
        return self

    def predict(self, X_test: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict target values for *X_test*."""
        assert self._readout is not None, "Must call fit() first."
        F = self._build_feature_matrix(X_test)
        return self._readout.predict(F)
