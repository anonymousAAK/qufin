"""Unit tests for qufin.ml modules (kernels, reservoir, classifiers)."""

from __future__ import annotations

import numpy as np
import pytest

from qufin.backends.mock import MockBackend
from qufin.ml.classifiers import VariationalQuantumClassifier, VQCConfig
from qufin.ml.kernels import ZZFeatureMap, quantum_kernel_matrix
from qufin.ml.reservoir import QuantumReservoir, QuantumReservoirConfig

# -----------------------------------------------------------------------
# Quantum kernel matrix
# -----------------------------------------------------------------------

class TestQuantumKernelMatrix:
    """Tests for quantum_kernel_matrix with MockBackend."""

    @pytest.fixture
    def backend(self) -> MockBackend:
        return MockBackend(seed=42)

    def test_kernel_matrix_symmetric(self, backend: MockBackend) -> None:
        rng = np.random.default_rng(0)
        X = rng.uniform(0, 2 * np.pi, size=(5, 2))
        K = quantum_kernel_matrix(X, n_qubits=2, backend=backend, reps=1)
        np.testing.assert_allclose(K, K.T, atol=1e-12)

    def test_kernel_matrix_psd(self, backend: MockBackend) -> None:
        rng = np.random.default_rng(1)
        X = rng.uniform(0, 2 * np.pi, size=(4, 2))
        K = quantum_kernel_matrix(X, n_qubits=2, backend=backend, reps=1)
        eigenvalues = np.linalg.eigvalsh(K)
        # All eigenvalues should be >= 0 (PSD)
        assert np.all(eigenvalues >= -1e-10)

    def test_kernel_matrix_diagonal_ones(self, backend: MockBackend) -> None:
        rng = np.random.default_rng(2)
        X = rng.uniform(0, 2 * np.pi, size=(3, 2))
        K = quantum_kernel_matrix(X, n_qubits=2, backend=backend, reps=1)
        np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-12)

    def test_kernel_matrix_shape(self, backend: MockBackend) -> None:
        rng = np.random.default_rng(3)
        X = rng.uniform(0, 2 * np.pi, size=(6, 2))
        K = quantum_kernel_matrix(X, n_qubits=2, backend=backend, reps=1)
        assert K.shape == (6, 6)


class TestZZFeatureMap:
    """Tests for ZZFeatureMap circuit builder."""

    def test_build_circuit_returns_circuit(self) -> None:
        pytest.importorskip("qiskit")
        x = np.array([0.5, 1.0])
        circ = ZZFeatureMap.build_circuit(x, n_qubits=2, reps=1)
        assert circ.num_qubits == 2


# -----------------------------------------------------------------------
# Quantum reservoir
# -----------------------------------------------------------------------

class TestQuantumReservoir:
    """Tests for QuantumReservoir with MockBackend."""

    @pytest.fixture
    def reservoir(self) -> QuantumReservoir:
        cfg = QuantumReservoirConfig(n_qubits=3, n_layers=2, seed=42)
        backend = MockBackend(seed=0)
        return QuantumReservoir(cfg, backend)

    def test_instantiation(self, reservoir: QuantumReservoir) -> None:
        assert reservoir is not None
        assert reservoir.config.n_qubits == 3

    def test_extract_features_returns_array(self, reservoir: QuantumReservoir) -> None:
        input_data = np.array([0.1, 0.2, 0.3])
        features = reservoir.extract_features(input_data)
        assert isinstance(features, np.ndarray)
        assert features.shape == (3,)

    def test_extract_features_bounded(self, reservoir: QuantumReservoir) -> None:
        """Z expectation values should be in [-1, 1]."""
        input_data = np.array([1.0, 2.0, 0.5])
        features = reservoir.extract_features(input_data)
        assert np.all(features >= -1.0 - 1e-10)
        assert np.all(features <= 1.0 + 1e-10)


# -----------------------------------------------------------------------
# Variational quantum classifier
# -----------------------------------------------------------------------

class TestVariationalQuantumClassifier:
    """Tests for VariationalQuantumClassifier with MockBackend."""

    def test_instantiation(self) -> None:
        cfg = VQCConfig(n_qubits=2, n_layers=1, seed=42)
        backend = MockBackend(seed=0)
        vqc = VariationalQuantumClassifier(cfg, backend)
        assert vqc is not None
        assert vqc.config.n_qubits == 2

    def test_config_creation(self) -> None:
        cfg = VQCConfig(n_qubits=4, n_layers=3, n_epochs=50)
        assert cfg.n_qubits == 4
        assert cfg.n_layers == 3
        assert cfg.n_epochs == 50

    def test_n_params_positive(self) -> None:
        cfg = VQCConfig(n_qubits=3, n_layers=2, seed=0)
        backend = MockBackend(seed=0)
        vqc = VariationalQuantumClassifier(cfg, backend)
        assert vqc._n_params() > 0
