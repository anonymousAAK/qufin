"""Unit tests for qufin.ml.quantum_autoencoder module."""

from __future__ import annotations

import numpy as np
import pytest

from qufin.backends.mock import MockBackend
from qufin.ml.quantum_autoencoder import (
    QAutoEncoderConfig,
    QAutoEncoderResult,
    QuantumAutoEncoder,
    classical_pca_anomaly,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def backend() -> MockBackend:
    return MockBackend(seed=42)


@pytest.fixture
def config() -> QAutoEncoderConfig:
    return QAutoEncoderConfig(
        n_qubits=4, n_latent=2, n_layers=1, n_epochs=2, shots=64, seed=42,
    )


@pytest.fixture
def encoder(config: QAutoEncoderConfig, backend: MockBackend) -> QuantumAutoEncoder:
    return QuantumAutoEncoder(config, backend)


@pytest.fixture
def sample_data() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.uniform(0, 1, (8, 4))


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestQAutoEncoderConfig:
    def test_defaults(self) -> None:
        cfg = QAutoEncoderConfig()
        assert cfg.n_qubits == 4
        assert cfg.n_latent == 2
        assert cfg.n_layers == 3
        assert cfg.n_epochs == 30

    def test_custom(self) -> None:
        cfg = QAutoEncoderConfig(n_qubits=6, n_latent=3, n_layers=5)
        assert cfg.n_qubits == 6
        assert cfg.n_latent == 3
        assert cfg.n_layers == 5


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


class TestQuantumAutoEncoderInit:
    def test_valid_config(self, backend: MockBackend) -> None:
        cfg = QAutoEncoderConfig(n_qubits=4, n_latent=2)
        enc = QuantumAutoEncoder(cfg, backend)
        assert enc.config.n_qubits == 4

    def test_latent_ge_qubits_raises(self, backend: MockBackend) -> None:
        cfg = QAutoEncoderConfig(n_qubits=3, n_latent=3)
        with pytest.raises(ValueError, match="n_latent must be strictly less"):
            QuantumAutoEncoder(cfg, backend)

    def test_latent_gt_qubits_raises(self, backend: MockBackend) -> None:
        cfg = QAutoEncoderConfig(n_qubits=3, n_latent=5)
        with pytest.raises(ValueError, match="n_latent must be strictly less"):
            QuantumAutoEncoder(cfg, backend)

    def test_too_few_qubits_raises(self, backend: MockBackend) -> None:
        cfg = QAutoEncoderConfig(n_qubits=1, n_latent=0)
        with pytest.raises(ValueError, match="n_qubits must be at least 2"):
            QuantumAutoEncoder(cfg, backend)

    def test_params_initialised(self, encoder: QuantumAutoEncoder) -> None:
        assert len(encoder.params) > 0


# ---------------------------------------------------------------------------
# Reconstruction error
# ---------------------------------------------------------------------------


class TestReconstructionError:
    def test_error_between_zero_and_one(
        self, encoder: QuantumAutoEncoder
    ) -> None:
        x = np.array([0.5, 0.3, 0.7, 0.1])
        err = encoder._reconstruction_error(x, encoder.params)
        assert 0.0 <= err <= 1.0

    def test_error_is_float(self, encoder: QuantumAutoEncoder) -> None:
        x = np.array([0.0, 0.0, 0.0, 0.0])
        err = encoder._reconstruction_error(x, encoder.params)
        assert isinstance(err, float)


# ---------------------------------------------------------------------------
# Training (fit)
# ---------------------------------------------------------------------------


class TestFit:
    def test_returns_result(
        self, encoder: QuantumAutoEncoder, sample_data: np.ndarray
    ) -> None:
        result = encoder.fit(sample_data)
        assert isinstance(result, QAutoEncoderResult)

    def test_loss_history_length(
        self, encoder: QuantumAutoEncoder, sample_data: np.ndarray
    ) -> None:
        result = encoder.fit(sample_data)
        assert len(result.loss_history) == encoder.config.n_epochs

    def test_anomaly_scores_shape(
        self, encoder: QuantumAutoEncoder, sample_data: np.ndarray
    ) -> None:
        result = encoder.fit(sample_data)
        assert result.anomaly_scores.shape == (len(sample_data),)

    def test_is_anomaly_shape(
        self, encoder: QuantumAutoEncoder, sample_data: np.ndarray
    ) -> None:
        result = encoder.fit(sample_data)
        assert result.is_anomaly.shape == (len(sample_data),)
        assert result.is_anomaly.dtype == np.bool_

    def test_threshold_positive(
        self, encoder: QuantumAutoEncoder, sample_data: np.ndarray
    ) -> None:
        result = encoder.fit(sample_data)
        assert result.threshold >= 0.0

    def test_wall_time_positive(
        self, encoder: QuantumAutoEncoder, sample_data: np.ndarray
    ) -> None:
        result = encoder.fit(sample_data)
        assert result.wall_time_s > 0

    def test_custom_threshold(
        self, backend: MockBackend, sample_data: np.ndarray
    ) -> None:
        cfg = QAutoEncoderConfig(
            n_qubits=4, n_latent=2, n_layers=1, n_epochs=1,
            shots=32, anomaly_threshold=0.3, seed=42,
        )
        enc = QuantumAutoEncoder(cfg, backend)
        result = enc.fit(sample_data)
        assert result.threshold == 0.3


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


class TestScore:
    def test_score_returns_result(
        self, encoder: QuantumAutoEncoder, sample_data: np.ndarray
    ) -> None:
        result = encoder.score(sample_data, threshold=0.5)
        assert isinstance(result, QAutoEncoderResult)

    def test_score_uses_threshold(
        self, encoder: QuantumAutoEncoder, sample_data: np.ndarray
    ) -> None:
        result = encoder.score(sample_data, threshold=0.9)
        assert result.threshold == 0.9

    def test_score_default_threshold(
        self, encoder: QuantumAutoEncoder, sample_data: np.ndarray
    ) -> None:
        result = encoder.score(sample_data)
        assert result.threshold == 0.5


# ---------------------------------------------------------------------------
# Classical PCA baseline
# ---------------------------------------------------------------------------


class TestClassicalPCAAnomaly:
    def test_returns_result(self, sample_data: np.ndarray) -> None:
        result = classical_pca_anomaly(sample_data, n_components=2)
        assert isinstance(result, QAutoEncoderResult)

    def test_scores_shape(self, sample_data: np.ndarray) -> None:
        result = classical_pca_anomaly(sample_data, n_components=2)
        assert result.anomaly_scores.shape == (len(sample_data),)

    def test_scores_non_negative(self, sample_data: np.ndarray) -> None:
        result = classical_pca_anomaly(sample_data, n_components=2)
        assert np.all(result.anomaly_scores >= 0.0)

    def test_custom_threshold(self, sample_data: np.ndarray) -> None:
        result = classical_pca_anomaly(sample_data, n_components=2, threshold=0.1)
        assert result.threshold == 0.1
