"""Unit tests for qufin.ml.quantum_transfer module."""

from __future__ import annotations

import numpy as np
import pytest

from qufin.backends.mock import MockBackend
from qufin.ml.quantum_transfer import (
    ClassicalTransferLearner,
    ClassicalTransferResult,
    PCAFeatureExtractor,
    QuantumTransferLearner,
    TradingSignal,
    TransferConfig,
    TransferResult,
    generate_synthetic_trading_data,
)

# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------


@pytest.fixture
def backend() -> MockBackend:
    return MockBackend(seed=42)


@pytest.fixture
def config() -> TransferConfig:
    return TransferConfig(
        n_pca_components=2,
        n_qubits=2,
        n_layers=1,
        max_iter=3,
        n_classes=3,
        seed=42,
    )


@pytest.fixture
def learner(config: TransferConfig, backend: MockBackend) -> QuantumTransferLearner:
    return QuantumTransferLearner(config, backend)


@pytest.fixture
def sample_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (15, 6))
    y = rng.integers(0, 3, size=15).astype(np.int64)
    return X, y


# -----------------------------------------------------------------------
# TradingSignal enum
# -----------------------------------------------------------------------


class TestTradingSignal:
    def test_values(self) -> None:
        assert TradingSignal.BUY == 0
        assert TradingSignal.HOLD == 1
        assert TradingSignal.SELL == 2

    def test_len(self) -> None:
        assert len(TradingSignal) == 3


# -----------------------------------------------------------------------
# TransferConfig
# -----------------------------------------------------------------------


class TestTransferConfig:
    def test_defaults(self) -> None:
        cfg = TransferConfig()
        assert cfg.n_pca_components == 4
        assert cfg.n_qubits == 4
        assert cfg.n_classes == 3

    def test_custom(self) -> None:
        cfg = TransferConfig(n_pca_components=8, n_qubits=8)
        assert cfg.n_pca_components == 8


# -----------------------------------------------------------------------
# PCAFeatureExtractor
# -----------------------------------------------------------------------


class TestPCAFeatureExtractor:
    def test_fit_transform_shape(self) -> None:
        pca = PCAFeatureExtractor(n_components=3)
        X = np.random.default_rng(0).normal(0, 1, (20, 8))
        X_pca = pca.fit_transform(X)
        assert X_pca.shape == (20, 3)

    def test_explained_variance_sums_leq_one(self) -> None:
        pca = PCAFeatureExtractor(n_components=3)
        X = np.random.default_rng(1).normal(0, 1, (50, 10))
        pca.fit(X)
        assert pca.explained_variance_ratio_ is not None
        assert pca.explained_variance_ratio_.sum() <= 1.0 + 1e-10

    def test_transform_without_fit_raises(self) -> None:
        pca = PCAFeatureExtractor(n_components=2)
        X = np.ones((5, 4))
        with pytest.raises(RuntimeError, match="fit"):
            pca.transform(X)

    def test_single_sample(self) -> None:
        pca = PCAFeatureExtractor(n_components=2)
        X = np.array([[1.0, 2.0, 3.0, 4.0]])
        X_pca = pca.fit_transform(X)
        assert X_pca.shape == (1, 2)

    def test_components_fewer_than_features(self) -> None:
        pca = PCAFeatureExtractor(n_components=2)
        X = np.random.default_rng(2).normal(0, 1, (30, 5))
        pca.fit(X)
        assert pca.components_ is not None
        assert pca.components_.shape == (2, 5)

    def test_mean_subtracted(self) -> None:
        pca = PCAFeatureExtractor(n_components=2)
        X = np.random.default_rng(3).normal(10, 1, (20, 4))
        pca.fit(X)
        assert pca.mean_ is not None
        np.testing.assert_allclose(pca.mean_, X.mean(axis=0))


# -----------------------------------------------------------------------
# QuantumTransferLearner
# -----------------------------------------------------------------------


class TestQuantumTransferLearner:
    def test_instantiation(self, learner: QuantumTransferLearner) -> None:
        assert learner.config.n_qubits == 2
        assert learner.config.n_pca_components == 2

    def test_n_params(self, learner: QuantumTransferLearner) -> None:
        # 2 qubits, 1 layer: 2*2*1 + 2 = 6
        assert learner._n_params() == 6

    def test_build_vqc_circuit(self, learner: QuantumTransferLearner) -> None:
        pytest.importorskip("qiskit")
        x = np.array([0.5, 1.0])
        params = np.zeros(learner._n_params())
        circ = learner._build_vqc_circuit(x, params)
        assert circ.num_qubits == 2

    def test_circuit_class_probabilities_sum_to_one(
        self, learner: QuantumTransferLearner
    ) -> None:
        x = np.array([0.5, 1.0])
        params = np.random.default_rng(0).uniform(0, 2 * np.pi, learner._n_params())
        probs = learner._circuit_class_probabilities(x, params)
        assert probs.shape == (3,)
        np.testing.assert_allclose(probs.sum(), 1.0, atol=1e-10)

    def test_fit_returns_result(
        self,
        learner: QuantumTransferLearner,
        sample_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        X, y = sample_data
        result = learner.fit(X, y)
        assert isinstance(result, TransferResult)
        assert result.vqc_params.shape == (learner._n_params(),)
        assert result.wall_time_s > 0.0
        assert 0.0 <= result.train_accuracy <= 1.0

    def test_predict_shape(
        self,
        learner: QuantumTransferLearner,
        sample_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        X, y = sample_data
        learner.fit(X, y)
        preds = learner.predict(X)
        assert preds.shape == (15,)
        assert np.all(preds >= 0)
        assert np.all(preds < 3)

    def test_predict_proba_shape(
        self,
        learner: QuantumTransferLearner,
        sample_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        X, y = sample_data
        learner.fit(X, y)
        probs = learner.predict_proba(X)
        assert probs.shape == (15, 3)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-10)

    def test_predict_before_fit_raises(
        self, learner: QuantumTransferLearner
    ) -> None:
        X = np.ones((5, 6))
        with pytest.raises(RuntimeError, match="fit"):
            learner.predict(X)

    def test_predict_proba_before_fit_raises(
        self, learner: QuantumTransferLearner
    ) -> None:
        X = np.ones((5, 6))
        with pytest.raises(RuntimeError, match="fit"):
            learner.predict_proba(X)

    def test_scale_features(self) -> None:
        X = np.array([[0.0, 10.0], [5.0, 20.0], [10.0, 30.0]])
        scaled = QuantumTransferLearner._scale_features(X)
        np.testing.assert_allclose(scaled[0], [0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(scaled[-1], [np.pi, np.pi], atol=1e-12)

    def test_pca_explained_variance_in_result(
        self,
        learner: QuantumTransferLearner,
        sample_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        X, y = sample_data
        result = learner.fit(X, y)
        assert result.pca_explained_variance_ratio.shape == (2,)
        assert np.all(result.pca_explained_variance_ratio >= 0)


# -----------------------------------------------------------------------
# ClassicalTransferLearner
# -----------------------------------------------------------------------


class TestClassicalTransferLearner:
    def test_fit_returns_result(self) -> None:
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, (15, 6))
        y = rng.integers(0, 3, size=15).astype(np.int64)
        ctl = ClassicalTransferLearner(
            n_pca_components=2,
            hidden_sizes=[4],
            n_epochs=3,
            seed=42,
        )
        result = ctl.fit(X, y)
        assert isinstance(result, ClassicalTransferResult)
        assert len(result.loss_history) == 3
        assert 0.0 <= result.train_accuracy <= 1.0

    def test_predict_shape(self) -> None:
        rng = np.random.default_rng(1)
        X = rng.normal(0, 1, (10, 4))
        y = rng.integers(0, 3, size=10).astype(np.int64)
        ctl = ClassicalTransferLearner(
            n_pca_components=2,
            hidden_sizes=[4],
            n_epochs=2,
            seed=0,
        )
        ctl.fit(X, y)
        preds = ctl.predict(X)
        assert preds.shape == (10,)

    def test_predict_proba_sums_to_one(self) -> None:
        rng = np.random.default_rng(2)
        X = rng.normal(0, 1, (10, 4))
        y = rng.integers(0, 3, size=10).astype(np.int64)
        ctl = ClassicalTransferLearner(
            n_pca_components=2,
            hidden_sizes=[4],
            n_epochs=2,
            seed=0,
        )
        ctl.fit(X, y)
        probs = ctl.predict_proba(X)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-10)

    def test_predict_before_fit_raises(self) -> None:
        ctl = ClassicalTransferLearner()
        with pytest.raises(RuntimeError, match="fit"):
            ctl.predict(np.ones((3, 5)))


# -----------------------------------------------------------------------
# generate_synthetic_trading_data
# -----------------------------------------------------------------------


class TestSyntheticData:
    def test_shapes(self) -> None:
        X, y = generate_synthetic_trading_data(n_samples=50, n_features=8)
        assert X.shape == (50, 8)
        assert y.shape == (50,)

    def test_labels_in_range(self) -> None:
        _X, y = generate_synthetic_trading_data(n_samples=100, n_classes=3)
        assert np.all(y >= 0)
        assert np.all(y < 3)

    def test_reproducible(self) -> None:
        X1, y1 = generate_synthetic_trading_data(seed=123)
        X2, y2 = generate_synthetic_trading_data(seed=123)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)
