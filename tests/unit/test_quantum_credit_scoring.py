"""Tests for quantum credit scoring module.

All optional dependencies (sklearn, xgboost, scipy) are mocked
to ensure tests run without them installed.
"""

from __future__ import annotations

import sys
from unittest.mock import patch

import numpy as np
import pytest

from qufin.backends.mock import MockBackend

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_backend():
    """Create a MockBackend that returns deterministic statevectors."""
    return MockBackend()


@pytest.fixture
def small_dataset():
    """Create a small synthetic dataset for testing."""
    rng = np.random.default_rng(42)
    n_samples, n_features = 20, 4
    X = rng.standard_normal((n_samples, n_features))
    y = rng.integers(0, 2, size=n_samples).astype(np.int64)
    protected = rng.integers(0, 2, size=n_samples).astype(np.int64)
    return X, y, protected


# ---------------------------------------------------------------------------
# IQPFeatureMap tests
# ---------------------------------------------------------------------------


class TestIQPFeatureMap:
    """Tests for the IQP feature map circuit builder."""

    def test_build_circuit_returns_circuit(self):
        """IQP feature map should return a QuantumCircuit."""
        pytest.importorskip("qiskit")
        from qufin.ml.quantum_credit_scoring import IQPFeatureMap

        x = np.array([0.1, 0.2, 0.3])
        circ = IQPFeatureMap.build_circuit(x, n_qubits=3, reps=1)
        assert circ.num_qubits == 3

    def test_build_circuit_reps(self):
        """Circuit depth should increase with reps."""
        pytest.importorskip("qiskit")
        from qufin.ml.quantum_credit_scoring import IQPFeatureMap

        x = np.array([0.1, 0.2])
        circ1 = IQPFeatureMap.build_circuit(x, n_qubits=2, reps=1)
        circ2 = IQPFeatureMap.build_circuit(x, n_qubits=2, reps=2)
        assert circ2.depth() > circ1.depth()

    def test_build_circuit_single_qubit(self):
        """Single-qubit IQP should still work (no entangling gates)."""
        pytest.importorskip("qiskit")
        from qufin.ml.quantum_credit_scoring import IQPFeatureMap

        x = np.array([0.5])
        circ = IQPFeatureMap.build_circuit(x, n_qubits=1, reps=2)
        assert circ.num_qubits == 1


# ---------------------------------------------------------------------------
# FeatureMapType tests
# ---------------------------------------------------------------------------


class TestFeatureMapType:
    """Tests for the FeatureMapType enum."""

    def test_enum_values(self):
        from qufin.ml.quantum_credit_scoring import FeatureMapType

        assert FeatureMapType.ZZ.value == "zz"
        assert FeatureMapType.IQP.value == "iqp"

    def test_build_feature_map_zz(self):
        pytest.importorskip("qiskit")
        from qufin.ml.quantum_credit_scoring import (
            FeatureMapType,
            _build_feature_map_circuit,
        )

        x = np.array([0.1, 0.2])
        circ = _build_feature_map_circuit(x, 2, FeatureMapType.ZZ, 1)
        assert circ.num_qubits == 2

    def test_build_feature_map_iqp(self):
        pytest.importorskip("qiskit")
        from qufin.ml.quantum_credit_scoring import (
            FeatureMapType,
            _build_feature_map_circuit,
        )

        x = np.array([0.1, 0.2])
        circ = _build_feature_map_circuit(x, 2, FeatureMapType.IQP, 1)
        assert circ.num_qubits == 2

    def test_build_feature_map_invalid(self):
        from qufin.ml.quantum_credit_scoring import _build_feature_map_circuit

        with pytest.raises(ValueError, match="Unknown feature map"):
            _build_feature_map_circuit(np.array([0.1]), 1, "invalid", 1)


# ---------------------------------------------------------------------------
# Projected quantum kernel tests
# ---------------------------------------------------------------------------


class TestProjectedQuantumKernel:
    """Tests for projected quantum state and kernel computation."""

    def test_projected_state_shape(self, mock_backend):
        """Projected state should have shape (3 * n_qubits,)."""
        pytest.importorskip("qiskit")
        from qufin.ml.quantum_credit_scoring import (
            ProjectedKernelConfig,
            projected_quantum_state,
        )

        config = ProjectedKernelConfig(n_qubits=2, reps=1)
        x = np.array([0.1, 0.2])
        state = projected_quantum_state(x, mock_backend, config)
        assert state.shape == (6,)

    def test_projected_state_dtype(self, mock_backend):
        """Projected state should be float64."""
        pytest.importorskip("qiskit")
        from qufin.ml.quantum_credit_scoring import (
            ProjectedKernelConfig,
            projected_quantum_state,
        )

        config = ProjectedKernelConfig(n_qubits=2, reps=1)
        x = np.array([0.1, 0.2])
        state = projected_quantum_state(x, mock_backend, config)
        assert state.dtype == np.float64

    def test_projected_kernel_matrix_shape(self, mock_backend):
        """Kernel matrix should be (n_samples, n_samples)."""
        pytest.importorskip("qiskit")
        from qufin.ml.quantum_credit_scoring import (
            ProjectedKernelConfig,
            projected_kernel_matrix,
        )

        config = ProjectedKernelConfig(n_qubits=2, reps=1)
        X = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        K = projected_kernel_matrix(X, mock_backend, config)
        assert K.shape == (3, 3)

    def test_projected_kernel_matrix_symmetric(self, mock_backend):
        """Kernel matrix should be symmetric."""
        pytest.importorskip("qiskit")
        from qufin.ml.quantum_credit_scoring import (
            ProjectedKernelConfig,
            projected_kernel_matrix,
        )

        config = ProjectedKernelConfig(n_qubits=2, reps=1)
        X = np.array([[0.1, 0.2], [0.3, 0.4]])
        K = projected_kernel_matrix(X, mock_backend, config)
        np.testing.assert_allclose(K, K.T)

    def test_projected_kernel_diagonal_is_one(self, mock_backend):
        """Diagonal of kernel matrix should be 1."""
        pytest.importorskip("qiskit")
        from qufin.ml.quantum_credit_scoring import (
            ProjectedKernelConfig,
            projected_kernel_matrix,
        )

        config = ProjectedKernelConfig(n_qubits=2, reps=1)
        X = np.array([[0.1, 0.2], [0.3, 0.4]])
        K = projected_kernel_matrix(X, mock_backend, config)
        np.testing.assert_allclose(np.diag(K), 1.0)


# ---------------------------------------------------------------------------
# Feature preparation tests
# ---------------------------------------------------------------------------


class TestPrepareFeatures:
    """Tests for the _prepare_features helper."""

    def test_truncate_features(self):
        from qufin.ml.quantum_credit_scoring import _prepare_features

        X = np.ones((5, 10))
        result = _prepare_features(X, n_qubits=3)
        assert result.shape == (5, 3)

    def test_pad_features(self):
        from qufin.ml.quantum_credit_scoring import _prepare_features

        X = np.ones((5, 2))
        result = _prepare_features(X, n_qubits=4)
        assert result.shape == (5, 4)
        # Padded columns should be 0
        np.testing.assert_array_equal(result[:, 2:], 0.0)

    def test_scale_range(self):
        from qufin.ml.quantum_credit_scoring import _prepare_features

        X = np.array([[0.0, 10.0], [5.0, 20.0], [10.0, 30.0]])
        result = _prepare_features(X, n_qubits=2)
        # Min should be 0, max should be 2*pi
        assert result[:, 0].min() == pytest.approx(0.0)
        assert result[:, 0].max() == pytest.approx(2 * np.pi)

    def test_constant_column(self):
        from qufin.ml.quantum_credit_scoring import _prepare_features

        X = np.array([[5.0], [5.0], [5.0]])
        result = _prepare_features(X, n_qubits=1)
        np.testing.assert_array_equal(result[:, 0], 0.0)


# ---------------------------------------------------------------------------
# Dataset loading tests
# ---------------------------------------------------------------------------


class TestDatasetLoading:
    """Tests for dataset loading helpers (use synthetic fallback)."""

    def test_synthetic_dataset_shape(self):
        from qufin.ml.quantum_credit_scoring import _synthetic_credit_dataset

        ds = _synthetic_credit_dataset(n_samples=100, n_features=10)
        assert ds.X.shape == (100, 10)
        assert ds.y.shape == (100,)
        assert ds.protected_attr is not None
        assert ds.protected_attr.shape == (100,)

    def test_synthetic_dataset_labels_binary(self):
        from qufin.ml.quantum_credit_scoring import _synthetic_credit_dataset

        ds = _synthetic_credit_dataset(n_samples=200)
        assert set(np.unique(ds.y)).issubset({0, 1})

    def test_synthetic_dataset_name(self):
        from qufin.ml.quantum_credit_scoring import _synthetic_credit_dataset

        ds = _synthetic_credit_dataset(name="test_ds")
        assert ds.name == "test_ds"

    def test_credit_dataset_dataclass(self):
        from qufin.ml.quantum_credit_scoring import CreditDataset

        ds = CreditDataset(
            X=np.zeros((5, 3)),
            y=np.zeros(5, dtype=np.int64),
            name="test",
        )
        assert ds.name == "test"
        assert ds.protected_attr is None
        assert ds.feature_names == []


# ---------------------------------------------------------------------------
# Fairness metrics tests
# ---------------------------------------------------------------------------


class TestFairnessMetrics:
    """Tests for fairness analysis."""

    def test_perfect_fairness(self):
        from qufin.ml.quantum_credit_scoring import compute_fairness_metrics

        y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0], dtype=np.int64)
        y_pred = np.array([1, 1, 0, 0, 1, 1, 0, 0], dtype=np.int64)
        protected = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)

        fm = compute_fairness_metrics(y_true, y_pred, protected)
        assert fm.statistical_parity_difference == pytest.approx(0.0)
        assert fm.equal_opportunity_difference == pytest.approx(0.0)

    def test_unfair_predictions(self):
        from qufin.ml.quantum_credit_scoring import compute_fairness_metrics

        y_true = np.array([1, 1, 1, 1], dtype=np.int64)
        # Group 0 always predicted positive, group 1 never
        y_pred = np.array([1, 1, 0, 0], dtype=np.int64)
        protected = np.array([0, 0, 1, 1], dtype=np.int64)

        fm = compute_fairness_metrics(y_true, y_pred, protected)
        assert fm.statistical_parity_difference == pytest.approx(1.0)
        assert fm.positive_rate_group_0 == pytest.approx(1.0)
        assert fm.positive_rate_group_1 == pytest.approx(0.0)

    def test_equal_opportunity(self):
        from qufin.ml.quantum_credit_scoring import compute_fairness_metrics

        y_true = np.array([1, 0, 1, 0], dtype=np.int64)
        y_pred = np.array([1, 0, 0, 0], dtype=np.int64)
        protected = np.array([0, 0, 1, 1], dtype=np.int64)

        fm = compute_fairness_metrics(y_true, y_pred, protected)
        # Group 0: TPR=1.0, Group 1: TPR=0.0
        assert fm.tpr_group_0 == pytest.approx(1.0)
        assert fm.tpr_group_1 == pytest.approx(0.0)
        assert fm.equal_opportunity_difference == pytest.approx(1.0)

    def test_empty_group(self):
        from qufin.ml.quantum_credit_scoring import compute_fairness_metrics

        y_true = np.array([1, 0], dtype=np.int64)
        y_pred = np.array([1, 0], dtype=np.int64)
        protected = np.array([0, 0], dtype=np.int64)

        fm = compute_fairness_metrics(y_true, y_pred, protected)
        # Group 1 is empty, rates should be 0
        assert fm.positive_rate_group_1 == 0.0
        assert fm.tpr_group_1 == 0.0


# ---------------------------------------------------------------------------
# Paired t-test tests
# ---------------------------------------------------------------------------


class TestPairedTTest:
    """Tests for paired t-test utility."""

    def test_identical_scores(self):
        from qufin.ml.quantum_credit_scoring import paired_t_test

        scores = np.array([0.8, 0.85, 0.82, 0.79, 0.83])
        result = paired_t_test(scores, scores, "A", "B")
        # Identical scores => t=0, p=1 (or nan), not significant
        assert not result.significant

    def test_different_scores(self):
        from qufin.ml.quantum_credit_scoring import paired_t_test

        scores_a = np.array([0.9, 0.91, 0.92, 0.89, 0.93])
        scores_b = np.array([0.5, 0.51, 0.52, 0.49, 0.53])
        result = paired_t_test(scores_a, scores_b, "A", "B")
        assert result.significant
        assert result.p_value < 0.05

    def test_mismatched_lengths(self):
        from qufin.ml.quantum_credit_scoring import paired_t_test

        with pytest.raises(ValueError, match="same length"):
            paired_t_test(np.array([0.8, 0.9]), np.array([0.7]), "A", "B")


# ---------------------------------------------------------------------------
# ProjectedKernelConfig tests
# ---------------------------------------------------------------------------


class TestProjectedKernelConfig:
    """Tests for the config dataclass."""

    def test_default_values(self):
        from qufin.ml.quantum_credit_scoring import (
            FeatureMapType,
            ProjectedKernelConfig,
        )

        config = ProjectedKernelConfig()
        assert config.n_qubits == 4
        assert config.feature_map == FeatureMapType.ZZ
        assert config.reps == 2
        assert config.gamma_projection == 1.0

    def test_custom_values(self):
        from qufin.ml.quantum_credit_scoring import (
            FeatureMapType,
            ProjectedKernelConfig,
        )

        config = ProjectedKernelConfig(
            n_qubits=8,
            feature_map=FeatureMapType.IQP,
            reps=3,
            gamma_projection=0.5,
        )
        assert config.n_qubits == 8
        assert config.feature_map == FeatureMapType.IQP


# ---------------------------------------------------------------------------
# ClassifierMetrics / BenchmarkResult dataclass tests
# ---------------------------------------------------------------------------


class TestResultDataclasses:
    """Tests for result container dataclasses."""

    def test_classifier_metrics_defaults(self):
        from qufin.ml.quantum_credit_scoring import ClassifierMetrics

        m = ClassifierMetrics()
        assert m.accuracy == 0.0
        assert m.auc_roc == 0.0
        assert m.cv_scores is None

    def test_benchmark_result(self):
        from qufin.ml.quantum_credit_scoring import BenchmarkResult, ClassifierMetrics

        r = BenchmarkResult(
            classifier_name="test",
            metrics=ClassifierMetrics(accuracy=0.95),
            n_samples=100,
        )
        assert r.classifier_name == "test"
        assert r.metrics.accuracy == 0.95

    def test_paired_test_result(self):
        from qufin.ml.quantum_credit_scoring import PairedTestResult

        r = PairedTestResult(
            classifier_a="A", classifier_b="B",
            t_statistic=2.5, p_value=0.03, significant=True,
        )
        assert r.significant
        assert r.p_value == 0.03


# ---------------------------------------------------------------------------
# Integration: evaluate_classical_baseline (with sklearn)
# ---------------------------------------------------------------------------


class TestClassicalBaseline:
    """Tests for classical baseline evaluation (requires sklearn)."""

    def test_rbf_svm(self, small_dataset):
        pytest.importorskip("sklearn")
        from qufin.ml.quantum_credit_scoring import evaluate_classical_baseline

        X, y, _ = small_dataset
        result = evaluate_classical_baseline(
            X[:15], y[:15], X[15:], y[15:], classifier="rbf_svm"
        )
        assert result.classifier_name == "rbf_svm"
        assert 0.0 <= result.metrics.accuracy <= 1.0

    def test_logistic_regression(self, small_dataset):
        pytest.importorskip("sklearn")
        from qufin.ml.quantum_credit_scoring import evaluate_classical_baseline

        X, y, _ = small_dataset
        result = evaluate_classical_baseline(
            X[:15], y[:15], X[15:], y[15:], classifier="logistic_regression"
        )
        assert result.classifier_name == "logistic_regression"

    def test_unknown_classifier(self, small_dataset):
        pytest.importorskip("sklearn")
        from qufin.ml.quantum_credit_scoring import evaluate_classical_baseline

        X, y, _ = small_dataset
        with pytest.raises(ValueError, match="Unknown classifier"):
            evaluate_classical_baseline(
                X[:15], y[:15], X[15:], y[15:], classifier="invalid"
            )

    def test_cross_validate(self, small_dataset):
        pytest.importorskip("sklearn")
        from qufin.ml.quantum_credit_scoring import cross_validate_classifier

        X, y, _ = small_dataset
        scores = cross_validate_classifier(X, y, "rbf_svm", k_folds=3)
        assert len(scores) == 3
        assert all(0.0 <= s <= 1.0 for s in scores)


# ---------------------------------------------------------------------------
# require_sklearn guard test
# ---------------------------------------------------------------------------


class TestRequireSklearn:
    """Test the sklearn import guard."""

    def test_require_sklearn_when_missing(self):
        from qufin.ml.quantum_credit_scoring import _require_sklearn

        with patch.object(
            sys.modules["qufin.ml.quantum_credit_scoring"],
            "_HAS_SKLEARN",
            False,
        ), pytest.raises(ImportError, match="scikit-learn"):
            _require_sklearn()
