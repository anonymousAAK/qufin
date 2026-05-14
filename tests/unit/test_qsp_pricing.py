"""Unit tests for QSP-based amplitude estimation and option pricing."""

from __future__ import annotations

import numpy as np
import pytest

from qufin.backends.mock import MockBackend
from qufin.options.amplitude_estimation.estimation_problem import EstimationProblem
from qufin.options.amplitude_estimation.qsp_pricing import (
    MultiVariableQSP,
    MultiVariableQSPConfig,
    MultiVariableQSPResult,
    QSPAmplitudeEstimation,
    QSPComparisonResult,
    QSPConfig,
    QSPResult,
    _build_signal_operator,
    _build_signal_rotation,
    chebyshev_coefficients_arcsin,
    compute_qsp_phases,
    evaluate_chebyshev,
    qsp_sequence,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_backend() -> MockBackend:
    return MockBackend(seed=42)


@pytest.fixture
def simple_estimation_problem() -> EstimationProblem:
    """A simple 2-qubit estimation problem with known amplitude."""
    from qiskit.circuit import QuantumCircuit

    # Create a state with known amplitude: sin^2(pi/6) = 0.25
    qc = QuantumCircuit(2)
    qc.ry(2 * np.arcsin(0.5), 0)  # amplitude = 0.25 on |1>
    qc.h(1)

    return EstimationProblem(
        state_preparation=qc,
        objective_qubits=[0],
        n_qubits=2,
    )


@pytest.fixture
def tiny_estimation_problem() -> EstimationProblem:
    """A 1-qubit estimation problem."""
    from qiskit.circuit import QuantumCircuit

    qc = QuantumCircuit(1)
    qc.ry(np.pi / 3, 0)  # sin^2(pi/6) = 0.25

    return EstimationProblem(
        state_preparation=qc,
        objective_qubits=[0],
        n_qubits=1,
    )


# ---------------------------------------------------------------------------
# Chebyshev coefficient tests
# ---------------------------------------------------------------------------


class TestChebyshevCoefficients:
    def test_returns_correct_length(self) -> None:
        """Should return the specified number of coefficients."""
        coeffs = chebyshev_coefficients_arcsin(8)
        assert len(coeffs) == 8

    def test_coefficients_are_finite(self) -> None:
        """All coefficients should be finite real numbers."""
        coeffs = chebyshev_coefficients_arcsin(10)
        assert all(np.isfinite(c) for c in coeffs)

    def test_first_coefficient_small(self) -> None:
        """First Chebyshev coefficient (constant term) should be near zero
        since arcsin is odd."""
        coeffs = chebyshev_coefficients_arcsin(10)
        # arcsin(0)/(pi/2) = 0, so constant term should be small
        assert abs(coeffs[0]) < 0.1

    def test_higher_degree_more_accurate(self) -> None:
        """Higher degree polynomial should approximate arcsin better."""
        x_test = 0.3
        true_val = np.arcsin(x_test) / (np.pi / 2)

        coeffs_low = chebyshev_coefficients_arcsin(4)
        coeffs_high = chebyshev_coefficients_arcsin(20)

        err_low = abs(evaluate_chebyshev(coeffs_low, x_test) - true_val)
        err_high = abs(evaluate_chebyshev(coeffs_high, x_test) - true_val)

        assert err_high <= err_low + 1e-10


# ---------------------------------------------------------------------------
# Chebyshev evaluation tests
# ---------------------------------------------------------------------------


class TestEvaluateChebyshev:
    def test_zero_coeffs(self) -> None:
        """Empty coefficients should return 0."""
        result = evaluate_chebyshev(np.array([]), 0.5)
        assert result == pytest.approx(0.0)

    def test_constant(self) -> None:
        """Single coefficient should return that constant."""
        result = evaluate_chebyshev(np.array([3.14]), 0.5)
        assert result == pytest.approx(3.14)

    def test_linear(self) -> None:
        """Coeffs [a0, a1] should evaluate to a0 + a1*x (Chebyshev T0, T1)."""
        result = evaluate_chebyshev(np.array([2.0, 3.0]), 0.5)
        assert result == pytest.approx(2.0 + 3.0 * 0.5)

    def test_at_boundaries(self) -> None:
        """Should work at x = -1 and x = 1."""
        coeffs = chebyshev_coefficients_arcsin(6)
        val_m1 = evaluate_chebyshev(coeffs, -1.0)
        val_p1 = evaluate_chebyshev(coeffs, 1.0)
        assert np.isfinite(val_m1)
        assert np.isfinite(val_p1)

    def test_array_input(self) -> None:
        """Should accept array inputs and return array."""
        coeffs = chebyshev_coefficients_arcsin(6)
        x = np.array([0.0, 0.3, 0.7])
        result = evaluate_chebyshev(coeffs, x)
        assert len(result) == 3
        assert all(np.isfinite(result))


# ---------------------------------------------------------------------------
# QSP phase factor tests
# ---------------------------------------------------------------------------


class TestQSPPhases:
    def test_returns_correct_length(self) -> None:
        """Phase factors should have length d+1 for degree-d polynomial."""
        coeffs = chebyshev_coefficients_arcsin(5)
        phases = compute_qsp_phases(coeffs)
        assert len(phases) == 6  # d+1

    def test_phases_are_finite(self) -> None:
        """All phase factors should be finite."""
        coeffs = chebyshev_coefficients_arcsin(8)
        phases = compute_qsp_phases(coeffs)
        assert all(np.isfinite(p) for p in phases)

    def test_empty_coefficients(self) -> None:
        """Empty coefficients should return single zero phase."""
        phases = compute_qsp_phases(np.array([]))
        assert len(phases) == 1
        assert phases[0] == 0.0


# ---------------------------------------------------------------------------
# Signal operator tests
# ---------------------------------------------------------------------------


class TestSignalOperator:
    def test_signal_operator_is_unitary(self) -> None:
        """Signal operator W(a) should be unitary."""
        W = _build_signal_operator(0.6)
        identity = np.eye(2, dtype=np.complex128)
        product = W.conj().T @ W
        np.testing.assert_allclose(product, identity, atol=1e-10)

    def test_signal_rotation_is_unitary(self) -> None:
        """Signal rotation should be unitary."""
        R = _build_signal_rotation(0.5)
        identity = np.eye(2, dtype=np.complex128)
        product = R.conj().T @ R
        np.testing.assert_allclose(product, identity, atol=1e-10)

    def test_signal_operator_determinant(self) -> None:
        """Signal operator should have determinant of magnitude 1."""
        W = _build_signal_operator(0.3)
        det = np.linalg.det(W)
        assert abs(abs(det) - 1.0) < 1e-10

    def test_signal_operator_boundary_zero(self) -> None:
        """W(0) should encode a=0."""
        W = _build_signal_operator(0.0)
        assert abs(W[0, 0]) < 1e-10  # (0,0) element is a=0

    def test_signal_operator_boundary_one(self) -> None:
        """W(1) should encode a=1."""
        W = _build_signal_operator(1.0)
        assert abs(W[0, 0] - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# QSP sequence tests
# ---------------------------------------------------------------------------


class TestQSPSequence:
    def test_returns_2x2_matrix(self) -> None:
        """QSP sequence should return a 2x2 matrix."""
        phases = np.array([0.1, 0.2, 0.3])
        U = qsp_sequence(phases, 0.5)
        assert U.shape == (2, 2)

    def test_result_is_unitary(self) -> None:
        """QSP sequence result should be (approximately) unitary."""
        phases = compute_qsp_phases(chebyshev_coefficients_arcsin(4))
        U = qsp_sequence(phases, 0.5)
        identity = np.eye(2, dtype=np.complex128)
        product = U.conj().T @ U
        np.testing.assert_allclose(product, identity, atol=1e-10)

    def test_varies_with_signal(self) -> None:
        """Different signal values should give different results."""
        phases = compute_qsp_phases(chebyshev_coefficients_arcsin(4))
        U1 = qsp_sequence(phases, 0.3)
        U2 = qsp_sequence(phases, 0.7)
        assert not np.allclose(U1, U2)


# ---------------------------------------------------------------------------
# QSP Amplitude Estimation tests
# ---------------------------------------------------------------------------


class TestQSPAmplitudeEstimation:
    def test_produces_valid_result(
        self, simple_estimation_problem: EstimationProblem, mock_backend: MockBackend,
    ) -> None:
        """QSP AE should return a well-formed QSPResult."""
        config = QSPConfig(polynomial_degree=4, shots=1024, seed=42)
        qsp = QSPAmplitudeEstimation(simple_estimation_problem, config, mock_backend)
        result = qsp.estimate()

        assert isinstance(result, QSPResult)
        assert 0.0 <= result.estimate <= 1.0
        assert result.wall_time_s > 0
        assert result.polynomial_degree == 4
        assert result.n_oracle_calls == 4 * 1024
        assert len(result.phase_factors) > 0

    def test_confidence_interval(
        self, simple_estimation_problem: EstimationProblem, mock_backend: MockBackend,
    ) -> None:
        """Confidence interval should be valid (low <= estimate <= high)."""
        config = QSPConfig(polynomial_degree=4, shots=2048, seed=42)
        qsp = QSPAmplitudeEstimation(simple_estimation_problem, config, mock_backend)
        result = qsp.estimate()

        ci_low, ci_high = result.confidence_interval
        assert ci_low <= ci_high
        assert ci_low >= 0.0
        assert ci_high <= 1.0

    def test_result_serializable(
        self, simple_estimation_problem: EstimationProblem, mock_backend: MockBackend,
    ) -> None:
        """QSPResult should be JSON-serializable."""
        config = QSPConfig(polynomial_degree=3, shots=512, seed=42)
        qsp = QSPAmplitudeEstimation(simple_estimation_problem, config, mock_backend)
        result = qsp.estimate()

        json_str = result.to_json()
        assert isinstance(json_str, str)
        assert "estimate" in json_str

    def test_different_degrees(
        self, simple_estimation_problem: EstimationProblem, mock_backend: MockBackend,
    ) -> None:
        """Different polynomial degrees should produce valid results."""
        for degree in [2, 5, 10]:
            config = QSPConfig(polynomial_degree=degree, shots=512, seed=42)
            qsp = QSPAmplitudeEstimation(simple_estimation_problem, config, mock_backend)
            result = qsp.estimate()
            assert 0.0 <= result.estimate <= 1.0
            assert result.polynomial_degree == degree

    def test_tiny_problem(
        self, tiny_estimation_problem: EstimationProblem, mock_backend: MockBackend,
    ) -> None:
        """Should work with a 1-qubit estimation problem."""
        config = QSPConfig(polynomial_degree=3, shots=512, seed=42)
        qsp = QSPAmplitudeEstimation(tiny_estimation_problem, config, mock_backend)
        result = qsp.estimate()
        assert isinstance(result, QSPResult)
        assert 0.0 <= result.estimate <= 1.0


# ---------------------------------------------------------------------------
# Multi-variable QSP tests
# ---------------------------------------------------------------------------


class TestMultiVariableQSP:
    def test_2_asset_estimation(self, mock_backend: MockBackend) -> None:
        """Multi-variable QSP should work for 2 assets."""
        config = MultiVariableQSPConfig(
            n_assets=2, polynomial_degree=3,
            n_qubits_per_asset=2, shots=1024, seed=42,
        )
        mv_qsp = MultiVariableQSP(config, mock_backend)
        result = mv_qsp.estimate()

        assert isinstance(result, MultiVariableQSPResult)
        assert result.n_assets == 2
        assert len(result.per_asset_estimates) == 2
        assert 0.0 <= result.estimate <= 1.0

    def test_3_asset_estimation(self, mock_backend: MockBackend) -> None:
        """Multi-variable QSP should work for 3 assets."""
        config = MultiVariableQSPConfig(
            n_assets=3, polynomial_degree=2,
            n_qubits_per_asset=2, shots=512, seed=42,
        )
        mv_qsp = MultiVariableQSP(config, mock_backend)
        result = mv_qsp.estimate()

        assert result.n_assets == 3
        assert len(result.per_asset_estimates) == 3

    def test_with_correlations(self, mock_backend: MockBackend) -> None:
        """Should handle a non-identity correlation matrix."""
        corr = np.array([[1.0, 0.5], [0.5, 1.0]])
        config = MultiVariableQSPConfig(
            n_assets=2, polynomial_degree=3,
            correlation_matrix=corr,
            n_qubits_per_asset=2, shots=512, seed=42,
        )
        mv_qsp = MultiVariableQSP(config, mock_backend)
        result = mv_qsp.estimate()

        assert isinstance(result, MultiVariableQSPResult)
        assert result.estimate >= 0.0

    def test_result_serializable(self, mock_backend: MockBackend) -> None:
        """MultiVariableQSPResult should be JSON-serializable."""
        config = MultiVariableQSPConfig(
            n_assets=2, polynomial_degree=2,
            n_qubits_per_asset=2, shots=256, seed=42,
        )
        mv_qsp = MultiVariableQSP(config, mock_backend)
        result = mv_qsp.estimate()

        json_str = result.to_json()
        assert isinstance(json_str, str)
        assert "n_assets" in json_str

    def test_default_identity_correlation(self, mock_backend: MockBackend) -> None:
        """With no correlation_matrix, should default to identity."""
        config = MultiVariableQSPConfig(
            n_assets=2, polynomial_degree=2,
            n_qubits_per_asset=2, shots=256, seed=42,
        )
        mv_qsp = MultiVariableQSP(config, mock_backend)
        np.testing.assert_allclose(mv_qsp._corr, np.eye(2))


# ---------------------------------------------------------------------------
# Config/dataclass tests
# ---------------------------------------------------------------------------


class TestConfigs:
    def test_qsp_config_defaults(self) -> None:
        config = QSPConfig()
        assert config.polynomial_degree == 10
        assert config.epsilon == 0.01
        assert config.shots == 4096
        assert config.seed == 42

    def test_qsp_result_defaults(self) -> None:
        result = QSPResult()
        assert result.estimate == 0.0
        assert result.polynomial_degree == 0
        assert result.n_oracle_calls == 0

    def test_comparison_result_defaults(self) -> None:
        result = QSPComparisonResult()
        assert result.qsp_estimate == 0.0
        assert result.true_amplitude is None
        assert result.accuracy_per_depth == {}

    def test_multi_variable_config_defaults(self) -> None:
        config = MultiVariableQSPConfig()
        assert config.n_assets == 2
        assert config.polynomial_degree == 6
        assert config.correlation_matrix is None
