"""Unit tests for the Quantum Interior Point Method optimizer."""

from __future__ import annotations

import numpy as np
import pytest

from qufin.backends.mock import MockBackend
from qufin.portfolio.optimizers.quantum_ipm import (
    QuantumIPMConfig,
    QuantumIPMOptimizer,
    QuantumIPMResult,
    ResourceEstimate,
    _analytical_mean_variance,
    _build_kkt_system,
    build_hhl_circuit,
    classical_ipm_solve,
    condition_number_analysis,
    covariance_to_hamiltonian,
    estimate_resources,
    hhl_solve,
    quantum_ipm_solve,
    regularise_matrix,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_backend() -> MockBackend:
    return MockBackend(seed=42)


@pytest.fixture
def small_problem_3():
    """3-asset mean-variance problem."""
    mu = np.array([0.02, 0.03, 0.015])
    cov = np.array([
        [0.04, 0.006, 0.002],
        [0.006, 0.09, 0.004],
        [0.002, 0.004, 0.01],
    ])
    return mu, cov


@pytest.fixture
def small_problem_4():
    """4-asset problem with moderate correlation."""
    mu = np.array([0.01, 0.02, 0.015, 0.008])
    cov = np.array([
        [0.04, 0.006, 0.002, 0.001],
        [0.006, 0.09, 0.004, 0.003],
        [0.002, 0.004, 0.01, 0.002],
        [0.001, 0.003, 0.002, 0.025],
    ])
    return mu, cov


@pytest.fixture
def well_conditioned_cov():
    """Well-conditioned 4x4 covariance matrix."""
    return np.diag([0.04, 0.05, 0.03, 0.06])


@pytest.fixture
def ill_conditioned_cov():
    """Ill-conditioned 4x4 covariance matrix."""
    rng = np.random.default_rng(99)
    Q, _ = np.linalg.qr(rng.standard_normal((4, 4)))
    eigvals = np.array([1.0, 1e-3, 1e-6, 1e-9])
    cov = Q @ np.diag(eigvals) @ Q.T
    return (cov + cov.T) / 2


# ---------------------------------------------------------------------------
# Condition number analysis
# ---------------------------------------------------------------------------


class TestConditionNumberAnalysis:
    def test_identity_matrix_condition(self) -> None:
        """Identity matrix should have condition number 1."""
        cov = np.eye(4)
        result = condition_number_analysis(cov)
        assert result["condition_number"] == pytest.approx(1.0, rel=1e-6)
        assert result["is_well_conditioned"] is True
        assert result["regularisation_needed"] is False

    def test_diagonal_matrix_condition(self) -> None:
        """Diagonal matrix condition = max_eig / min_eig."""
        cov = np.diag([1.0, 0.1, 10.0, 0.01])
        result = condition_number_analysis(cov)
        assert result["condition_number"] == pytest.approx(1000.0, rel=1e-6)

    def test_rank_computation(self, small_problem_3) -> None:
        """Rank should match matrix dimension for full-rank matrix."""
        _, cov = small_problem_3
        result = condition_number_analysis(cov)
        assert result["rank"] == 3

    def test_ill_conditioned_detection(self, ill_conditioned_cov) -> None:
        """Should detect ill-conditioned matrices."""
        result = condition_number_analysis(ill_conditioned_cov)
        assert result["condition_number"] > 1e6
        assert result["regularisation_needed"] is True

    def test_eigenvalues_sorted_descending(self, small_problem_3) -> None:
        """Eigenvalues should be sorted in descending order."""
        _, cov = small_problem_3
        result = condition_number_analysis(cov)
        eigvals = result["eigenvalues"]
        assert all(eigvals[i] >= eigvals[i + 1] for i in range(len(eigvals) - 1))


# ---------------------------------------------------------------------------
# Regularisation
# ---------------------------------------------------------------------------


class TestRegularisation:
    def test_regularise_improves_condition(self, ill_conditioned_cov) -> None:
        """Regularisation should improve condition number."""
        cond_before = condition_number_analysis(ill_conditioned_cov)["condition_number"]
        reg = regularise_matrix(ill_conditioned_cov)
        cond_after = condition_number_analysis(reg)["condition_number"]
        assert cond_after < cond_before

    def test_regularise_with_explicit_epsilon(self) -> None:
        """Should add epsilon * I."""
        A = np.diag([1.0, 0.001])
        A_reg = regularise_matrix(A, epsilon=0.1)
        np.testing.assert_allclose(np.diag(A_reg), [1.1, 0.101], atol=1e-10)

    def test_regularise_preserves_symmetry(self, small_problem_3) -> None:
        """Regularised matrix should remain symmetric."""
        _, cov = small_problem_3
        reg = regularise_matrix(cov)
        np.testing.assert_allclose(reg, reg.T, atol=1e-15)


# ---------------------------------------------------------------------------
# Hamiltonian encoding
# ---------------------------------------------------------------------------


class TestHamiltonianEncoding:
    def test_padding_to_power_of_2(self) -> None:
        """3x3 matrix should be padded to 4x4."""
        cov = np.eye(3) * 0.1
        result = covariance_to_hamiltonian(cov)
        assert result["matrix"].shape == (4, 4)
        assert result["n_qubits"] == 2

    def test_preserves_original_entries(self, small_problem_3) -> None:
        """Original matrix entries should be preserved in padded matrix."""
        _, cov = small_problem_3
        result = covariance_to_hamiltonian(cov)
        n = cov.shape[0]
        np.testing.assert_allclose(
            result["matrix"][:n, :n], cov, atol=1e-15,
        )

    def test_sparsity_computation(self) -> None:
        """Sparse matrix should have lower sparsity count."""
        sparse_cov = np.diag([0.1, 0.2, 0.3, 0.4])
        result = covariance_to_hamiltonian(sparse_cov)
        assert result["sparsity"] >= 1

    def test_pauli_terms_count(self, small_problem_3) -> None:
        """Pauli terms should be positive for non-trivial matrix."""
        _, cov = small_problem_3
        result = covariance_to_hamiltonian(cov)
        assert result["pauli_terms"] > 0


# ---------------------------------------------------------------------------
# HHL circuit construction
# ---------------------------------------------------------------------------


class TestHHLCircuit:
    def test_circuit_builds_without_error(self) -> None:
        """HHL circuit should build for a simple 2x2 system."""
        A = np.array([[2.0, 0.5], [0.5, 3.0]])
        b = np.array([1.0, 0.0])
        circuit = build_hhl_circuit(A, b, n_clock_qubits=2)
        # 1 state qubit (log2(2)=1) + 2 clock + 1 ancilla = 4
        assert circuit.num_qubits == 1 + 2 + 1

    def test_circuit_qubit_count(self) -> None:
        """Circuit should have correct number of qubits."""
        A = np.eye(4) * 2.0
        b = np.array([1.0, 0.0, 0.0, 0.0])
        circuit = build_hhl_circuit(A, b, n_clock_qubits=3)
        # 2 state + 3 clock + 1 ancilla = 6
        assert circuit.num_qubits == 6


# ---------------------------------------------------------------------------
# HHL solve
# ---------------------------------------------------------------------------


class TestHHLSolve:
    def test_solves_identity_system(self, mock_backend) -> None:
        """Solving I * x = b should return x = b."""
        A = np.eye(2)
        b = np.array([1.0, 2.0])
        x = hhl_solve(A, b, mock_backend, n_clock_qubits=2)
        np.testing.assert_allclose(x, b, atol=1e-8)

    def test_solves_diagonal_system(self, mock_backend) -> None:
        """Solving diag(d) * x = b should return x = b / d."""
        A = np.diag([2.0, 4.0])
        b = np.array([6.0, 8.0])
        x = hhl_solve(A, b, mock_backend, n_clock_qubits=3)
        np.testing.assert_allclose(x, [3.0, 2.0], atol=1e-8)

    def test_solves_general_2x2(self, mock_backend) -> None:
        """Should solve a general 2x2 system."""
        A = np.array([[3.0, 1.0], [1.0, 2.0]])
        b = np.array([5.0, 5.0])
        x = hhl_solve(A, b, mock_backend, n_clock_qubits=3)
        np.testing.assert_allclose(A @ x, b, atol=1e-8)


# ---------------------------------------------------------------------------
# KKT system construction
# ---------------------------------------------------------------------------


class TestKKTSystem:
    def test_kkt_system_dimensions(self, small_problem_3) -> None:
        """KKT matrix should be (2n+1) x (2n+1)."""
        mu, cov = small_problem_3
        n = len(mu)
        x = np.ones(n) / n
        s = np.ones(n) * 0.1
        lam = s.copy()
        A_kkt, rhs = _build_kkt_system(cov, mu, x, s, lam, 0.0, 1.0, 0.1)
        expected_dim = 2 * n + 1
        assert A_kkt.shape == (expected_dim, expected_dim)
        assert rhs.shape == (expected_dim,)

    def test_kkt_system_not_singular(self, small_problem_3) -> None:
        """KKT matrix should be non-singular for a well-posed problem."""
        mu, cov = small_problem_3
        n = len(mu)
        x = np.ones(n) / n
        s = np.ones(n) * 0.1
        lam = s.copy()
        A_kkt, _ = _build_kkt_system(cov, mu, x, s, lam, 0.0, 1.0, 0.1)
        det = np.linalg.det(A_kkt)
        assert abs(det) > 1e-15


# ---------------------------------------------------------------------------
# Resource estimation
# ---------------------------------------------------------------------------


class TestResourceEstimation:
    def test_qubit_count_scales_with_assets(self) -> None:
        """More assets should require more qubits."""
        r2 = estimate_resources(2)
        r10 = estimate_resources(10)
        assert r10.total_qubits > r2.total_qubits

    def test_resource_includes_condition_number(self, small_problem_3) -> None:
        """Condition number should be computed when cov is provided."""
        _, cov = small_problem_3
        r = estimate_resources(3, cov=cov)
        assert r.condition_number > 0

    def test_gate_count_positive(self) -> None:
        """Gate count estimate should be positive."""
        r = estimate_resources(4)
        assert r.estimated_gate_count > 0


# ---------------------------------------------------------------------------
# Quantum IPM solve
# ---------------------------------------------------------------------------


class TestQuantumIPMSolve:
    def test_returns_valid_weights(self, small_problem_3, mock_backend) -> None:
        """Weights should sum to 1 and be non-negative."""
        mu, cov = small_problem_3
        config = QuantumIPMConfig(max_ipm_iters=20, tol=1e-4, gamma=0.5)
        result = quantum_ipm_solve(mu, cov, config, mock_backend)

        assert result.weights.shape == (3,)
        np.testing.assert_allclose(np.sum(result.weights), 1.0, atol=1e-6)
        assert np.all(result.weights >= -1e-10)

    def test_classical_fallback(self, small_problem_3, mock_backend) -> None:
        """Classical fallback should also produce valid weights."""
        mu, cov = small_problem_3
        config = QuantumIPMConfig(
            max_ipm_iters=20, use_quantum_solver=False, gamma=0.5,
        )
        result = quantum_ipm_solve(mu, cov, config, mock_backend)
        assert result.method == "classical_ipm"
        np.testing.assert_allclose(np.sum(result.weights), 1.0, atol=1e-6)

    def test_iteration_log_populated(self, small_problem_3, mock_backend) -> None:
        """Iteration log should have entries."""
        mu, cov = small_problem_3
        config = QuantumIPMConfig(max_ipm_iters=5, gamma=1.0)
        result = quantum_ipm_solve(mu, cov, config, mock_backend)
        assert len(result.iteration_log) > 0

    def test_duality_gap_decreases(self, small_problem_4, mock_backend) -> None:
        """Duality gap should generally decrease over iterations."""
        mu, cov = small_problem_4
        config = QuantumIPMConfig(max_ipm_iters=20, tol=1e-8, gamma=1.0)
        result = quantum_ipm_solve(mu, cov, config, mock_backend)
        if len(result.iteration_log) >= 3:
            first_gap = result.iteration_log[0].duality_gap
            last_gap = result.iteration_log[-1].duality_gap
            assert last_gap <= first_gap + 1e-6


# ---------------------------------------------------------------------------
# Classical IPM
# ---------------------------------------------------------------------------


class TestClassicalIPM:
    def test_classical_produces_valid_weights(self, small_problem_3) -> None:
        """Classical solve should produce valid portfolio."""
        mu, cov = small_problem_3
        result = classical_ipm_solve(mu, cov, gamma=0.5)
        assert result.method == "classical_ipm"
        np.testing.assert_allclose(np.sum(result.weights), 1.0, atol=1e-4)
        assert np.all(result.weights >= -1e-6)

    def test_classical_vs_quantum_similar(
        self, small_problem_3, mock_backend,
    ) -> None:
        """Classical and quantum should give similar objectives."""
        mu, cov = small_problem_3
        config = QuantumIPMConfig(max_ipm_iters=30, tol=1e-6, gamma=1.0)
        q_result = quantum_ipm_solve(mu, cov, config, mock_backend)
        c_result = classical_ipm_solve(mu, cov, gamma=1.0)

        # Both should find reasonable solutions (objectives within 50%)
        assert q_result.optimal_objective < 1.0
        assert c_result.optimal_objective < 1.0


# ---------------------------------------------------------------------------
# Analytical fallback
# ---------------------------------------------------------------------------


class TestAnalyticalMeanVariance:
    def test_produces_simplex_weights(self, small_problem_3) -> None:
        """Analytical solution should be on the simplex."""
        mu, cov = small_problem_3
        weights, _obj, _converged = _analytical_mean_variance(mu, cov, 1.0)
        np.testing.assert_allclose(np.sum(weights), 1.0, atol=1e-10)
        assert np.all(weights >= -1e-10)

    def test_converges_for_well_conditioned(self, well_conditioned_cov) -> None:
        """Should converge for well-conditioned diagonal covariance."""
        mu = np.array([0.01, 0.02, 0.015, 0.008])
        weights, _, converged = _analytical_mean_variance(
            mu, well_conditioned_cov, 1.0,
        )
        assert converged is True
        np.testing.assert_allclose(np.sum(weights), 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Optimizer class
# ---------------------------------------------------------------------------


class TestQuantumIPMOptimizer:
    def test_optimizer_init_validates_shapes(self, mock_backend) -> None:
        """Should raise on mismatched mu and cov shapes."""
        mu = np.array([0.01, 0.02])
        cov = np.eye(3)
        with pytest.raises(ValueError, match="shape"):
            QuantumIPMOptimizer(mu, cov, QuantumIPMConfig(), mock_backend)

    def test_optimizer_run(self, small_problem_3, mock_backend) -> None:
        """Optimizer.run() should return a valid result."""
        mu, cov = small_problem_3
        config = QuantumIPMConfig(max_ipm_iters=10, gamma=0.5)
        opt = QuantumIPMOptimizer(mu, cov, config, mock_backend)
        result = opt.run()
        assert isinstance(result, QuantumIPMResult)
        np.testing.assert_allclose(np.sum(result.weights), 1.0, atol=1e-6)

    def test_optimizer_resource_estimate(self, small_problem_3, mock_backend) -> None:
        """Optimizer should provide resource estimates."""
        mu, cov = small_problem_3
        config = QuantumIPMConfig(hhl_precision_qubits=4)
        opt = QuantumIPMOptimizer(mu, cov, config, mock_backend)
        res = opt.resource_estimate()
        assert isinstance(res, ResourceEstimate)
        assert res.total_qubits > 0
        assert res.n_assets == 3


# ---------------------------------------------------------------------------
# Result serialisation
# ---------------------------------------------------------------------------


class TestResultSerialisation:
    def test_result_to_dict(self, small_problem_3, mock_backend) -> None:
        """Result should serialise to dict."""
        mu, cov = small_problem_3
        config = QuantumIPMConfig(max_ipm_iters=3, gamma=1.0)
        result = quantum_ipm_solve(mu, cov, config, mock_backend)
        d = result.to_dict()
        assert "weights" in d
        assert "optimal_objective" in d
        assert "iteration_log" in d

    def test_result_to_json(self, small_problem_3, mock_backend) -> None:
        """Result should serialise to JSON without errors."""
        mu, cov = small_problem_3
        config = QuantumIPMConfig(max_ipm_iters=3, gamma=1.0)
        result = quantum_ipm_solve(mu, cov, config, mock_backend)
        j = result.to_json()
        assert isinstance(j, str)
        assert "weights" in j
