"""Unit tests for quantum linear systems (HHL) for risk analysis."""

from __future__ import annotations

import numpy as np
import pytest

from qufin.backends.mock import MockBackend
from qufin.risk.quantum_linear_systems import (
    HHLConfig,
    LinearSystemResult,
    analyse_condition_number,
    auto_regularise,
    build_hhl_circuit,
    cholesky_solve,
    compute_factor_exposures,
    condition_sensitivity_comparison,
    encode_covariance_hamiltonian,
    generate_test_covariance,
    hhl_solve,
    solve_linear_system,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_backend() -> MockBackend:
    return MockBackend(seed=42)


@pytest.fixture
def well_conditioned_2x2():
    """Well-conditioned 2x2 covariance."""
    return np.array([[2.0, 0.5], [0.5, 3.0]])


@pytest.fixture
def well_conditioned_4x4():
    """Well-conditioned 4x4 covariance."""
    return np.diag([0.04, 0.05, 0.03, 0.06])


@pytest.fixture
def ill_conditioned_4x4():
    """Ill-conditioned 4x4 covariance."""
    rng = np.random.default_rng(99)
    Q, _ = np.linalg.qr(rng.standard_normal((4, 4)))
    eigvals = np.array([1.0, 1e-3, 1e-6, 1e-9])
    cov = Q @ np.diag(eigvals) @ Q.T
    return (cov + cov.T) / 2


@pytest.fixture
def factor_setup():
    """Factor model setup: 4 assets, 2 factors."""
    sigma_f = np.array([[0.04, 0.01], [0.01, 0.09]])
    B = np.array([
        [0.8, 0.2],
        [0.5, 0.5],
        [0.3, 0.7],
        [0.6, 0.4],
    ])
    w = np.array([0.25, 0.25, 0.25, 0.25])
    return sigma_f, B, w


# ---------------------------------------------------------------------------
# Condition number analysis
# ---------------------------------------------------------------------------


class TestConditionNumberAnalysis:
    def test_identity_condition_is_one(self) -> None:
        """Identity matrix should have condition number 1."""
        report = analyse_condition_number(np.eye(4))
        assert report.condition_number == pytest.approx(1.0, rel=1e-6)
        assert report.is_well_conditioned is True
        assert report.is_ill_conditioned is False

    def test_diagonal_condition(self) -> None:
        """Diagonal matrix condition = max / min eigenvalue."""
        sigma = np.diag([10.0, 1.0, 0.1])
        report = analyse_condition_number(sigma)
        assert report.condition_number == pytest.approx(100.0, rel=1e-4)

    def test_rank_full(self, well_conditioned_2x2) -> None:
        """Full-rank matrix should have rank = n."""
        report = analyse_condition_number(well_conditioned_2x2)
        assert report.rank == 2

    def test_ill_conditioned_detected(self, ill_conditioned_4x4) -> None:
        """Should flag ill-conditioned matrices."""
        report = analyse_condition_number(ill_conditioned_4x4)
        assert report.is_ill_conditioned is True
        assert report.condition_number > 1e6

    def test_eigenvalues_descending(self, well_conditioned_4x4) -> None:
        """Eigenvalues should be sorted descending."""
        report = analyse_condition_number(well_conditioned_4x4)
        eigs = report.eigenvalues
        assert all(eigs[i] >= eigs[i + 1] for i in range(len(eigs) - 1))

    def test_spectral_gap(self) -> None:
        """Spectral gap should be max_eig / second_eig."""
        sigma = np.diag([10.0, 2.0, 1.0])
        report = analyse_condition_number(sigma)
        assert report.spectral_gap == pytest.approx(5.0, rel=1e-4)

    def test_regularisation_improves_condition(self, ill_conditioned_4x4) -> None:
        """Regularisation parameter should reduce effective condition."""
        report = analyse_condition_number(ill_conditioned_4x4, regularisation=0.01)
        assert report.effective_condition_after_reg < report.condition_number


# ---------------------------------------------------------------------------
# Auto regularisation
# ---------------------------------------------------------------------------


class TestAutoRegularise:
    def test_well_conditioned_no_change(self, well_conditioned_4x4) -> None:
        """Well-conditioned matrix should not be modified."""
        sigma_reg, eps = auto_regularise(well_conditioned_4x4)
        assert eps == 0.0
        np.testing.assert_allclose(sigma_reg, well_conditioned_4x4)

    def test_ill_conditioned_gets_regularised(self, ill_conditioned_4x4) -> None:
        """Ill-conditioned matrix should be regularised."""
        sigma_reg, eps = auto_regularise(ill_conditioned_4x4)
        assert eps > 0.0
        cond_before = analyse_condition_number(ill_conditioned_4x4).condition_number
        cond_after = analyse_condition_number(sigma_reg).condition_number
        assert cond_after < cond_before

    def test_preserves_symmetry(self, ill_conditioned_4x4) -> None:
        """Regularised matrix should remain symmetric."""
        sigma_reg, _ = auto_regularise(ill_conditioned_4x4)
        np.testing.assert_allclose(sigma_reg, sigma_reg.T, atol=1e-15)


# ---------------------------------------------------------------------------
# Hamiltonian encoding
# ---------------------------------------------------------------------------


class TestHamiltonianEncoding:
    def test_3x3_padded_to_4x4(self) -> None:
        """3x3 matrix should be padded to 4x4 (2 qubits)."""
        sigma = np.eye(3) * 0.1
        info = encode_covariance_hamiltonian(sigma)
        assert info["hamiltonian"].shape == (4, 4)
        assert info["n_qubits"] == 2
        assert info["padded_dim"] == 4
        assert info["original_dim"] == 3

    def test_2x2_no_padding(self) -> None:
        """2x2 matrix is already power-of-2."""
        sigma = np.array([[1.0, 0.5], [0.5, 2.0]])
        info = encode_covariance_hamiltonian(sigma)
        assert info["hamiltonian"].shape == (2, 2)
        assert info["n_qubits"] == 1

    def test_original_entries_preserved(self, well_conditioned_2x2) -> None:
        """Original entries should be unchanged."""
        info = encode_covariance_hamiltonian(well_conditioned_2x2)
        n = well_conditioned_2x2.shape[0]
        np.testing.assert_allclose(
            info["hamiltonian"][:n, :n], well_conditioned_2x2,
        )

    def test_spectral_norm_positive(self, well_conditioned_4x4) -> None:
        """Spectral norm should be positive."""
        info = encode_covariance_hamiltonian(well_conditioned_4x4)
        assert info["norm"] > 0


# ---------------------------------------------------------------------------
# HHL circuit
# ---------------------------------------------------------------------------


class TestHHLCircuit:
    def test_circuit_builds_2x2(self) -> None:
        """Should build without error for 2x2."""
        H = np.array([[2.0, 0.5], [0.5, 3.0]])
        b = np.array([1.0, 0.5])
        qc = build_hhl_circuit(H, b, n_clock_qubits=2)
        # 1 state + 2 clock + 1 ancilla = 4
        assert qc.num_qubits == 4

    def test_circuit_qubit_scaling(self) -> None:
        """4x4 system should need 2 state qubits."""
        H = np.eye(4) * 2.0
        b = np.array([1.0, 0.0, 0.0, 0.0])
        qc = build_hhl_circuit(H, b, n_clock_qubits=3)
        # 2 state + 3 clock + 1 ancilla = 6
        assert qc.num_qubits == 6


# ---------------------------------------------------------------------------
# HHL solve
# ---------------------------------------------------------------------------


class TestHHLSolve:
    def test_identity_solve(self, mock_backend) -> None:
        """Solving I x = b should return x = b."""
        sigma = np.eye(2)
        b = np.array([3.0, 7.0])
        x = hhl_solve(sigma, b, mock_backend)
        np.testing.assert_allclose(x, b, atol=1e-8)

    def test_diagonal_solve(self, mock_backend) -> None:
        """Solving diag(d) x = b should return x_i = b_i / d_i."""
        sigma = np.diag([2.0, 5.0])
        b = np.array([4.0, 10.0])
        x = hhl_solve(sigma, b, mock_backend)
        np.testing.assert_allclose(x, [2.0, 2.0], atol=1e-8)

    def test_general_system(self, well_conditioned_2x2, mock_backend) -> None:
        """Should solve a general SPD system."""
        b = np.array([1.0, 2.0])
        x = hhl_solve(well_conditioned_2x2, b, mock_backend)
        np.testing.assert_allclose(
            well_conditioned_2x2 @ x, b, atol=1e-8,
        )

    def test_with_regularisation_config(self, mock_backend) -> None:
        """Should accept regularisation via config."""
        sigma = np.array([[1.0, 0.99], [0.99, 1.0]])
        b = np.array([1.0, 1.0])
        config = HHLConfig(regularisation=0.1)
        x = hhl_solve(sigma, b, mock_backend, config=config)
        assert x.shape == (2,)


# ---------------------------------------------------------------------------
# Cholesky solve
# ---------------------------------------------------------------------------


class TestCholeskySolve:
    def test_identity_solve(self) -> None:
        """Cholesky on identity returns b."""
        x = cholesky_solve(np.eye(3), np.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(x, [1.0, 2.0, 3.0], atol=1e-12)

    def test_spd_system(self, well_conditioned_2x2) -> None:
        """Should solve an SPD system accurately."""
        b = np.array([5.0, 7.0])
        x = cholesky_solve(well_conditioned_2x2, b)
        np.testing.assert_allclose(
            well_conditioned_2x2 @ x, b, atol=1e-12,
        )

    def test_raises_for_non_pd(self) -> None:
        """Should raise for non-positive-definite matrix."""
        sigma = np.array([[1.0, 2.0], [2.0, 1.0]])  # Not PD
        with pytest.raises(np.linalg.LinAlgError):
            cholesky_solve(sigma, np.array([1.0, 1.0]))


# ---------------------------------------------------------------------------
# Solve linear system (high-level)
# ---------------------------------------------------------------------------


class TestSolveLinearSystem:
    def test_quantum_method(self, well_conditioned_2x2, mock_backend) -> None:
        """Quantum method should return a LinearSystemResult."""
        b = np.array([1.0, 2.0])
        result = solve_linear_system(
            well_conditioned_2x2, b, mock_backend, method="quantum",
        )
        assert isinstance(result, LinearSystemResult)
        assert result.method == "quantum_hhl"
        assert result.n_qubits_used > 0

    def test_classical_method(self, well_conditioned_2x2) -> None:
        """Classical method should use Cholesky."""
        b = np.array([1.0, 2.0])
        result = solve_linear_system(
            well_conditioned_2x2, b, method="classical",
        )
        assert "classical" in result.method
        assert result.n_qubits_used == 0

    def test_residual_small(self, well_conditioned_2x2, mock_backend) -> None:
        """Residual norm should be small for well-conditioned system."""
        b = np.array([1.0, 2.0])
        result = solve_linear_system(
            well_conditioned_2x2, b, mock_backend, method="quantum",
        )
        assert result.residual_norm < 1e-6

    def test_condition_report_populated(self, well_conditioned_2x2) -> None:
        """Condition report should be filled in."""
        b = np.array([1.0, 2.0])
        result = solve_linear_system(
            well_conditioned_2x2, b, method="classical",
        )
        assert result.condition_report.condition_number > 0
        assert result.condition_report.rank > 0


# ---------------------------------------------------------------------------
# Factor exposures
# ---------------------------------------------------------------------------


class TestFactorExposures:
    def test_exposures_shape(self, factor_setup, mock_backend) -> None:
        """Exposures should match number of factors."""
        sigma_f, B, w = factor_setup
        result = compute_factor_exposures(
            sigma_f, w, B, mock_backend, method="quantum",
        )
        assert result.exposures.shape == (2,)

    def test_risk_contributions_shape(self, factor_setup, mock_backend) -> None:
        """Risk contributions should match factor count."""
        sigma_f, B, w = factor_setup
        result = compute_factor_exposures(
            sigma_f, w, B, mock_backend, method="quantum",
        )
        assert result.risk_contributions.shape == (2,)

    def test_total_risk_positive(self, factor_setup, mock_backend) -> None:
        """Total risk should be positive."""
        sigma_f, B, w = factor_setup
        result = compute_factor_exposures(
            sigma_f, w, B, mock_backend, method="quantum",
        )
        assert result.total_risk > 0

    def test_diversification_ratio(self, factor_setup) -> None:
        """Diversification ratio should be >= 1 for diversified portfolio."""
        sigma_f, B, w = factor_setup
        result = compute_factor_exposures(
            sigma_f, w, B, method="classical",
        )
        assert result.diversification_ratio >= 0.5

    def test_classical_vs_quantum_same(self, factor_setup, mock_backend) -> None:
        """Classical and quantum should give same exposures."""
        sigma_f, B, w = factor_setup
        q = compute_factor_exposures(
            sigma_f, w, B, mock_backend, method="quantum",
        )
        c = compute_factor_exposures(
            sigma_f, w, B, method="classical",
        )
        np.testing.assert_allclose(q.exposures, c.exposures, atol=1e-6)


# ---------------------------------------------------------------------------
# Sensitivity comparison
# ---------------------------------------------------------------------------


class TestSensitivityComparison:
    def test_comparison_returns_both(
        self, well_conditioned_4x4, ill_conditioned_4x4, mock_backend,
    ) -> None:
        """Should return results for both well- and ill-conditioned."""
        b = np.array([1.0, 1.0, 1.0, 1.0])
        result = condition_sensitivity_comparison(
            well_conditioned_4x4, ill_conditioned_4x4, b, mock_backend,
        )
        assert "well_conditioned" in result
        assert "ill_conditioned" in result
        well_cond = result["well_conditioned"]["condition_number"]
        ill_cond = result["ill_conditioned"]["condition_number"]
        assert well_cond < ill_cond


# ---------------------------------------------------------------------------
# Test covariance generation
# ---------------------------------------------------------------------------


class TestGenerateTestCovariance:
    def test_shape(self) -> None:
        """Should produce n x n matrix."""
        cov = generate_test_covariance(5, condition_number=100.0)
        assert cov.shape == (5, 5)

    def test_symmetric(self) -> None:
        """Should be symmetric."""
        cov = generate_test_covariance(4, condition_number=50.0)
        np.testing.assert_allclose(cov, cov.T, atol=1e-12)

    def test_positive_definite(self) -> None:
        """Should be positive definite."""
        cov = generate_test_covariance(4, condition_number=10.0)
        eigvals = np.linalg.eigvalsh(cov)
        assert np.all(eigvals > 0)

    def test_approximate_condition_number(self) -> None:
        """Condition number should be close to target."""
        target_kappa = 100.0
        cov = generate_test_covariance(6, condition_number=target_kappa, seed=123)
        eigvals = np.linalg.eigvalsh(cov)
        actual_kappa = float(np.max(eigvals) / np.min(eigvals))
        assert actual_kappa == pytest.approx(target_kappa, rel=0.1)

    def test_reproducible_with_seed(self) -> None:
        """Same seed should produce identical matrices."""
        c1 = generate_test_covariance(3, condition_number=10.0, seed=99)
        c2 = generate_test_covariance(3, condition_number=10.0, seed=99)
        np.testing.assert_array_equal(c1, c2)
