"""Unit tests for qufin.risk.quantum_entropy module."""

from __future__ import annotations

import numpy as np
import pytest

from qufin.risk.quantum_entropy import (
    EntropyResult,
    RelativeEntropyResult,
    _to_density_matrix,
    portfolio_diversification_score,
    quantum_relative_entropy,
    shannon_entropy,
    von_neumann_entropy,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def identity_cov() -> np.ndarray:
    """Identity covariance (maximally diversified)."""
    return np.eye(4)


@pytest.fixture
def rank1_cov() -> np.ndarray:
    """Rank-1 covariance (single factor)."""
    v = np.array([1.0, 1.0, 1.0, 1.0])
    return np.outer(v, v)


@pytest.fixture
def realistic_cov() -> np.ndarray:
    """Realistic 3x3 covariance matrix."""
    return np.array([
        [0.04, 0.006, 0.002],
        [0.006, 0.09, 0.004],
        [0.002, 0.004, 0.01],
    ])


# ---------------------------------------------------------------------------
# Density matrix conversion
# ---------------------------------------------------------------------------


class TestToDensityMatrix:
    def test_trace_one(self, identity_cov: np.ndarray) -> None:
        rho = _to_density_matrix(identity_cov)
        assert np.trace(rho) == pytest.approx(1.0)

    def test_positive_semidefinite(self, realistic_cov: np.ndarray) -> None:
        rho = _to_density_matrix(realistic_cov)
        eigvals = np.linalg.eigvalsh(rho)
        assert all(e >= -1e-10 for e in eigvals)

    def test_symmetric(self, realistic_cov: np.ndarray) -> None:
        rho = _to_density_matrix(realistic_cov)
        assert np.allclose(rho, rho.T)

    def test_zero_matrix(self) -> None:
        """Zero matrix should return uniform density."""
        rho = _to_density_matrix(np.zeros((3, 3)))
        assert np.trace(rho) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Von Neumann entropy
# ---------------------------------------------------------------------------


class TestVonNeumannEntropy:
    def test_returns_entropy_result(self, identity_cov: np.ndarray) -> None:
        result = von_neumann_entropy(identity_cov)
        assert isinstance(result, EntropyResult)

    def test_identity_max_entropy(self, identity_cov: np.ndarray) -> None:
        """Identity matrix should have maximum entropy = log(n)."""
        result = von_neumann_entropy(identity_cov)
        assert result.entropy == pytest.approx(np.log(4), rel=1e-6)

    def test_normalised_entropy_identity(self, identity_cov: np.ndarray) -> None:
        result = von_neumann_entropy(identity_cov)
        assert result.normalised_entropy == pytest.approx(1.0, rel=1e-6)

    def test_rank1_low_entropy(self, rank1_cov: np.ndarray) -> None:
        """Rank-1 matrix should have zero entropy."""
        result = von_neumann_entropy(rank1_cov)
        assert result.entropy == pytest.approx(0.0, abs=1e-8)

    def test_effective_rank_identity(self, identity_cov: np.ndarray) -> None:
        result = von_neumann_entropy(identity_cov)
        assert result.effective_rank == pytest.approx(4.0, rel=1e-6)

    def test_entropy_non_negative(self, realistic_cov: np.ndarray) -> None:
        result = von_neumann_entropy(realistic_cov)
        assert result.entropy >= 0.0

    def test_method_label(self, identity_cov: np.ndarray) -> None:
        result = von_neumann_entropy(identity_cov)
        assert result.method == "von_neumann"

    def test_1x1_matrix(self) -> None:
        result = von_neumann_entropy(np.array([[5.0]]))
        assert result.entropy == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Quantum relative entropy
# ---------------------------------------------------------------------------


class TestQuantumRelativeEntropy:
    def test_same_matrices_zero(self, identity_cov: np.ndarray) -> None:
        """S(rho || rho) = 0."""
        result = quantum_relative_entropy(identity_cov, identity_cov)
        assert isinstance(result, RelativeEntropyResult)
        assert result.divergence == pytest.approx(0.0, abs=1e-8)

    def test_non_negative(self, realistic_cov: np.ndarray, identity_cov: np.ndarray) -> None:
        cov_3 = identity_cov[:3, :3]
        result = quantum_relative_entropy(realistic_cov, cov_3)
        assert result.divergence >= -1e-10

    def test_symmetrised(self, realistic_cov: np.ndarray) -> None:
        cov2 = np.eye(3) * 0.05
        result = quantum_relative_entropy(realistic_cov, cov2)
        if result.is_finite:
            assert result.symmetrised >= 0.0

    def test_identical_matrices_symmetrised_zero(self, realistic_cov: np.ndarray) -> None:
        result = quantum_relative_entropy(realistic_cov, realistic_cov)
        assert result.symmetrised == pytest.approx(0.0, abs=1e-8)

    def test_different_matrices_positive(self) -> None:
        A = np.diag([1.0, 2.0, 3.0])
        B = np.diag([3.0, 2.0, 1.0])
        result = quantum_relative_entropy(A, B)
        assert result.is_finite
        assert result.divergence >= 0.0


# ---------------------------------------------------------------------------
# Shannon entropy
# ---------------------------------------------------------------------------


class TestShannonEntropy:
    def test_uniform_weights(self) -> None:
        w = np.ones(4) / 4
        result = shannon_entropy(w)
        assert result.entropy == pytest.approx(np.log(4), rel=1e-6)

    def test_concentrated_weights(self) -> None:
        w = np.array([1.0, 0.0, 0.0, 0.0])
        result = shannon_entropy(w)
        assert result.entropy == pytest.approx(0.0, abs=1e-10)

    def test_method_label(self) -> None:
        result = shannon_entropy(np.ones(3))
        assert result.method == "shannon"

    def test_zero_weights(self) -> None:
        result = shannon_entropy(np.zeros(3))
        assert result.entropy == 0.0

    def test_normalised_entropy(self) -> None:
        w = np.ones(5) / 5
        result = shannon_entropy(w)
        assert result.normalised_entropy == pytest.approx(1.0, rel=1e-6)


# ---------------------------------------------------------------------------
# Diversification score
# ---------------------------------------------------------------------------


class TestDiversificationScore:
    def test_returns_dict(self, realistic_cov: np.ndarray) -> None:
        result = portfolio_diversification_score(realistic_cov)
        assert isinstance(result, dict)
        assert "von_neumann" in result
        assert "shannon" in result

    def test_default_equal_weights(self, realistic_cov: np.ndarray) -> None:
        result = portfolio_diversification_score(realistic_cov)
        assert result["normalised_shannon"] == pytest.approx(1.0, rel=1e-6)

    def test_custom_weights(self, realistic_cov: np.ndarray) -> None:
        w = np.array([0.5, 0.3, 0.2])
        result = portfolio_diversification_score(realistic_cov, weights=w)
        assert result["shannon"] > 0
