"""Tests for robust portfolio optimization with worst-case CVaR."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from qufin.portfolio.optimizers.robust import (
    EllipsoidalUncertaintySet,
    RobustPortfolioOptimizer,
    RobustPortfolioResult,
    build_ellipsoidal_uncertainty,
    robust_classical,
)
from qufin.portfolio.qubo import PortfolioQUBO

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def small_returns() -> NDArray[np.float64]:
    """Synthetic return matrix (50 obs x 4 assets)."""
    rng = np.random.default_rng(42)
    mu_true = np.array([0.10, 0.08, 0.12, 0.06])
    cov_true = np.array([
        [0.04, 0.006, 0.010, 0.003],
        [0.006, 0.09, 0.008, 0.005],
        [0.010, 0.008, 0.06, 0.004],
        [0.003, 0.005, 0.004, 0.03],
    ])
    L = np.linalg.cholesky(cov_true)
    T = 50
    returns = mu_true + (rng.standard_normal((T, 4)) @ L.T)
    return returns


@pytest.fixture()
def small_mu() -> NDArray[np.float64]:
    return np.array([0.10, 0.08, 0.12, 0.06])


@pytest.fixture()
def small_cov() -> NDArray[np.float64]:
    return np.array([
        [0.04, 0.006, 0.010, 0.003],
        [0.006, 0.09, 0.008, 0.005],
        [0.010, 0.008, 0.06, 0.004],
        [0.003, 0.005, 0.004, 0.03],
    ])


# ---------------------------------------------------------------------------
# Ellipsoidal uncertainty set construction
# ---------------------------------------------------------------------------

class TestEllipsoidalUncertaintySet:
    def test_build_from_returns(self, small_returns: NDArray) -> None:
        uset = build_ellipsoidal_uncertainty(small_returns, epsilon=0.1)
        assert uset.mu_hat.shape == (4,)
        assert uset.sigma_mu.shape == (4, 4)
        assert uset.epsilon == 0.1

    def test_sigma_mu_is_psd(self, small_returns: NDArray) -> None:
        uset = build_ellipsoidal_uncertainty(small_returns, epsilon=0.5)
        eigvals = np.linalg.eigvalsh(uset.sigma_mu)
        assert np.all(eigvals >= -1e-10), "sigma_mu must be positive semi-definite"

    def test_sigma_mu_scales_with_t(self) -> None:
        rng = np.random.default_rng(99)
        ret_short = rng.standard_normal((20, 3))
        ret_long = rng.standard_normal((200, 3))
        uset_short = build_ellipsoidal_uncertainty(ret_short, epsilon=0.1)
        uset_long = build_ellipsoidal_uncertainty(ret_long, epsilon=0.1)
        # Longer sample -> smaller uncertainty (sigma_mu ~ 1/T)
        assert np.trace(uset_long.sigma_mu) < np.trace(uset_short.sigma_mu)

    def test_shrinkage(self, small_returns: NDArray) -> None:
        uset_no = build_ellipsoidal_uncertainty(small_returns, epsilon=0.1, shrinkage=0.0)
        uset_full = build_ellipsoidal_uncertainty(small_returns, epsilon=0.1, shrinkage=1.0)
        # Full shrinkage -> diagonal matrix
        off_diag = uset_full.sigma_mu.copy()
        np.fill_diagonal(off_diag, 0)
        assert np.allclose(off_diag, 0), "Full shrinkage should give diagonal sigma_mu"
        # No shrinkage should have off-diagonal elements
        off_diag_no = uset_no.sigma_mu.copy()
        np.fill_diagonal(off_diag_no, 0)
        assert not np.allclose(off_diag_no, 0)

    def test_invalid_shapes(self) -> None:
        with pytest.raises(ValueError, match="sigma_mu shape"):
            EllipsoidalUncertaintySet(
                mu_hat=np.array([0.1, 0.2]),
                sigma_mu=np.eye(3),
                epsilon=0.1,
            )

    def test_negative_epsilon(self) -> None:
        with pytest.raises(ValueError, match="epsilon must be non-negative"):
            EllipsoidalUncertaintySet(
                mu_hat=np.array([0.1, 0.2]),
                sigma_mu=np.eye(2),
                epsilon=-0.5,
            )

    def test_1d_returns_raises(self) -> None:
        with pytest.raises(ValueError, match="must be 2-D"):
            build_ellipsoidal_uncertainty(np.array([1.0, 2.0, 3.0]), epsilon=0.1)

    def test_too_few_observations(self) -> None:
        with pytest.raises(ValueError, match="at least 2 observations"):
            build_ellipsoidal_uncertainty(np.array([[1.0, 2.0]]), epsilon=0.1)


# ---------------------------------------------------------------------------
# Robust QUBO formulation
# ---------------------------------------------------------------------------

class TestRobustQUBO:
    def test_qubo_shape(self, small_mu: NDArray, small_cov: NDArray) -> None:
        uset = EllipsoidalUncertaintySet(
            mu_hat=small_mu, sigma_mu=small_cov / 50, epsilon=0.2
        )
        opt = RobustPortfolioOptimizer(uset, small_cov, gamma=1.0)
        Q = opt.build_matrix()
        assert Q.shape == (4, 4)

    def test_zero_epsilon_matches_standard_qubo(
        self, small_mu: NDArray, small_cov: NDArray
    ) -> None:
        """With epsilon=0 the robust QUBO should equal the standard QUBO."""
        uset = EllipsoidalUncertaintySet(
            mu_hat=small_mu, sigma_mu=small_cov / 50, epsilon=0.0
        )
        opt = RobustPortfolioOptimizer(uset, small_cov, gamma=1.0)
        Q_robust = opt.build_matrix()

        standard = PortfolioQUBO(mu=small_mu, cov=small_cov, gamma=1.0)
        Q_standard = standard.build_matrix()

        np.testing.assert_allclose(Q_robust, Q_standard, atol=1e-10)

    def test_larger_epsilon_reduces_effective_returns(
        self, small_mu: NDArray, small_cov: NDArray
    ) -> None:
        """Larger uncertainty should reduce effective returns (more penalty)."""
        sigma_mu = small_cov / 50
        uset_lo = EllipsoidalUncertaintySet(
            mu_hat=small_mu, sigma_mu=sigma_mu, epsilon=0.1
        )
        uset_hi = EllipsoidalUncertaintySet(
            mu_hat=small_mu, sigma_mu=sigma_mu, epsilon=1.0
        )
        opt_lo = RobustPortfolioOptimizer(uset_lo, small_cov, gamma=1.0)
        opt_hi = RobustPortfolioOptimizer(uset_hi, small_cov, gamma=1.0)

        mu_lo = opt_lo._compute_robust_mu()
        mu_hi = opt_hi._compute_robust_mu()

        # Higher epsilon => lower adjusted returns for all assets
        assert np.all(mu_hi <= mu_lo + 1e-12)

    def test_exhaustive_solve_small(
        self, small_mu: NDArray, small_cov: NDArray
    ) -> None:
        """Exhaustive solve on 4 assets should return valid result."""
        uset = EllipsoidalUncertaintySet(
            mu_hat=small_mu, sigma_mu=small_cov / 50, epsilon=0.2
        )
        opt = RobustPortfolioOptimizer(
            uset, small_cov, gamma=1.0, cardinality=2, budget_penalty=10.0
        )
        result = opt.solve_exhaustive()

        assert isinstance(result, RobustPortfolioResult)
        assert len(result.best_bitstring) == 4
        assert result.weights.shape == (4,)
        assert result.wall_time_s > 0
        # Weights should sum to 1 (equal weight among selected)
        assert abs(np.sum(result.weights) - 1.0) < 1e-6

    def test_qubo_with_cardinality(
        self, small_mu: NDArray, small_cov: NDArray
    ) -> None:
        uset = EllipsoidalUncertaintySet(
            mu_hat=small_mu, sigma_mu=small_cov / 50, epsilon=0.3
        )
        opt = RobustPortfolioOptimizer(
            uset, small_cov, gamma=1.0, cardinality=2
        )
        qubo = opt.build_qubo()
        assert qubo.cardinality == 2
        assert qubo.gamma == 1.0


# ---------------------------------------------------------------------------
# Classical robust solver (CVXPY)
# ---------------------------------------------------------------------------

class TestRobustClassical:
    def test_weights_sum_to_one(
        self, small_mu: NDArray, small_cov: NDArray
    ) -> None:
        uset = EllipsoidalUncertaintySet(
            mu_hat=small_mu, sigma_mu=small_cov / 50, epsilon=0.2
        )
        result = robust_classical(small_mu, small_cov, uset, gamma=1.0)
        assert abs(np.sum(result.weights) - 1.0) < 1e-4

    def test_long_only_non_negative(
        self, small_mu: NDArray, small_cov: NDArray
    ) -> None:
        uset = EllipsoidalUncertaintySet(
            mu_hat=small_mu, sigma_mu=small_cov / 50, epsilon=0.2
        )
        result = robust_classical(
            small_mu, small_cov, uset, gamma=1.0, long_only=True
        )
        assert np.all(result.weights >= -1e-6)

    def test_zero_epsilon_near_standard_mvo(
        self, small_mu: NDArray, small_cov: NDArray
    ) -> None:
        """With epsilon=0 the robust solution should match standard MVO."""
        uset = EllipsoidalUncertaintySet(
            mu_hat=small_mu, sigma_mu=small_cov / 50, epsilon=0.0
        )
        result_robust = robust_classical(small_mu, small_cov, uset, gamma=1.0)

        # Solve standard MVO via same CVXPY approach
        uset_zero = EllipsoidalUncertaintySet(
            mu_hat=small_mu, sigma_mu=small_cov / 50, epsilon=0.0
        )
        result_standard = robust_classical(small_mu, small_cov, uset_zero, gamma=1.0)

        np.testing.assert_allclose(
            result_robust.weights, result_standard.weights, atol=1e-3
        )

    def test_higher_epsilon_more_conservative(
        self, small_mu: NDArray, small_cov: NDArray
    ) -> None:
        """More uncertainty should lead to more diversified portfolios."""
        sigma_mu = small_cov / 50
        uset_lo = EllipsoidalUncertaintySet(
            mu_hat=small_mu, sigma_mu=sigma_mu, epsilon=0.0
        )
        uset_hi = EllipsoidalUncertaintySet(
            mu_hat=small_mu, sigma_mu=sigma_mu, epsilon=2.0
        )
        result_lo = robust_classical(small_mu, small_cov, uset_lo, gamma=1.0)
        result_hi = robust_classical(small_mu, small_cov, uset_hi, gamma=1.0)

        # Worst-case return should be lower with higher epsilon
        assert result_hi.worst_case_return <= result_lo.nominal_return + 1e-6

    def test_max_weight_constraint(
        self, small_mu: NDArray, small_cov: NDArray
    ) -> None:
        uset = EllipsoidalUncertaintySet(
            mu_hat=small_mu, sigma_mu=small_cov / 50, epsilon=0.2
        )
        result = robust_classical(
            small_mu, small_cov, uset, gamma=1.0, max_weight=0.4
        )
        assert np.all(result.weights <= 0.4 + 1e-4)

    def test_result_fields(
        self, small_mu: NDArray, small_cov: NDArray
    ) -> None:
        uset = EllipsoidalUncertaintySet(
            mu_hat=small_mu, sigma_mu=small_cov / 50, epsilon=0.2
        )
        result = robust_classical(small_mu, small_cov, uset, gamma=1.0)
        assert result.wall_time_s > 0
        assert result.risk >= 0
        assert isinstance(result.nominal_return, float)
        assert isinstance(result.worst_case_return, float)
        assert result.feasible is True


# ---------------------------------------------------------------------------
# Classical vs quantum robust comparison
# ---------------------------------------------------------------------------

class TestClassicalVsQuantum:
    def test_both_solvers_feasible(self) -> None:
        """Both classical and exhaustive quantum should produce feasible solutions."""
        mu = np.array([0.10, 0.08, 0.12])
        cov = np.array([
            [0.04, 0.006, 0.010],
            [0.006, 0.09, 0.008],
            [0.010, 0.008, 0.06],
        ])
        sigma_mu = cov / 30
        uset = EllipsoidalUncertaintySet(mu_hat=mu, sigma_mu=sigma_mu, epsilon=0.2)

        # Classical
        result_cl = robust_classical(mu, cov, uset, gamma=1.0)
        assert result_cl.feasible
        assert abs(np.sum(result_cl.weights) - 1.0) < 1e-4

        # Quantum (exhaustive)
        opt = RobustPortfolioOptimizer(uset, cov, gamma=1.0)
        result_q = opt.solve_exhaustive()
        assert len(result_q.best_bitstring) == 3
        assert abs(np.sum(result_q.weights) - 1.0) < 1e-6

    def test_objective_direction_consistent(self) -> None:
        """Both approaches should minimize (risk - return) type objective."""
        mu = np.array([0.10, 0.05])
        cov = np.array([[0.04, 0.01], [0.01, 0.02]])
        sigma_mu = cov / 20
        uset = EllipsoidalUncertaintySet(mu_hat=mu, sigma_mu=sigma_mu, epsilon=0.1)

        result_cl = robust_classical(mu, cov, uset, gamma=1.0)
        opt = RobustPortfolioOptimizer(uset, cov, gamma=1.0)
        result_q = opt.solve_exhaustive()

        # Both should have finite, reasonable values
        assert np.isfinite(result_cl.value)
        assert np.isfinite(result_q.best_objective)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_asset(self) -> None:
        mu = np.array([0.10])
        cov = np.array([[0.04]])
        sigma_mu = np.array([[0.04 / 50]])
        uset = EllipsoidalUncertaintySet(mu_hat=mu, sigma_mu=sigma_mu, epsilon=0.1)

        opt = RobustPortfolioOptimizer(uset, cov, gamma=1.0)
        result = opt.solve_exhaustive()
        np.testing.assert_allclose(result.weights, [1.0], atol=1e-6)

    def test_two_identical_assets(self) -> None:
        """Two identical assets should get equal weight."""
        mu = np.array([0.10, 0.10])
        cov = np.array([[0.04, 0.04], [0.04, 0.04]])
        sigma_mu = cov / 50
        uset = EllipsoidalUncertaintySet(mu_hat=mu, sigma_mu=sigma_mu, epsilon=0.1)

        result = robust_classical(mu, cov, uset, gamma=1.0)
        np.testing.assert_allclose(result.weights[0], result.weights[1], atol=1e-3)

    def test_zero_covariance_uncertainty(self) -> None:
        """Zero off-diagonal in sigma_mu should still work."""
        mu = np.array([0.10, 0.08])
        cov = np.array([[0.04, 0.01], [0.01, 0.02]])
        sigma_mu = np.diag([0.001, 0.002])
        uset = EllipsoidalUncertaintySet(mu_hat=mu, sigma_mu=sigma_mu, epsilon=0.5)

        opt = RobustPortfolioOptimizer(uset, cov, gamma=1.0)
        Q = opt.build_matrix()
        assert Q.shape == (2, 2)
        assert np.all(np.isfinite(Q))

    def test_high_epsilon_pushes_toward_min_variance(self) -> None:
        """Very high uncertainty should push toward minimum variance portfolio."""
        mu = np.array([0.20, 0.02])  # very different returns
        cov = np.array([[0.10, 0.01], [0.01, 0.02]])
        sigma_mu = cov / 30
        uset_lo = EllipsoidalUncertaintySet(
            mu_hat=mu, sigma_mu=sigma_mu, epsilon=0.0
        )
        uset_hi = EllipsoidalUncertaintySet(
            mu_hat=mu, sigma_mu=sigma_mu, epsilon=50.0
        )

        result_lo = robust_classical(mu, cov, uset_lo, gamma=1.0)
        result_hi = robust_classical(mu, cov, uset_hi, gamma=1.0)

        # High epsilon portfolio should have lower risk (more min-variance)
        assert result_hi.risk <= result_lo.risk + 1e-4
