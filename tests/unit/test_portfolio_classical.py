"""Tests for classical portfolio optimization baselines."""

from __future__ import annotations

import numpy as np

from qufin.portfolio.classical.black_litterman import black_litterman
from qufin.portfolio.classical.hrp import hrp
from qufin.portfolio.classical.mean_variance import Objective, mean_variance
from qufin.portfolio.classical.risk_parity import risk_parity


class TestMeanVariance:
    def test_weights_sum_to_one(self, sample_mu: np.ndarray, sample_cov: np.ndarray) -> None:
        result = mean_variance(sample_mu, sample_cov, objective=Objective.MIN_VARIANCE)
        assert abs(np.sum(result.weights) - 1.0) < 1e-6

    def test_long_only_non_negative(self, sample_mu: np.ndarray, sample_cov: np.ndarray) -> None:
        result = mean_variance(sample_mu, sample_cov, long_only=True)
        assert np.all(result.weights >= -1e-8)

    def test_min_variance_has_lower_vol(
        self, sample_mu: np.ndarray, sample_cov: np.ndarray
    ) -> None:
        mv = mean_variance(sample_mu, sample_cov, objective=Objective.MIN_VARIANCE)
        # Equal weight portfolio
        w_eq = np.ones(len(sample_mu)) / len(sample_mu)
        vol_eq = float(np.sqrt(w_eq @ sample_cov @ w_eq))
        assert mv.volatility <= vol_eq + 1e-6

    def test_sharpe_ratio_computed(
        self, sample_mu: np.ndarray, sample_cov: np.ndarray
    ) -> None:
        result = mean_variance(sample_mu, sample_cov)
        assert isinstance(result.sharpe_ratio, float)

    def test_max_weight_constraint(
        self, sample_mu: np.ndarray, sample_cov: np.ndarray
    ) -> None:
        result = mean_variance(sample_mu, sample_cov, max_weight=0.3)
        assert np.all(result.weights <= 0.3 + 1e-6)


class TestBlackLitterman:
    def test_no_views_returns_equilibrium(self) -> None:
        cov = np.array([[0.04, 0.006], [0.006, 0.09]])
        caps = np.array([1e9, 2e9])
        result = black_litterman(cov, caps)
        np.testing.assert_array_almost_equal(result.posterior_mu, result.equilibrium_mu)

    def test_with_views_shifts_returns(self) -> None:
        cov = np.array([[0.04, 0.006], [0.006, 0.09]])
        caps = np.array([1e9, 2e9])
        # View: asset 0 will outperform by 5%
        P = np.array([[1, -1]])
        Q = np.array([0.05])
        result_no_view = black_litterman(cov, caps)
        result_view = black_litterman(cov, caps, P=P, Q=Q)
        # Asset 0's posterior return should be higher with the bullish view
        assert result_view.posterior_mu[0] > result_no_view.posterior_mu[0]

    def test_posterior_cov_shape(self) -> None:
        n = 3
        rng = np.random.default_rng(42)
        A = rng.normal(size=(100, n))
        cov = np.cov(A, rowvar=False)
        caps = np.array([1e9, 2e9, 3e9])
        result = black_litterman(cov, caps)
        assert result.posterior_cov.shape == (n, n)


class TestRiskParity:
    def test_weights_sum_to_one(self, sample_cov: np.ndarray) -> None:
        result = risk_parity(sample_cov)
        assert abs(np.sum(result.weights) - 1.0) < 1e-6

    def test_equal_risk_contribution(self, sample_cov: np.ndarray) -> None:
        result = risk_parity(sample_cov)
        # Risk contributions should be approximately equal
        rc = result.risk_contributions
        mean_rc = np.mean(rc)
        assert np.all(np.abs(rc - mean_rc) < mean_rc * 0.15)  # within 15%

    def test_positive_weights(self, sample_cov: np.ndarray) -> None:
        result = risk_parity(sample_cov)
        assert np.all(result.weights > 0)


class TestHRP:
    def test_weights_sum_to_one(self, sample_returns: np.ndarray) -> None:
        result = hrp(sample_returns)
        assert abs(np.sum(result.weights) - 1.0) < 1e-6

    def test_positive_weights(self, sample_returns: np.ndarray) -> None:
        result = hrp(sample_returns)
        assert np.all(result.weights > 0)

    def test_cluster_order_length(self, sample_returns: np.ndarray) -> None:
        result = hrp(sample_returns)
        assert len(result.cluster_order) == sample_returns.shape[1]

    def test_different_linkage_methods(self, sample_returns: np.ndarray) -> None:
        for method in ["single", "complete", "average"]:
            result = hrp(sample_returns, linkage_method=method)
            assert abs(np.sum(result.weights) - 1.0) < 1e-6
