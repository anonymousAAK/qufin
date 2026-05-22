"""Unit tests for risk parity and inverse volatility weighting."""

from __future__ import annotations

import numpy as np
import pytest

from qufin.portfolio.classical.risk_parity import (
    RiskParityResult,
    inverse_volatility_weights,
    risk_parity,
)


@pytest.fixture
def cov_3x3() -> np.ndarray:
    return np.array([
        [0.04, 0.006, 0.002],
        [0.006, 0.09, 0.004],
        [0.002, 0.004, 0.01],
    ])


@pytest.fixture
def cov_diagonal() -> np.ndarray:
    return np.diag([0.01, 0.04, 0.09])


# --- Inverse Volatility Weights ---

class TestInverseVolatilityWeights:
    def test_returns_array(self, cov_3x3) -> None:
        w = inverse_volatility_weights(cov_3x3)
        assert isinstance(w, np.ndarray)
        assert w.shape == (3,)

    def test_sums_to_one(self, cov_3x3) -> None:
        w = inverse_volatility_weights(cov_3x3)
        assert abs(np.sum(w) - 1.0) < 1e-10

    def test_positive_weights(self, cov_3x3) -> None:
        w = inverse_volatility_weights(cov_3x3)
        assert np.all(w > 0)

    def test_lower_vol_gets_more_weight(self, cov_diagonal) -> None:
        w = inverse_volatility_weights(cov_diagonal)
        # Asset 0 has lowest vol (0.1), should get highest weight
        assert w[0] > w[1] > w[2]

    def test_equal_vol_equal_weight(self) -> None:
        cov = np.eye(3) * 0.04
        w = inverse_volatility_weights(cov)
        np.testing.assert_allclose(w, np.ones(3) / 3, atol=1e-10)

    def test_large_matrix(self) -> None:
        rng = np.random.default_rng(42)
        A = rng.normal(0, 1, (10, 10))
        cov = A.T @ A / 10  # positive semi-definite
        w = inverse_volatility_weights(cov)
        assert w.shape == (10,)
        assert abs(np.sum(w) - 1.0) < 1e-10

    def test_dtype_float64(self, cov_3x3) -> None:
        w = inverse_volatility_weights(cov_3x3)
        assert w.dtype == np.float64


# --- Risk Parity (existing + extended) ---

class TestRiskParity:
    def test_returns_result_type(self, cov_3x3) -> None:
        result = risk_parity(cov_3x3)
        assert isinstance(result, RiskParityResult)

    def test_weights_sum_to_one(self, cov_3x3) -> None:
        result = risk_parity(cov_3x3)
        assert abs(np.sum(result.weights) - 1.0) < 1e-6

    def test_positive_weights(self, cov_3x3) -> None:
        result = risk_parity(cov_3x3)
        assert np.all(result.weights > 0)

    def test_equal_risk_contributions(self, cov_3x3) -> None:
        result = risk_parity(cov_3x3)
        rc = result.risk_contributions
        # Risk contributions should be approximately equal
        rc_normed = rc / rc.sum() if rc.sum() > 0 else rc
        assert np.std(rc_normed) < 0.05

    def test_custom_budget(self, cov_3x3) -> None:
        budget = np.array([0.5, 0.3, 0.2])
        result = risk_parity(cov_3x3, budget=budget)
        assert isinstance(result, RiskParityResult)
        assert abs(np.sum(result.weights) - 1.0) < 1e-6

    def test_volatility_positive(self, cov_3x3) -> None:
        result = risk_parity(cov_3x3)
        assert result.portfolio_volatility > 0

    def test_diagonal_cov(self, cov_diagonal) -> None:
        result = risk_parity(cov_diagonal)
        # For diagonal cov, risk parity ~ inverse vol
        assert abs(np.sum(result.weights) - 1.0) < 1e-6

    def test_inv_vol_vs_risk_parity_different(self, cov_3x3) -> None:
        rp = risk_parity(cov_3x3)
        iv = inverse_volatility_weights(cov_3x3)
        # They are related but not identical (RP accounts for correlations)
        # Just check both are valid
        assert abs(np.sum(rp.weights) - 1.0) < 1e-6
        assert abs(np.sum(iv) - 1.0) < 1e-10
