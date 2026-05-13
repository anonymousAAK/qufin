"""Unit tests for Cornish-Fisher VaR and Kelly criterion utilities."""

from __future__ import annotations

import numpy as np
import pytest

from qufin.risk.cornish_fisher import (
    cornish_fisher_es,
    cornish_fisher_var,
    fractional_kelly,
    kelly_criterion,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def normal_returns(rng: np.random.Generator) -> np.ndarray:
    """Large sample of perfectly normal returns."""
    return rng.normal(0.001, 0.02, 10_000)


@pytest.fixture
def skewed_returns(rng: np.random.Generator) -> np.ndarray:
    """Returns with negative skew and fat tails."""
    base = rng.normal(0.001, 0.02, 5000)
    # Add occasional large negative jumps
    jumps = rng.choice([-0.08, -0.10, -0.12], size=50)
    return np.concatenate([base, jumps])


@pytest.fixture
def positive_sharpe_returns(rng: np.random.Generator) -> np.ndarray:
    """Returns with clearly positive mean."""
    return rng.normal(0.005, 0.02, 1000)


# ---------------------------------------------------------------------------
# Tests: cornish_fisher_var
# ---------------------------------------------------------------------------

class TestCornishFisherVar:
    def test_returns_all_keys(self, normal_returns: np.ndarray) -> None:
        result = cornish_fisher_var(normal_returns, confidence=0.99)
        expected_keys = {"var", "var_gaussian", "skewness", "excess_kurtosis", "z_cf"}
        assert set(result.keys()) == expected_keys

    def test_cf_differs_from_gaussian_for_skewed(
        self, skewed_returns: np.ndarray
    ) -> None:
        result = cornish_fisher_var(skewed_returns, confidence=0.99)
        assert result["var"] != pytest.approx(result["var_gaussian"], abs=1e-6)

    def test_cf_approx_gaussian_for_normal(
        self, normal_returns: np.ndarray
    ) -> None:
        result = cornish_fisher_var(normal_returns, confidence=0.99)
        # For truly normal data, CF and Gaussian VaR should be close
        np.testing.assert_allclose(
            result["var"], result["var_gaussian"], rtol=0.1
        )

    def test_higher_confidence_higher_var(
        self, normal_returns: np.ndarray
    ) -> None:
        var_95 = cornish_fisher_var(normal_returns, confidence=0.95)["var"]
        var_99 = cornish_fisher_var(normal_returns, confidence=0.99)["var"]
        assert var_99 > var_95

    def test_var_positive(self, skewed_returns: np.ndarray) -> None:
        result = cornish_fisher_var(skewed_returns, confidence=0.99)
        # VaR should be positive for returns centered near zero
        assert result["var"] > 0


# ---------------------------------------------------------------------------
# Tests: cornish_fisher_es
# ---------------------------------------------------------------------------

class TestCornishFisherES:
    def test_es_gte_var(self, skewed_returns: np.ndarray) -> None:
        var_cf = cornish_fisher_var(skewed_returns, confidence=0.99)["var"]
        es = cornish_fisher_es(skewed_returns, confidence=0.99)
        assert es >= var_cf - 1e-10  # small tolerance for numerical noise

    def test_es_positive(self, normal_returns: np.ndarray) -> None:
        es = cornish_fisher_es(normal_returns, confidence=0.99)
        assert es > 0

    def test_es_larger_at_higher_confidence(
        self, skewed_returns: np.ndarray
    ) -> None:
        es_95 = cornish_fisher_es(skewed_returns, confidence=0.95)
        es_99 = cornish_fisher_es(skewed_returns, confidence=0.99)
        assert es_99 >= es_95


# ---------------------------------------------------------------------------
# Tests: kelly_criterion
# ---------------------------------------------------------------------------

class TestKellyCriterion:
    def test_positive_kelly_for_positive_sharpe(
        self, positive_sharpe_returns: np.ndarray
    ) -> None:
        result = kelly_criterion(positive_sharpe_returns, risk_free_rate=0.0)
        assert result["full_kelly"] > 0
        assert result["sharpe"] > 0

    def test_half_kelly_is_half(
        self, positive_sharpe_returns: np.ndarray
    ) -> None:
        result = kelly_criterion(positive_sharpe_returns)
        np.testing.assert_allclose(
            result["half_kelly"], result["full_kelly"] / 2.0, rtol=1e-12
        )

    def test_all_keys_present(self, normal_returns: np.ndarray) -> None:
        result = kelly_criterion(normal_returns)
        expected = {"full_kelly", "half_kelly", "expected_return", "volatility", "sharpe"}
        assert set(result.keys()) == expected

    def test_volatility_positive(self, normal_returns: np.ndarray) -> None:
        result = kelly_criterion(normal_returns)
        assert result["volatility"] > 0


# ---------------------------------------------------------------------------
# Tests: fractional_kelly
# ---------------------------------------------------------------------------

class TestFractionalKelly:
    def test_fraction_one_equals_full_kelly(
        self, positive_sharpe_returns: np.ndarray
    ) -> None:
        full = kelly_criterion(positive_sharpe_returns)
        frac = fractional_kelly(
            positive_sharpe_returns,
            fraction=1.0,
            estimation_error_adjustment=False,
        )
        np.testing.assert_allclose(
            frac["kelly_fraction"], full["full_kelly"], rtol=1e-12
        )

    def test_estimation_error_reduces_kelly(
        self, positive_sharpe_returns: np.ndarray
    ) -> None:
        without = fractional_kelly(
            positive_sharpe_returns,
            fraction=0.5,
            estimation_error_adjustment=False,
        )
        with_adj = fractional_kelly(
            positive_sharpe_returns,
            fraction=0.5,
            estimation_error_adjustment=True,
        )
        assert abs(with_adj["adjusted_kelly"]) < abs(without["kelly_fraction"])

    def test_confidence_band_order(
        self, positive_sharpe_returns: np.ndarray
    ) -> None:
        result = fractional_kelly(positive_sharpe_returns, fraction=0.5)
        lower, upper = result["confidence_band"]
        assert lower <= upper

    def test_all_keys_present(
        self, positive_sharpe_returns: np.ndarray
    ) -> None:
        result = fractional_kelly(positive_sharpe_returns)
        expected = {"fraction", "kelly_fraction", "adjusted_kelly", "confidence_band"}
        assert set(result.keys()) == expected
