"""Unit tests for classical VaR and Expected Shortfall."""

from __future__ import annotations

import numpy as np
import pytest

from qufin.risk.classical_var import (
    VaRResult,
    historical_var,
    monte_carlo_var,
    parametric_var,
    portfolio_var,
)


@pytest.fixture
def normal_returns() -> np.ndarray:
    """Simulated normal returns for testing."""
    rng = np.random.default_rng(42)
    return rng.normal(0.0005, 0.02, 1000)


@pytest.fixture
def multi_asset_returns() -> np.ndarray:
    """Multi-asset returns, shape (500, 3)."""
    rng = np.random.default_rng(42)
    return rng.multivariate_normal(
        [0.0005, 0.0003, 0.0008],
        [[0.0004, 0.0001, 0.0002],
         [0.0001, 0.0003, 0.00015],
         [0.0002, 0.00015, 0.0005]],
        500,
    )


class TestHistoricalVaR:
    def test_positive_var(self, normal_returns: np.ndarray) -> None:
        result = historical_var(normal_returns, confidence=0.95)
        assert result.var > 0
        assert result.expected_shortfall >= result.var

    def test_higher_confidence_higher_var(self, normal_returns: np.ndarray) -> None:
        r95 = historical_var(normal_returns, confidence=0.95)
        r99 = historical_var(normal_returns, confidence=0.99)
        assert r99.var > r95.var

    def test_es_exceeds_var(self, normal_returns: np.ndarray) -> None:
        result = historical_var(normal_returns, confidence=0.95)
        assert result.expected_shortfall >= result.var

    def test_method_label(self, normal_returns: np.ndarray) -> None:
        result = historical_var(normal_returns)
        assert result.method == "historical"

    def test_dollar_var(self, normal_returns: np.ndarray) -> None:
        result = historical_var(normal_returns, portfolio_value=1_000_000)
        assert result.var_dollar == result.var * 1_000_000
        assert result.es_dollar == result.expected_shortfall * 1_000_000


class TestParametricVaR:
    def test_positive_var(self, normal_returns: np.ndarray) -> None:
        result = parametric_var(normal_returns, confidence=0.95)
        assert result.var > 0

    def test_es_exceeds_var(self, normal_returns: np.ndarray) -> None:
        result = parametric_var(normal_returns, confidence=0.95)
        assert result.expected_shortfall >= result.var

    def test_close_to_historical(self, normal_returns: np.ndarray) -> None:
        """For normal data, parametric should be close to historical."""
        hist = historical_var(normal_returns, confidence=0.95)
        param = parametric_var(normal_returns, confidence=0.95)
        assert abs(hist.var - param.var) / param.var < 0.3


class TestMonteCarloVaR:
    def test_positive_var(self, normal_returns: np.ndarray) -> None:
        result = monte_carlo_var(normal_returns, confidence=0.95, seed=42)
        assert result.var > 0

    def test_deterministic(self, normal_returns: np.ndarray) -> None:
        r1 = monte_carlo_var(normal_returns, seed=42, n_simulations=10_000)
        r2 = monte_carlo_var(normal_returns, seed=42, n_simulations=10_000)
        assert r1.var == r2.var

    def test_multi_asset(self, multi_asset_returns: np.ndarray) -> None:
        result = monte_carlo_var(multi_asset_returns, confidence=0.95, seed=42)
        assert result.var > 0


class TestPortfolioVaR:
    def test_historical_method(self, multi_asset_returns: np.ndarray) -> None:
        weights = np.array([0.4, 0.3, 0.3])
        result = portfolio_var(multi_asset_returns, weights, method="historical")
        assert result.var > 0
        assert result.method == "historical"

    def test_parametric_method(self, multi_asset_returns: np.ndarray) -> None:
        weights = np.array([0.4, 0.3, 0.3])
        result = portfolio_var(multi_asset_returns, weights, method="parametric")
        assert result.var > 0

    def test_invalid_method(self, multi_asset_returns: np.ndarray) -> None:
        weights = np.array([0.4, 0.3, 0.3])
        with pytest.raises(ValueError, match="Unknown method"):
            portfolio_var(multi_asset_returns, weights, method="invalid")
