"""Unit tests for GARCH volatility forecasting."""

from __future__ import annotations

import numpy as np
import pytest

from qufin.risk.garch import (
    GARCHResult,
    fit_garch,
    forecast_evaluate,
    rolling_garch_forecast,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def clustered_returns(rng: np.random.Generator) -> np.ndarray:
    """Synthetic returns with volatility clustering (GARCH-like).

    Generates 500 observations with time-varying variance.
    """
    n = 500
    returns = np.zeros(n)
    sigma2 = np.zeros(n)
    omega, alpha, beta = 0.00001, 0.1, 0.85
    sigma2[0] = omega / (1 - alpha - beta)

    for t in range(1, n):
        sigma2[t] = omega + alpha * returns[t - 1] ** 2 + beta * sigma2[t - 1]
        returns[t] = rng.normal(0, np.sqrt(sigma2[t]))

    return returns


# ---------------------------------------------------------------------------
# Tests: fit_garch
# ---------------------------------------------------------------------------

class TestFitGarch:
    def test_garch_returns_result(self, clustered_returns: np.ndarray) -> None:
        result = fit_garch(clustered_returns, model="garch")
        assert isinstance(result, GARCHResult)
        assert result.model_type == "garch"

    def test_egarch(self, clustered_returns: np.ndarray) -> None:
        result = fit_garch(clustered_returns, model="egarch")
        assert isinstance(result, GARCHResult)
        assert result.model_type == "egarch"

    def test_gjr(self, clustered_returns: np.ndarray) -> None:
        result = fit_garch(clustered_returns, model="gjr")
        assert isinstance(result, GARCHResult)
        assert result.model_type == "gjr"

    def test_forecast_length(self, clustered_returns: np.ndarray) -> None:
        horizon = 5
        result = fit_garch(clustered_returns, model="garch", horizon=horizon)
        assert len(result.forecast) == horizon

    def test_conditional_vol_length(self, clustered_returns: np.ndarray) -> None:
        result = fit_garch(clustered_returns, model="garch")
        assert len(result.conditional_vol) == len(clustered_returns)

    def test_aic_bic_finite(self, clustered_returns: np.ndarray) -> None:
        result = fit_garch(clustered_returns, model="garch")
        assert np.isfinite(result.aic)
        assert np.isfinite(result.bic)

    def test_loglikelihood_finite(self, clustered_returns: np.ndarray) -> None:
        result = fit_garch(clustered_returns, model="garch")
        assert np.isfinite(result.loglikelihood)

    def test_params_dict_nonempty(self, clustered_returns: np.ndarray) -> None:
        result = fit_garch(clustered_returns, model="garch")
        assert isinstance(result.params, dict)
        assert len(result.params) > 0

    def test_unknown_model_raises(self, clustered_returns: np.ndarray) -> None:
        with pytest.raises(ValueError, match="Unknown model"):
            fit_garch(clustered_returns, model="bogus")

    def test_forecast_positive(self, clustered_returns: np.ndarray) -> None:
        result = fit_garch(clustered_returns, model="garch", horizon=3)
        assert np.all(result.forecast > 0)

    def test_conditional_vol_positive(self, clustered_returns: np.ndarray) -> None:
        result = fit_garch(clustered_returns, model="garch")
        assert np.all(result.conditional_vol >= 0)


# ---------------------------------------------------------------------------
# Tests: forecast_evaluate
# ---------------------------------------------------------------------------

class TestForecastEvaluate:
    def test_returns_all_keys(self) -> None:
        rng = np.random.default_rng(42)
        actual = np.abs(rng.normal(0.01, 0.005, 100))
        predicted = actual + rng.normal(0, 0.001, 100)
        predicted = np.abs(predicted)

        result = forecast_evaluate(actual, predicted)
        assert set(result.keys()) == {"mae", "rmse", "qlike", "mz_r2"}

    def test_values_finite(self) -> None:
        rng = np.random.default_rng(42)
        actual = np.abs(rng.normal(0.01, 0.005, 100))
        predicted = np.abs(rng.normal(0.01, 0.005, 100))

        result = forecast_evaluate(actual, predicted)
        for key, val in result.items():
            assert np.isfinite(val), f"{key} is not finite: {val}"

    def test_perfect_forecast(self) -> None:
        actual = np.abs(np.random.default_rng(42).normal(0.01, 0.005, 50))
        result = forecast_evaluate(actual, actual)
        np.testing.assert_allclose(result["mae"], 0.0, atol=1e-12)
        np.testing.assert_allclose(result["rmse"], 0.0, atol=1e-12)

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="Length mismatch"):
            forecast_evaluate(np.ones(10), np.ones(5))


# ---------------------------------------------------------------------------
# Tests: rolling_garch_forecast
# ---------------------------------------------------------------------------

class TestRollingGarchForecast:
    @pytest.mark.slow
    def test_returns_correct_length(self, clustered_returns: np.ndarray) -> None:
        window = 200
        step = 50
        n = len(clustered_returns)
        expected_len = len(range(0, n - window, step))

        forecasts = rolling_garch_forecast(
            clustered_returns, model="garch", window=window, step=step
        )
        assert len(forecasts) == expected_len

    @pytest.mark.slow
    def test_forecasts_positive(self, clustered_returns: np.ndarray) -> None:
        forecasts = rolling_garch_forecast(
            clustered_returns, model="garch", window=200, step=100
        )
        assert np.all(forecasts > 0)

    def test_window_too_large_raises(self, clustered_returns: np.ndarray) -> None:
        with pytest.raises(ValueError, match="Window"):
            rolling_garch_forecast(
                clustered_returns, model="garch", window=len(clustered_returns) + 1
            )
