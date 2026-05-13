"""Unit tests for Fama-French factor model integration."""

from __future__ import annotations

import numpy as np
import pytest

from qufin.portfolio.classical.factor_models import (
    FactorExposureResult,
    FactorModelResult,
    build_factor_model,
    estimate_factor_exposures,
    factor_expected_returns,
    factor_model_cov,
    risk_decomposition,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def synthetic_factor_data(rng: np.random.Generator):
    """Generate synthetic asset returns driven by known factor exposures.

    3 factors, 5 assets, 500 observations.
    """
    n_obs, n_assets, n_factors = 500, 5, 3

    # True betas: each asset has different exposure to each factor
    true_betas = rng.standard_normal((n_assets, n_factors)) * 0.5
    true_alpha = rng.normal(0.0005, 0.0002, n_assets)

    factor_returns = rng.normal(0.001, 0.01, (n_obs, n_factors))
    noise = rng.normal(0.0, 0.005, (n_obs, n_assets))

    # r = alpha + B @ f + eps
    asset_returns = true_alpha + factor_returns @ true_betas.T + noise

    return {
        "returns": asset_returns,
        "factor_returns": factor_returns,
        "true_betas": true_betas,
        "true_alpha": true_alpha,
        "n_obs": n_obs,
        "n_assets": n_assets,
        "n_factors": n_factors,
    }


# ---------------------------------------------------------------------------
# Tests: estimate_factor_exposures
# ---------------------------------------------------------------------------

class TestEstimateFactorExposures:
    def test_shapes(self, synthetic_factor_data: dict) -> None:
        d = synthetic_factor_data
        result = estimate_factor_exposures(d["returns"], d["factor_returns"])

        assert result.betas.shape == (d["n_assets"], d["n_factors"])
        assert result.alpha.shape == (d["n_assets"],)
        assert result.r_squared.shape == (d["n_assets"],)
        assert result.residual_cov.shape == (d["n_assets"], d["n_assets"])

    def test_r_squared_in_range(self, synthetic_factor_data: dict) -> None:
        d = synthetic_factor_data
        result = estimate_factor_exposures(d["returns"], d["factor_returns"])
        assert np.all(result.r_squared >= 0.0)
        assert np.all(result.r_squared <= 1.0)

    def test_residual_cov_is_diagonal(self, synthetic_factor_data: dict) -> None:
        d = synthetic_factor_data
        result = estimate_factor_exposures(d["returns"], d["factor_returns"])
        off_diag = result.residual_cov - np.diag(np.diag(result.residual_cov))
        np.testing.assert_allclose(off_diag, 0.0, atol=1e-15)

    def test_window_uses_subset(self, synthetic_factor_data: dict) -> None:
        d = synthetic_factor_data
        window = 100
        result = estimate_factor_exposures(
            d["returns"], d["factor_returns"], window=window
        )
        # Should still produce valid shapes
        assert result.betas.shape == (d["n_assets"], d["n_factors"])
        # Factor names defaulted
        assert len(result.factor_names) == d["n_factors"]

    def test_custom_factor_names(self, synthetic_factor_data: dict) -> None:
        d = synthetic_factor_data
        names = ["MKT", "SMB", "HML"]
        result = estimate_factor_exposures(
            d["returns"], d["factor_returns"], factor_names=names
        )
        assert result.factor_names == names

    def test_single_factor(self, rng: np.random.Generator) -> None:
        n_obs, n_assets = 200, 4
        factor_returns = rng.normal(0.001, 0.01, (n_obs, 1))
        true_beta = rng.standard_normal((n_assets, 1))
        asset_returns = factor_returns @ true_beta.T + rng.normal(0, 0.005, (n_obs, n_assets))

        result = estimate_factor_exposures(asset_returns, factor_returns)
        assert result.betas.shape == (n_assets, 1)
        assert result.alpha.shape == (n_assets,)


# ---------------------------------------------------------------------------
# Tests: factor_model_cov
# ---------------------------------------------------------------------------

class TestFactorModelCov:
    def test_correct_shape(self, synthetic_factor_data: dict) -> None:
        d = synthetic_factor_data
        exposures = estimate_factor_exposures(d["returns"], d["factor_returns"])
        f_cov = np.cov(d["factor_returns"], rowvar=False)
        cov = factor_model_cov(exposures, f_cov)
        assert cov.shape == (d["n_assets"], d["n_assets"])

    def test_symmetric(self, synthetic_factor_data: dict) -> None:
        d = synthetic_factor_data
        exposures = estimate_factor_exposures(d["returns"], d["factor_returns"])
        f_cov = np.cov(d["factor_returns"], rowvar=False)
        cov = factor_model_cov(exposures, f_cov)
        np.testing.assert_allclose(cov, cov.T, atol=1e-12)

    def test_positive_semidefinite(self, synthetic_factor_data: dict) -> None:
        d = synthetic_factor_data
        exposures = estimate_factor_exposures(d["returns"], d["factor_returns"])
        f_cov = np.cov(d["factor_returns"], rowvar=False)
        cov = factor_model_cov(exposures, f_cov)
        eigenvalues = np.linalg.eigvalsh(cov)
        assert np.all(eigenvalues >= -1e-10)


# ---------------------------------------------------------------------------
# Tests: factor_expected_returns
# ---------------------------------------------------------------------------

class TestFactorExpectedReturns:
    def test_correct_shape(self, synthetic_factor_data: dict) -> None:
        d = synthetic_factor_data
        exposures = estimate_factor_exposures(d["returns"], d["factor_returns"])
        premium = d["factor_returns"].mean(axis=0)
        mu = factor_expected_returns(exposures, premium)
        assert mu.shape == (d["n_assets"],)


# ---------------------------------------------------------------------------
# Tests: build_factor_model
# ---------------------------------------------------------------------------

class TestBuildFactorModel:
    def test_end_to_end(self, synthetic_factor_data: dict) -> None:
        d = synthetic_factor_data
        result = build_factor_model(d["returns"], d["factor_returns"])

        assert isinstance(result, FactorModelResult)
        assert result.expected_returns.shape == (d["n_assets"],)
        assert result.factor_cov.shape == (d["n_factors"], d["n_factors"])
        assert result.cov.shape == (d["n_assets"], d["n_assets"])
        assert isinstance(result.exposures, FactorExposureResult)

    def test_with_window(self, synthetic_factor_data: dict) -> None:
        d = synthetic_factor_data
        result = build_factor_model(
            d["returns"], d["factor_returns"], window=200
        )
        assert result.cov.shape == (d["n_assets"], d["n_assets"])

    def test_single_factor_1d(self, rng: np.random.Generator) -> None:
        """Passing 1-D factor returns should work via atleast_2d."""
        n_obs, n_assets = 200, 3
        factor_returns = rng.normal(0.001, 0.01, n_obs)
        asset_returns = (
            np.outer(factor_returns, rng.standard_normal(n_assets))
            + rng.normal(0, 0.005, (n_obs, n_assets))
        )
        result = build_factor_model(asset_returns, factor_returns)
        assert result.factor_cov.shape == (1, 1)
        assert result.cov.shape == (n_assets, n_assets)


# ---------------------------------------------------------------------------
# Tests: risk_decomposition
# ---------------------------------------------------------------------------

class TestRiskDecomposition:
    def test_systematic_plus_idiosyncratic_equals_total(
        self, synthetic_factor_data: dict
    ) -> None:
        d = synthetic_factor_data
        exposures = estimate_factor_exposures(d["returns"], d["factor_returns"])
        f_cov = np.cov(d["factor_returns"], rowvar=False)

        weights = np.ones(d["n_assets"]) / d["n_assets"]
        decomp = risk_decomposition(weights, exposures, f_cov)

        np.testing.assert_allclose(
            decomp["systematic_variance"] + decomp["idiosyncratic_variance"],
            decomp["total_variance"],
            rtol=1e-10,
        )

    def test_systematic_pct_in_range(self, synthetic_factor_data: dict) -> None:
        d = synthetic_factor_data
        exposures = estimate_factor_exposures(d["returns"], d["factor_returns"])
        f_cov = np.cov(d["factor_returns"], rowvar=False)
        weights = np.ones(d["n_assets"]) / d["n_assets"]
        decomp = risk_decomposition(weights, exposures, f_cov)

        assert 0.0 <= decomp["systematic_pct"] <= 1.0

    def test_factor_contributions_shape(self, synthetic_factor_data: dict) -> None:
        d = synthetic_factor_data
        exposures = estimate_factor_exposures(d["returns"], d["factor_returns"])
        f_cov = np.cov(d["factor_returns"], rowvar=False)
        weights = np.ones(d["n_assets"]) / d["n_assets"]
        decomp = risk_decomposition(weights, exposures, f_cov)

        assert decomp["factor_contributions"].shape == (d["n_factors"],)

    def test_factor_contributions_sum_to_systematic(
        self, synthetic_factor_data: dict
    ) -> None:
        d = synthetic_factor_data
        exposures = estimate_factor_exposures(d["returns"], d["factor_returns"])
        f_cov = np.cov(d["factor_returns"], rowvar=False)
        weights = np.ones(d["n_assets"]) / d["n_assets"]
        decomp = risk_decomposition(weights, exposures, f_cov)

        np.testing.assert_allclose(
            decomp["factor_contributions"].sum(),
            decomp["systematic_variance"],
            rtol=1e-10,
        )
