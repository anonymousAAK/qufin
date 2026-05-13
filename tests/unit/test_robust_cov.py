"""Unit tests for robust covariance estimators."""

from __future__ import annotations

import numpy as np
import pytest

from qufin.portfolio.classical.robust_cov import (
    CovEstimateResult,
    constant_correlation,
    ledoit_wolf,
    oracle_approx_shrinkage,
    select_estimator,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def wide_returns(rng: np.random.Generator) -> np.ndarray:
    """Returns where n < p (30 observations, 50 assets) -- shrinkage useful."""
    return rng.normal(0.0005, 0.02, (30, 50))


@pytest.fixture
def tall_returns(rng: np.random.Generator) -> np.ndarray:
    """Returns where n >> p (500 observations, 5 assets)."""
    return rng.normal(0.0005, 0.02, (500, 5))


def _is_psd(mat: np.ndarray, tol: float = -1e-10) -> bool:
    return bool(np.all(np.linalg.eigvalsh(mat) >= tol))


def _is_symmetric(mat: np.ndarray, atol: float = 1e-12) -> bool:
    return bool(np.allclose(mat, mat.T, atol=atol))


# ---------------------------------------------------------------------------
# Tests: ledoit_wolf
# ---------------------------------------------------------------------------

class TestLedoitWolf:
    def test_returns_cov_estimate_result(self, tall_returns: np.ndarray) -> None:
        result = ledoit_wolf(tall_returns)
        assert isinstance(result, CovEstimateResult)
        assert result.method == "ledoit_wolf"

    def test_psd_and_symmetric(self, wide_returns: np.ndarray) -> None:
        result = ledoit_wolf(wide_returns)
        assert _is_symmetric(result.cov)
        assert _is_psd(result.cov)

    def test_shrinkage_in_range(self, wide_returns: np.ndarray) -> None:
        result = ledoit_wolf(wide_returns)
        assert 0.0 <= result.shrinkage_intensity <= 1.0

    def test_correct_shape(self, wide_returns: np.ndarray) -> None:
        result = ledoit_wolf(wide_returns)
        n, p = wide_returns.shape
        assert result.cov.shape == (p, p)
        assert result.n_obs == n

    def test_same_shape_as_np_cov(self, tall_returns: np.ndarray) -> None:
        sample = np.cov(tall_returns, rowvar=False)
        result = ledoit_wolf(tall_returns)
        assert result.cov.shape == sample.shape


# ---------------------------------------------------------------------------
# Tests: oracle_approx_shrinkage
# ---------------------------------------------------------------------------

class TestOracleApproxShrinkage:
    def test_psd_and_symmetric(self, wide_returns: np.ndarray) -> None:
        result = oracle_approx_shrinkage(wide_returns)
        assert _is_symmetric(result.cov)
        assert _is_psd(result.cov)

    def test_shrinkage_in_range(self, wide_returns: np.ndarray) -> None:
        result = oracle_approx_shrinkage(wide_returns)
        assert 0.0 <= result.shrinkage_intensity <= 1.0

    def test_method_name(self, tall_returns: np.ndarray) -> None:
        result = oracle_approx_shrinkage(tall_returns)
        assert result.method == "oas"


# ---------------------------------------------------------------------------
# Tests: constant_correlation
# ---------------------------------------------------------------------------

class TestConstantCorrelation:
    def test_psd_and_symmetric(self, wide_returns: np.ndarray) -> None:
        result = constant_correlation(wide_returns)
        assert _is_symmetric(result.cov)
        assert _is_psd(result.cov)

    def test_shrinkage_in_range(self, wide_returns: np.ndarray) -> None:
        result = constant_correlation(wide_returns)
        assert 0.0 <= result.shrinkage_intensity <= 1.0

    def test_method_name(self, tall_returns: np.ndarray) -> None:
        result = constant_correlation(tall_returns)
        assert result.method == "constant_correlation"


# ---------------------------------------------------------------------------
# Tests: select_estimator
# ---------------------------------------------------------------------------

class TestSelectEstimator:
    @pytest.mark.parametrize("method", ["ledoit_wolf", "oas", "constant_correlation", "sample"])
    def test_dispatches_correctly(self, method: str, tall_returns: np.ndarray) -> None:
        result = select_estimator(tall_returns, method=method)
        assert isinstance(result, CovEstimateResult)
        if method == "sample":
            assert result.shrinkage_intensity is None
        else:
            assert result.shrinkage_intensity is not None

    def test_auto_uses_lw_when_ratio_low(self, wide_returns: np.ndarray) -> None:
        """n/p = 30/50 = 0.6 < 10 --> should use ledoit_wolf."""
        result = select_estimator(wide_returns, method="auto")
        assert result.method == "ledoit_wolf"

    def test_auto_uses_sample_when_ratio_high(self, tall_returns: np.ndarray) -> None:
        """n/p = 500/5 = 100 >= 10 --> should use sample."""
        result = select_estimator(tall_returns, method="auto")
        assert result.method == "sample"

    def test_unknown_method_raises(self, tall_returns: np.ndarray) -> None:
        with pytest.raises(ValueError, match="Unknown method"):
            select_estimator(tall_returns, method="bogus")


# ---------------------------------------------------------------------------
# Tests: shrinkage reduces condition number
# ---------------------------------------------------------------------------

class TestShrinkageQuality:
    def test_shrinkage_reduces_condition_number(
        self, wide_returns: np.ndarray
    ) -> None:
        sample_cov = np.cov(wide_returns, rowvar=False)
        cond_sample = np.linalg.cond(sample_cov)

        lw_result = ledoit_wolf(wide_returns)
        cond_lw = np.linalg.cond(lw_result.cov)

        # Shrinkage should improve (reduce) the condition number
        assert cond_lw < cond_sample

    def test_all_estimators_same_shape(self, wide_returns: np.ndarray) -> None:
        sample = np.cov(wide_returns, rowvar=False)
        lw = ledoit_wolf(wide_returns).cov
        oas = oracle_approx_shrinkage(wide_returns).cov
        cc = constant_correlation(wide_returns).cov
        assert lw.shape == sample.shape == oas.shape == cc.shape
