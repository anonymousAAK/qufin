"""Unit tests for American option pricing via quantum-accelerated LSM."""

from __future__ import annotations

import numpy as np
import pytest

from qufin.options.amplitude_estimation.american_qae import (
    AmericanQAESpec,
    BasisType,
    QuantumLSM,
    ResourceEstimate,
    american_binomial,
    build_basis,
    estimate_resources,
    price_american_classical,
    price_american_qae,
)


# ---------------------------------------------------------------------------
# Basis function encoding tests
# ---------------------------------------------------------------------------


class TestBasisFunctions:
    """Tests for basis function construction."""

    def test_polynomial_basis_shape(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        basis = build_basis(x, BasisType.POLYNOMIAL, degree=3)
        assert basis.shape == (5, 4)

    def test_polynomial_basis_constant_column(self) -> None:
        x = np.array([1.0, 2.0, 3.0])
        basis = build_basis(x, BasisType.POLYNOMIAL, degree=2)
        # First column (degree 0) should be all ones
        np.testing.assert_allclose(basis[:, 0], 1.0)

    def test_laguerre_basis_shape(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0])
        basis = build_basis(x, BasisType.LAGUERRE, degree=3)
        assert basis.shape == (4, 4)

    def test_laguerre_basis_l0_is_ones(self) -> None:
        x = np.array([1.0, 2.0, 3.0])
        basis = build_basis(x, BasisType.LAGUERRE, degree=2)
        np.testing.assert_allclose(basis[:, 0], 1.0)

    def test_polynomial_degree_zero(self) -> None:
        x = np.array([10.0, 20.0, 30.0])
        basis = build_basis(x, BasisType.POLYNOMIAL, degree=0)
        assert basis.shape == (3, 1)
        np.testing.assert_allclose(basis[:, 0], 1.0)

    def test_laguerre_degree_one(self) -> None:
        x = np.array([1.0, 2.0, 3.0])
        basis = build_basis(x, BasisType.LAGUERRE, degree=1)
        assert basis.shape == (3, 2)
        # L_0 = 1
        np.testing.assert_allclose(basis[:, 0], 1.0)

    def test_basis_not_nan(self) -> None:
        x = np.array([90.0, 95.0, 100.0, 105.0, 110.0])
        for bt in BasisType:
            basis = build_basis(x, bt, degree=4)
            assert not np.any(np.isnan(basis)), f"NaN in {bt.value} basis"

    def test_basis_handles_identical_values(self) -> None:
        """Regression test: constant input should not cause division by zero."""
        x = np.array([100.0, 100.0, 100.0])
        basis = build_basis(x, BasisType.POLYNOMIAL, degree=2)
        assert not np.any(np.isnan(basis))
        assert not np.any(np.isinf(basis))


# ---------------------------------------------------------------------------
# Classical regression tests
# ---------------------------------------------------------------------------


class TestClassicalRegression:
    """Tests for the classical OLS regression fallback."""

    def test_linear_fit(self) -> None:
        """Regression should perfectly fit a linear relationship."""
        x = np.linspace(80, 120, 50)
        y = 2.0 * x + 3.0
        basis = build_basis(x, BasisType.POLYNOMIAL, degree=1)
        predicted = QuantumLSM._classical_regression(basis, y)
        np.testing.assert_allclose(predicted, y, atol=1e-8)

    def test_quadratic_fit(self) -> None:
        x = np.linspace(80, 120, 50)
        y = 0.5 * x**2 - 10.0 * x + 100.0
        basis = build_basis(x, BasisType.POLYNOMIAL, degree=2)
        predicted = QuantumLSM._classical_regression(basis, y)
        np.testing.assert_allclose(predicted, y, atol=1e-6)


# ---------------------------------------------------------------------------
# American put pricing: >= European put
# ---------------------------------------------------------------------------


class TestAmericanPutPricing:
    """Verify American put >= European put (early exercise premium)."""

    def test_american_put_geq_european_put_classical(self) -> None:
        """American put (LSM) should be >= European put (BS)."""
        s0, k, r, sigma, T = 100.0, 100.0, 0.05, 0.2, 1.0

        result = price_american_classical(
            s0=s0, k=k, r=r, sigma=sigma, T=T,
            is_call=False, n_steps=50, n_paths=50_000, seed=42,
        )
        american_price = result["price"]

        # Black-Scholes European put
        from scipy.stats import norm

        d1 = (np.log(s0 / k) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        european_put = k * np.exp(-r * T) * norm.cdf(-d2) - s0 * norm.cdf(-d1)

        assert american_price >= european_put - 0.5, (
            f"American put {american_price:.4f} should be >= "
            f"European put {european_put:.4f}"
        )

    def test_american_put_via_price_function(self) -> None:
        """price_american_qae with no backend should give sensible put price."""
        result = price_american_qae(
            s0=100.0, k=100.0, r=0.05, sigma=0.2, T=1.0,
            is_call=False, n_steps=30, n_paths=10_000,
            backend=None, seed=42,
        )
        assert result.price > 0
        assert result.std_err > 0
        assert result.std_err < result.price
        assert not result.quantum_regression_used

    def test_american_call_no_early_exercise_no_dividend(self) -> None:
        """For a non-dividend American call, early exercise is never optimal.

        The American call price should approximately equal the European call.
        """
        s0, k, r, sigma, T = 100.0, 100.0, 0.05, 0.3, 1.0

        result = price_american_classical(
            s0=s0, k=k, r=r, sigma=sigma, T=T,
            is_call=True, n_steps=50, n_paths=50_000, seed=42,
        )
        american_call = result["price"]

        from scipy.stats import norm

        d1 = (np.log(s0 / k) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        european_call = s0 * norm.cdf(d1) - k * np.exp(-r * T) * norm.cdf(d2)

        # Should be very close since no dividends
        assert abs(american_call - european_call) < 1.5, (
            f"American call {american_call:.4f} should be close to "
            f"European call {european_call:.4f}"
        )


# ---------------------------------------------------------------------------
# Comparison with binomial tree
# ---------------------------------------------------------------------------


class TestBinomialComparison:
    """Compare LSM prices against the CRR binomial tree reference."""

    def test_put_vs_binomial(self) -> None:
        s0, k, r, sigma, T = 100.0, 100.0, 0.05, 0.2, 1.0

        binom_price = american_binomial(
            s0=s0, k=k, r=r, sigma=sigma, T=T, is_call=False, n_steps=500,
        )
        lsm_result = price_american_classical(
            s0=s0, k=k, r=r, sigma=sigma, T=T,
            is_call=False, n_steps=50, n_paths=100_000, seed=42,
        )
        lsm_price = lsm_result["price"]

        # LSM should be within ~1% of binomial for this standard case
        assert abs(lsm_price - binom_price) < 0.5, (
            f"LSM {lsm_price:.4f} vs binomial {binom_price:.4f}"
        )

    def test_deep_itm_put(self) -> None:
        """Deep in-the-money put should have price close to discounted intrinsic."""
        s0, k, r, sigma, T = 80.0, 120.0, 0.05, 0.2, 1.0

        binom_price = american_binomial(
            s0=s0, k=k, r=r, sigma=sigma, T=T, is_call=False, n_steps=500,
        )
        # Deep ITM American put should be close to intrinsic value
        assert binom_price >= k - s0 - 0.01
        assert binom_price > 0

    def test_otm_put(self) -> None:
        """Out-of-the-money put should still have positive value."""
        binom_price = american_binomial(
            s0=120.0, k=100.0, r=0.05, sigma=0.2, T=1.0, is_call=False,
        )
        assert binom_price > 0
        assert binom_price < 10  # OTM, shouldn't be too large


# ---------------------------------------------------------------------------
# QuantumLSM class tests
# ---------------------------------------------------------------------------


class TestQuantumLSM:
    """Tests for the QuantumLSM class."""

    def test_classical_fallback(self) -> None:
        """QuantumLSM without backend should use classical regression."""
        spec = AmericanQAESpec(
            s0=100.0, k=100.0, r=0.05, sigma=0.2, T=1.0,
            is_call=False, n_steps=20, n_paths=5_000, seed=42,
        )
        qlsm = QuantumLSM(spec, backend=None)
        result = qlsm.price()

        assert result.price > 0
        assert not result.quantum_regression_used
        assert result.classical_price > 0

    def test_exercise_boundary_populated(self) -> None:
        """Exercise boundary should have entries for an ITM put."""
        spec = AmericanQAESpec(
            s0=100.0, k=110.0, r=0.05, sigma=0.3, T=1.0,
            is_call=False, n_steps=20, n_paths=10_000, seed=42,
        )
        qlsm = QuantumLSM(spec, backend=None)
        result = qlsm.price()

        assert len(result.exercise_boundary) > 0
        # Boundary values should be positive stock prices
        for step, s_boundary in result.exercise_boundary.items():
            assert s_boundary > 0
            assert isinstance(step, int)

    def test_laguerre_basis(self) -> None:
        """Should work with Laguerre basis functions."""
        spec = AmericanQAESpec(
            s0=100.0, k=100.0, r=0.05, sigma=0.2, T=1.0,
            is_call=False, n_steps=20, n_paths=5_000,
            basis_type=BasisType.LAGUERRE, basis_degree=3, seed=42,
        )
        qlsm = QuantumLSM(spec, backend=None)
        result = qlsm.price()
        assert result.price > 0

    def test_result_has_wall_time(self) -> None:
        spec = AmericanQAESpec(
            s0=100.0, k=100.0, r=0.05, sigma=0.2, T=1.0,
            is_call=False, n_steps=10, n_paths=1_000, seed=42,
        )
        qlsm = QuantumLSM(spec, backend=None)
        result = qlsm.price()
        assert result.wall_time_s > 0

    def test_result_serializable(self) -> None:
        """Result should be JSON-serializable via the base Result class."""
        spec = AmericanQAESpec(
            s0=100.0, k=100.0, r=0.05, sigma=0.2, T=1.0,
            is_call=False, n_steps=10, n_paths=1_000, seed=42,
        )
        qlsm = QuantumLSM(spec, backend=None)
        result = qlsm.price()
        json_str = result.to_json()
        assert '"price"' in json_str
        assert '"exercise_boundary"' in json_str


# ---------------------------------------------------------------------------
# Resource estimation
# ---------------------------------------------------------------------------


class TestResourceEstimate:
    """Tests for quantum resource estimation."""

    def test_basic_estimate(self) -> None:
        res = estimate_resources(n_steps=50, basis_degree=3)
        assert isinstance(res, ResourceEstimate)
        assert res.n_steps == 50
        assert res.n_basis == 4  # degree 3 -> 4 basis functions
        assert res.qubits_regression > 0
        assert res.qubits_qae > res.qubits_regression
        assert res.total_qubits >= res.qubits_regression
        assert res.total_qubits >= res.qubits_qae
        assert res.circuit_depth_regression > 0
        assert res.total_circuits > 0

    def test_more_steps_more_circuits(self) -> None:
        res_10 = estimate_resources(n_steps=10)
        res_50 = estimate_resources(n_steps=50)
        assert res_50.total_circuits > res_10.total_circuits

    def test_more_layers_deeper_circuits(self) -> None:
        res_1 = estimate_resources(vqe_layers=1)
        res_4 = estimate_resources(vqe_layers=4)
        assert res_4.circuit_depth_regression > res_1.circuit_depth_regression

    def test_qae_qubits_scale_with_eval_qubits(self) -> None:
        res_3 = estimate_resources(n_eval_qubits_qae=3)
        res_8 = estimate_resources(n_eval_qubits_qae=8)
        assert res_8.qubits_qae > res_3.qubits_qae

    def test_resource_values_positive(self) -> None:
        res = estimate_resources()
        assert res.n_basis > 0
        assert res.qubits_regression > 0
        assert res.circuit_depth_regression > 0
        assert res.total_circuits > 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case and regression tests."""

    def test_zero_volatility_put(self) -> None:
        """With sigma ~ 0, American put = max(K*exp(-rT) - S0, K - S0, 0)."""
        price = american_binomial(
            s0=100.0, k=110.0, r=0.05, sigma=0.001, T=1.0,
            is_call=False, n_steps=200,
        )
        # Near-zero vol: immediate exercise gives K - S = 10
        # Holding gives K*exp(-rT) - S0 = 110*exp(-0.05) - 100 ~ 4.63
        # So early exercise is optimal, price ~ 10
        assert abs(price - 10.0) < 0.5

    def test_very_short_maturity(self) -> None:
        """Short maturity option should be close to intrinsic."""
        price = american_binomial(
            s0=95.0, k=100.0, r=0.05, sigma=0.2, T=0.01,
            is_call=False, n_steps=100,
        )
        intrinsic = max(100.0 - 95.0, 0.0)
        assert abs(price - intrinsic) < 0.5

    def test_spec_defaults(self) -> None:
        spec = AmericanQAESpec()
        assert spec.s0 == 100.0
        assert spec.is_call is False
        assert spec.basis_type == BasisType.POLYNOMIAL
        assert spec.basis_degree == 3
