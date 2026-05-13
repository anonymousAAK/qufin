"""Unit tests for multi-asset option pricing via QAE."""

from __future__ import annotations

import numpy as np
import pytest

from qufin.options.amplitude_estimation.multi_asset_qae import (
    MultiAssetSpec,
    build_multi_asset_estimation_problem,
    price_multi_asset_mc,
)

# ---------------------------------------------------------------------------
# Tests: MultiAssetSpec validation
# ---------------------------------------------------------------------------

class TestMultiAssetSpec:
    def test_default_construction(self) -> None:
        spec = MultiAssetSpec()
        assert spec.n_assets == 2
        assert spec.effective_weights == [0.5, 0.5]

    def test_correlation_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="Correlation matrix shape"):
            MultiAssetSpec(
                spots=[100.0, 100.0, 100.0],
                sigma=[0.2, 0.2, 0.2],
                correlation=np.eye(2),  # wrong shape for 3 assets
            )

    def test_sigma_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="Length of sigma"):
            MultiAssetSpec(
                spots=[100.0, 100.0],
                sigma=[0.2],  # wrong length
            )

    def test_weights_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="Length of weights"):
            MultiAssetSpec(
                spots=[100.0, 100.0],
                sigma=[0.2, 0.2],
                weights=[0.5],  # wrong length
            )

    def test_invalid_payoff_type_raises(self) -> None:
        with pytest.raises(ValueError, match="payoff_type"):
            MultiAssetSpec(payoff_type="invalid")

    def test_custom_weights(self) -> None:
        spec = MultiAssetSpec(
            spots=[100.0, 100.0],
            sigma=[0.2, 0.2],
            weights=[0.3, 0.7],
        )
        assert spec.effective_weights == [0.3, 0.7]


# ---------------------------------------------------------------------------
# Tests: price_multi_asset_mc
# ---------------------------------------------------------------------------

class TestPriceMultiAssetMC:
    def test_returns_expected_keys(self) -> None:
        spec = MultiAssetSpec()
        result = price_multi_asset_mc(spec, n_paths=10_000, seed=42)
        expected_keys = {"price", "std_error", "ci_low", "ci_high", "n_paths"}
        assert set(result.keys()) == expected_keys

    def test_basket_call_nonnegative(self) -> None:
        spec = MultiAssetSpec(payoff_type="basket_call")
        result = price_multi_asset_mc(spec, n_paths=10_000, seed=42)
        assert result["price"] >= 0.0

    def test_basket_put_nonnegative(self) -> None:
        spec = MultiAssetSpec(payoff_type="basket_put")
        result = price_multi_asset_mc(spec, n_paths=10_000, seed=42)
        assert result["price"] >= 0.0

    def test_two_uncorrelated_assets(self) -> None:
        spec = MultiAssetSpec(
            spots=[100.0, 100.0],
            strikes=100.0,
            r=0.05,
            sigma=[0.2, 0.2],
            correlation=np.eye(2),
            T=1.0,
            payoff_type="basket_call",
        )
        result = price_multi_asset_mc(spec, n_paths=50_000, seed=42)
        assert result["price"] > 0.0
        assert result["std_error"] > 0.0
        assert result["ci_low"] < result["ci_high"]

    def test_three_correlated_assets(self) -> None:
        corr = np.array([
            [1.0, 0.5, 0.3],
            [0.5, 1.0, 0.4],
            [0.3, 0.4, 1.0],
        ])
        spec = MultiAssetSpec(
            spots=[100.0, 110.0, 90.0],
            strikes=100.0,
            r=0.05,
            sigma=[0.2, 0.25, 0.15],
            correlation=corr,
            T=1.0,
            payoff_type="basket_call",
        )
        result = price_multi_asset_mc(spec, n_paths=50_000, seed=42)
        assert result["price"] > 0.0
        assert result["n_paths"] == 50_000

    def test_best_of_call(self) -> None:
        spec = MultiAssetSpec(
            spots=[100.0, 100.0],
            strikes=100.0,
            sigma=[0.2, 0.2],
            payoff_type="best_of_call",
        )
        result = price_multi_asset_mc(spec, n_paths=20_000, seed=42)
        assert result["price"] > 0.0

    def test_worst_of_call(self) -> None:
        spec = MultiAssetSpec(
            spots=[100.0, 100.0],
            strikes=100.0,
            sigma=[0.2, 0.2],
            payoff_type="worst_of_call",
        )
        result = price_multi_asset_mc(spec, n_paths=20_000, seed=42)
        # worst-of-call price should be >= 0
        assert result["price"] >= 0.0

    def test_best_of_call_gte_basket_call(self) -> None:
        """best-of-call should be at least as expensive as basket call."""
        spec_basket = MultiAssetSpec(
            spots=[100.0, 100.0],
            strikes=100.0,
            sigma=[0.2, 0.2],
            payoff_type="basket_call",
        )
        spec_best = MultiAssetSpec(
            spots=[100.0, 100.0],
            strikes=100.0,
            sigma=[0.2, 0.2],
            payoff_type="best_of_call",
        )
        mc_basket = price_multi_asset_mc(spec_basket, n_paths=50_000, seed=42)
        mc_best = price_multi_asset_mc(spec_best, n_paths=50_000, seed=42)
        # best-of should be >= basket with some tolerance for MC noise
        assert mc_best["price"] >= mc_basket["price"] - 3 * mc_best["std_error"]


# ---------------------------------------------------------------------------
# Tests: build_multi_asset_estimation_problem
# ---------------------------------------------------------------------------

class TestBuildEstimationProblem:
    def test_returns_problem_and_rescale(self) -> None:
        spec = MultiAssetSpec(
            spots=[100.0, 100.0],
            strikes=100.0,
            sigma=[0.2, 0.2],
            n_qubits_per_asset=2,
        )
        problem, rescale = build_multi_asset_estimation_problem(spec)
        assert rescale > 0.0
        assert problem.n_qubits == 2 * 2 + 1  # 2 assets * 2 qubits + 1 ancilla
        assert problem.objective_qubits == [4]

    def test_three_asset_qubit_count(self) -> None:
        corr = np.array([
            [1.0, 0.5, 0.3],
            [0.5, 1.0, 0.4],
            [0.3, 0.4, 1.0],
        ])
        spec = MultiAssetSpec(
            spots=[100.0, 110.0, 90.0],
            strikes=100.0,
            sigma=[0.2, 0.25, 0.15],
            correlation=corr,
            n_qubits_per_asset=2,
        )
        problem, rescale = build_multi_asset_estimation_problem(spec)
        assert problem.n_qubits == 3 * 2 + 1  # 3 assets * 2 qubits + 1 ancilla
        assert rescale > 0.0
