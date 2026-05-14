"""Tests for Quantum Monte Carlo integration module."""

from __future__ import annotations

import numpy as np
import pytest

from qufin.options.amplitude_estimation.qmc import (
    EuropeanQMCSpec,
    QMCResourceEstimate,
    QMCResult,
    break_even_analysis,
    classical_mc_estimation,
    compare_qmc_vs_classical,
    estimate_qmc_resources,
    median_of_means_qmc,
    price_european_classical_mc,
    price_european_qmc,
    quantum_mean_estimation,
    resource_table,
)

# ---------------------------------------------------------------------------
# QMCResult dataclass tests
# ---------------------------------------------------------------------------


class TestQMCResult:
    def test_defaults(self) -> None:
        r = QMCResult()
        assert r.estimate == 0.0
        assert r.std_error == 0.0
        assert r.n_oracle_calls == 0
        assert r.method == "qmc"
        assert r.metadata == {}
        assert r.confidence_interval == (0.0, 0.0)

    def test_custom_values(self) -> None:
        r = QMCResult(
            estimate=3.14,
            std_error=0.01,
            n_oracle_calls=1024,
            speedup_factor=100.0,
            method="median_of_means",
        )
        assert r.estimate == 3.14
        assert r.speedup_factor == 100.0


class TestQMCResourceEstimate:
    def test_defaults(self) -> None:
        r = QMCResourceEstimate()
        assert r.n_logical_qubits == 0
        assert r.t_gate_count == 0
        assert r.break_even_epsilon == 0.0
        assert r.problem_description == ""

    def test_custom(self) -> None:
        r = QMCResourceEstimate(
            n_logical_qubits=20,
            t_gate_count=50000,
            problem_description="test",
        )
        assert r.n_logical_qubits == 20


class TestEuropeanQMCSpec:
    def test_defaults(self) -> None:
        spec = EuropeanQMCSpec()
        assert spec.s0 == 100.0
        assert spec.k == 100.0
        assert spec.r == 0.05
        assert spec.sigma == 0.2
        assert spec.T == 1.0
        assert spec.is_call is True
        assert spec.n_price_qubits == 4
        assert spec.n_precision_qubits == 6


# ---------------------------------------------------------------------------
# quantum_mean_estimation tests
# ---------------------------------------------------------------------------


class TestQuantumMeanEstimation:
    def test_uniform_distribution(self) -> None:
        """Mean of [0,1,2,3] with uniform weights should be ~1.5."""
        f_values = np.array([0.0, 1.0, 2.0, 3.0])
        probs = np.array([0.25, 0.25, 0.25, 0.25])
        result = quantum_mean_estimation(f_values, probs, n_precision_qubits=8, seed=42)
        assert result.method == "qmc"
        assert result.n_oracle_calls == 256  # 2^8
        assert result.n_qubits > 0
        assert result.wall_time_s > 0
        # With 8 precision qubits, estimate should be within ~0.5 of true mean
        assert abs(result.estimate - 1.5) < 1.0

    def test_delta_distribution(self) -> None:
        """All probability on one value."""
        f_values = np.array([0.0, 5.0, 0.0])
        probs = np.array([0.0, 1.0, 0.0])
        result = quantum_mean_estimation(f_values, probs, n_precision_qubits=10, seed=42)
        # Should be close to 5.0
        assert abs(result.estimate - 5.0) < 1.0

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            quantum_mean_estimation(
                np.array([1.0, 2.0]),
                np.array([0.5]),
                seed=42,
            )

    def test_confidence_interval_contains_estimate(self) -> None:
        f_values = np.array([1.0, 2.0, 3.0, 4.0])
        probs = np.array([0.1, 0.2, 0.3, 0.4])
        result = quantum_mean_estimation(f_values, probs, n_precision_qubits=6, seed=42)
        lo, hi = result.confidence_interval
        assert lo <= result.estimate <= hi

    def test_oracle_calls_scale_with_precision(self) -> None:
        f_values = np.array([1.0, 2.0, 3.0, 4.0])
        probs = np.array([0.25, 0.25, 0.25, 0.25])
        r4 = quantum_mean_estimation(f_values, probs, n_precision_qubits=4, seed=42)
        r8 = quantum_mean_estimation(f_values, probs, n_precision_qubits=8, seed=42)
        assert r8.n_oracle_calls > r4.n_oracle_calls
        assert r8.n_oracle_calls == 256
        assert r4.n_oracle_calls == 16

    def test_metadata_populated(self) -> None:
        f_values = np.array([1.0, 2.0])
        probs = np.array([0.5, 0.5])
        result = quantum_mean_estimation(f_values, probs, seed=42)
        assert "amplitude" in result.metadata
        assert "epsilon" in result.metadata
        assert "true_mean" in result.metadata

    def test_zero_probabilities(self) -> None:
        """All zero probabilities should still work (normalised to uniform)."""
        f_values = np.array([1.0, 2.0, 3.0])
        probs = np.array([0.0, 0.0, 0.0])
        # Should not raise; zero-sum gets handled
        result = quantum_mean_estimation(f_values, probs, seed=42)
        assert isinstance(result, QMCResult)

    def test_single_value(self) -> None:
        f_values = np.array([42.0])
        probs = np.array([1.0])
        result = quantum_mean_estimation(f_values, probs, n_precision_qubits=8, seed=42)
        # f_range = 0 -> defaults to 1.0
        assert isinstance(result, QMCResult)

    def test_99_confidence(self) -> None:
        """99% CI should be wider than 95% CI."""
        f_values = np.array([1.0, 2.0, 3.0, 4.0])
        probs = np.array([0.25, 0.25, 0.25, 0.25])
        r95 = quantum_mean_estimation(f_values, probs, confidence=0.95, seed=42)
        r99 = quantum_mean_estimation(f_values, probs, confidence=0.99, seed=42)
        ci95_width = r95.confidence_interval[1] - r95.confidence_interval[0]
        ci99_width = r99.confidence_interval[1] - r99.confidence_interval[0]
        assert ci99_width > ci95_width


# ---------------------------------------------------------------------------
# median_of_means_qmc tests
# ---------------------------------------------------------------------------


class TestMedianOfMeansQMC:
    def test_basic(self) -> None:
        f_values = np.array([1.0, 2.0, 3.0, 4.0])
        probs = np.array([0.25, 0.25, 0.25, 0.25])
        result = median_of_means_qmc(
            f_values, probs, n_precision_qubits=6, n_blocks=5, seed=42
        )
        assert result.method == "median_of_means"
        assert result.n_oracle_calls == 5 * 64  # 5 blocks * 2^6
        assert "n_blocks" in result.metadata
        assert result.metadata["n_blocks"] == 5

    def test_even_blocks_becomes_odd(self) -> None:
        f_values = np.array([1.0, 2.0])
        probs = np.array([0.5, 0.5])
        result = median_of_means_qmc(f_values, probs, n_blocks=4, seed=42)
        # 4 -> 5 (odd)
        assert result.metadata["n_blocks"] == 5

    def test_single_block(self) -> None:
        f_values = np.array([1.0, 2.0, 3.0])
        probs = np.array([0.33, 0.34, 0.33])
        result = median_of_means_qmc(f_values, probs, n_blocks=1, seed=42)
        assert result.metadata["n_blocks"] == 1

    def test_zero_blocks(self) -> None:
        f_values = np.array([1.0, 2.0])
        probs = np.array([0.5, 0.5])
        result = median_of_means_qmc(f_values, probs, n_blocks=0, seed=42)
        assert result.metadata["n_blocks"] == 1

    def test_block_estimates_stored(self) -> None:
        f_values = np.array([1.0, 2.0, 3.0, 4.0])
        probs = np.array([0.25, 0.25, 0.25, 0.25])
        result = median_of_means_qmc(f_values, probs, n_blocks=7, seed=42)
        assert len(result.metadata["block_estimates"]) == 7


# ---------------------------------------------------------------------------
# classical_mc_estimation tests
# ---------------------------------------------------------------------------


class TestClassicalMCEstimation:
    def test_basic(self) -> None:
        f_values = np.array([1.0, 2.0, 3.0, 4.0])
        probs = np.array([0.25, 0.25, 0.25, 0.25])
        result = classical_mc_estimation(
            f_values, probs, n_samples=10000, seed=42
        )
        assert result.method == "classical"
        assert result.n_qubits == 0
        assert result.speedup_factor == 1.0
        # Should be close to 2.5
        assert abs(result.estimate - 2.5) < 0.2

    def test_convergence(self) -> None:
        """More samples should give smaller std_error."""
        f_values = np.array([0.0, 1.0])
        probs = np.array([0.5, 0.5])
        r100 = classical_mc_estimation(f_values, probs, n_samples=100, seed=42)
        r10000 = classical_mc_estimation(f_values, probs, n_samples=10000, seed=42)
        assert r10000.std_error < r100.std_error


# ---------------------------------------------------------------------------
# European option pricing tests
# ---------------------------------------------------------------------------


class TestEuropeanQMC:
    def test_call_price_positive(self) -> None:
        spec = EuropeanQMCSpec(s0=100, k=100, r=0.05, sigma=0.2, T=1.0, is_call=True)
        result = price_european_qmc(spec, seed=42)
        assert result.estimate > 0
        assert "option_type" in result.metadata
        assert result.metadata["option_type"] == "call"

    def test_put_price_positive(self) -> None:
        spec = EuropeanQMCSpec(s0=100, k=100, r=0.05, sigma=0.2, T=1.0, is_call=False)
        result = price_european_qmc(spec, seed=42)
        assert result.estimate > 0
        assert result.metadata["option_type"] == "put"

    def test_deep_itm_call(self) -> None:
        """Deep ITM call should have high price."""
        spec = EuropeanQMCSpec(s0=150, k=100, r=0.05, sigma=0.2, T=1.0, is_call=True)
        result = price_european_qmc(spec, seed=42)
        # Intrinsic value ~ 50, discounted ~ 47.5
        assert result.estimate > 20

    def test_classical_mc_baseline(self) -> None:
        spec = EuropeanQMCSpec(s0=100, k=100)
        result = price_european_classical_mc(spec, n_samples=50000, seed=42)
        assert result.estimate > 0
        assert result.method == "classical"
        assert result.metadata["n_samples"] == 50000

    def test_compare_qmc_vs_classical(self) -> None:
        spec = EuropeanQMCSpec(s0=100, k=100, n_precision_qubits=6)
        comparison = compare_qmc_vs_classical(spec, n_classical_samples=10000, seed=42)
        assert "qmc" in comparison
        assert "classical" in comparison
        assert comparison["qmc_oracle_calls"] > 0
        assert comparison["classical_samples"] == 10000


# ---------------------------------------------------------------------------
# Resource estimation tests
# ---------------------------------------------------------------------------


class TestEstimateQMCResources:
    def test_single_asset(self) -> None:
        res = estimate_qmc_resources(n_price_qubits=4, n_precision_qubits=6, n_assets=1)
        assert res.n_logical_qubits > 0
        assert res.t_gate_count > 0
        assert res.n_physical_qubits > 0
        assert res.n_oracle_calls == 64  # 2^6
        assert "1-asset" in res.problem_description

    def test_multi_asset(self) -> None:
        res_1 = estimate_qmc_resources(n_price_qubits=4, n_precision_qubits=6, n_assets=1)
        res_3 = estimate_qmc_resources(n_price_qubits=4, n_precision_qubits=6, n_assets=3)
        assert res_3.n_logical_qubits > res_1.n_logical_qubits
        assert res_3.t_gate_count > res_1.t_gate_count
        assert "3-asset" in res_3.problem_description

    def test_precision_scaling(self) -> None:
        """More precision qubits -> more oracle calls."""
        res_4 = estimate_qmc_resources(n_precision_qubits=4)
        res_8 = estimate_qmc_resources(n_precision_qubits=8)
        assert res_8.n_oracle_calls == 16 * res_4.n_oracle_calls
        assert res_8.t_gate_count > res_4.t_gate_count

    def test_classical_equivalent(self) -> None:
        res = estimate_qmc_resources(n_precision_qubits=8)
        # Classical needs O(1/eps^2) samples, quantum O(1/eps)
        assert res.classical_samples_equivalent > res.n_oracle_calls

    def test_break_even_epsilon_positive(self) -> None:
        res = estimate_qmc_resources()
        assert res.break_even_epsilon > 0


class TestBreakEvenAnalysis:
    def test_returns_results(self) -> None:
        results = break_even_analysis(
            n_price_qubits_range=[4],
            n_precision_range=[6],
            n_assets_range=[1],
        )
        assert len(results) == 1
        assert "n_assets" in results[0]
        assert "quantum_advantage" in results[0]
        assert "advantage_ratio" in results[0]

    def test_defaults(self) -> None:
        results = break_even_analysis()
        # 3 assets * 4 price * 4 precision = 48
        assert len(results) == 48

    def test_multi_asset_more_expensive(self) -> None:
        results = break_even_analysis(
            n_price_qubits_range=[4],
            n_precision_range=[6],
            n_assets_range=[1, 5],
        )
        single = next(r for r in results if r["n_assets"] == 1)
        multi = next(r for r in results if r["n_assets"] == 5)
        assert multi["n_logical_qubits"] > single["n_logical_qubits"]


class TestResourceTable:
    def test_default(self) -> None:
        tbl = resource_table()
        assert len(tbl) == 5  # 5 precision levels
        for row in tbl:
            assert "precision_qubits" in row
            assert "epsilon" in row
            assert "logical_qubits" in row
            assert "t_gates" in row

    def test_custom_range(self) -> None:
        tbl = resource_table(precision_range=[4, 8])
        assert len(tbl) == 2
        assert tbl[0]["precision_qubits"] == 4
        assert tbl[1]["precision_qubits"] == 8

    def test_epsilon_decreases_with_precision(self) -> None:
        tbl = resource_table(precision_range=[4, 6, 8])
        epsilons = [row["epsilon"] for row in tbl]
        assert epsilons[0] > epsilons[1] > epsilons[2]
