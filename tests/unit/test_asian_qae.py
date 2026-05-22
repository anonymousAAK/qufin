"""Unit tests for qufin.options.amplitude_estimation.asian_qae module."""

from __future__ import annotations

import numpy as np
import pytest

from qufin.options.amplitude_estimation.asian_qae import (
    AsianQAEResult,
    AsianQAESpec,
    _classical_asian_mc,
    _compute_asian_payoffs,
    _discretise_gbm_step,
    compare_asian_pricing,
    generate_path_distribution,
    price_asian_option_qae,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def call_spec() -> AsianQAESpec:
    return AsianQAESpec(
        s0=100.0, k=100.0, r=0.05, sigma=0.2, T=1.0,
        n_steps=4, is_call=True, averaging="arithmetic",
        n_price_qubits=3, n_precision_qubits=4,
    )


@pytest.fixture
def put_spec() -> AsianQAESpec:
    return AsianQAESpec(
        s0=100.0, k=105.0, r=0.05, sigma=0.2, T=1.0,
        n_steps=4, is_call=False, averaging="arithmetic",
        n_price_qubits=3, n_precision_qubits=4,
    )


@pytest.fixture
def geo_spec() -> AsianQAESpec:
    return AsianQAESpec(
        s0=100.0, k=100.0, r=0.05, sigma=0.2, T=1.0,
        n_steps=4, is_call=True, averaging="geometric",
        n_price_qubits=3, n_precision_qubits=4,
    )


# ---------------------------------------------------------------------------
# Spec tests
# ---------------------------------------------------------------------------


class TestAsianQAESpec:
    def test_defaults(self) -> None:
        spec = AsianQAESpec()
        assert spec.s0 == 100.0
        assert spec.k == 100.0
        assert spec.n_steps == 4
        assert spec.averaging == "arithmetic"

    def test_custom(self) -> None:
        spec = AsianQAESpec(s0=50.0, k=55.0, n_steps=8, averaging="geometric")
        assert spec.s0 == 50.0
        assert spec.averaging == "geometric"


# ---------------------------------------------------------------------------
# Path discretisation
# ---------------------------------------------------------------------------


class TestDiscretiseGBMStep:
    def test_output_shapes(self) -> None:
        s_prev = np.array([100.0])
        prices, trans = _discretise_gbm_step(s_prev, 0.25, 0.05, 0.2, 3)
        assert prices.shape == (8,)
        assert trans.shape == (1, 8)

    def test_probabilities_sum_to_one(self) -> None:
        s_prev = np.array([100.0, 105.0])
        _prices, trans = _discretise_gbm_step(s_prev, 0.25, 0.05, 0.2, 3)
        for row in trans:
            assert row.sum() == pytest.approx(1.0, abs=1e-6)

    def test_prices_positive(self) -> None:
        s_prev = np.array([100.0])
        prices, _ = _discretise_gbm_step(s_prev, 0.25, 0.05, 0.2, 3)
        assert np.all(prices > 0)


# ---------------------------------------------------------------------------
# Path distribution generation
# ---------------------------------------------------------------------------


class TestGeneratePathDistribution:
    def test_length(self, call_spec: AsianQAESpec) -> None:
        grids, probs = generate_path_distribution(call_spec)
        # n_steps + 1 (initial + each step)
        assert len(grids) == call_spec.n_steps + 1
        assert len(probs) == call_spec.n_steps + 1

    def test_initial_spot(self, call_spec: AsianQAESpec) -> None:
        grids, probs = generate_path_distribution(call_spec)
        assert grids[0][0] == call_spec.s0
        assert probs[0][0] == 1.0

    def test_terminal_probs_sum(self, call_spec: AsianQAESpec) -> None:
        _, probs = generate_path_distribution(call_spec)
        assert probs[-1].sum() == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Payoff computation
# ---------------------------------------------------------------------------


class TestComputeAsianPayoffs:
    def test_call_payoffs_non_negative(self, call_spec: AsianQAESpec) -> None:
        grids, probs = generate_path_distribution(call_spec)
        payoffs, _ = _compute_asian_payoffs(grids, probs, call_spec)
        assert np.all(payoffs >= 0)

    def test_put_payoffs_non_negative(self, put_spec: AsianQAESpec) -> None:
        grids, probs = generate_path_distribution(put_spec)
        payoffs, _ = _compute_asian_payoffs(grids, probs, put_spec)
        assert np.all(payoffs >= 0)

    def test_geometric_averaging(self, geo_spec: AsianQAESpec) -> None:
        grids, probs = generate_path_distribution(geo_spec)
        payoffs, _ = _compute_asian_payoffs(grids, probs, geo_spec)
        assert np.all(payoffs >= 0)


# ---------------------------------------------------------------------------
# Classical MC pricing
# ---------------------------------------------------------------------------


class TestClassicalAsianMC:
    def test_call_price_positive(self, call_spec: AsianQAESpec) -> None:
        price = _classical_asian_mc(call_spec, n_samples=5000, seed=42)
        assert price >= 0

    def test_put_price_positive(self, put_spec: AsianQAESpec) -> None:
        price = _classical_asian_mc(put_spec, n_samples=5000, seed=42)
        assert price >= 0

    def test_geometric_price_positive(self, geo_spec: AsianQAESpec) -> None:
        price = _classical_asian_mc(geo_spec, n_samples=5000, seed=42)
        assert price >= 0

    def test_reproducibility(self, call_spec: AsianQAESpec) -> None:
        p1 = _classical_asian_mc(call_spec, n_samples=1000, seed=99)
        p2 = _classical_asian_mc(call_spec, n_samples=1000, seed=99)
        assert p1 == p2


# ---------------------------------------------------------------------------
# QAE pricing
# ---------------------------------------------------------------------------


class TestPriceAsianOptionQAE:
    def test_returns_result(self, call_spec: AsianQAESpec) -> None:
        result = price_asian_option_qae(call_spec, seed=42)
        assert isinstance(result, AsianQAEResult)

    def test_price_non_negative(self, call_spec: AsianQAESpec) -> None:
        result = price_asian_option_qae(call_spec, seed=42)
        assert result.price >= 0

    def test_std_error_non_negative(self, call_spec: AsianQAESpec) -> None:
        result = price_asian_option_qae(call_spec, seed=42)
        assert result.std_error >= 0

    def test_wall_time_positive(self, call_spec: AsianQAESpec) -> None:
        result = price_asian_option_qae(call_spec, seed=42)
        assert result.wall_time_s > 0

    def test_averaging_label(self, geo_spec: AsianQAESpec) -> None:
        result = price_asian_option_qae(geo_spec, seed=42)
        assert result.averaging == "geometric"

    def test_metadata_populated(self, call_spec: AsianQAESpec) -> None:
        result = price_asian_option_qae(call_spec, seed=42)
        assert "n_steps" in result.metadata
        assert "discount_factor" in result.metadata


# ---------------------------------------------------------------------------
# Comparison utility
# ---------------------------------------------------------------------------


class TestCompareAsianPricing:
    def test_returns_dict(self, call_spec: AsianQAESpec) -> None:
        result = compare_asian_pricing(call_spec, n_classical_samples=1000, seed=42)
        assert isinstance(result, dict)
        assert "qae_price" in result
        assert "classical_price" in result

    def test_both_prices_non_negative(self, call_spec: AsianQAESpec) -> None:
        result = compare_asian_pricing(call_spec, n_classical_samples=1000, seed=42)
        assert result["qae_price"] >= 0
        assert result["classical_price"] >= 0
