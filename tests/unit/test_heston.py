"""Unit tests for Heston model pricing."""

from __future__ import annotations

import numpy as np
import pytest

from qufin.options.heston import (
    HestonParams,
    heston_european_price,
    heston_terminal_distribution,
    heston_weak_euler_terminal,
    heston_strong_euler_terminal,
    resource_estimates,
)


@pytest.fixture
def default_params() -> HestonParams:
    return HestonParams(
        s0=100, v0=0.04, r=0.05, kappa=2.0,
        theta=0.04, xi=0.3, rho=-0.7, T=1.0,
    )


class TestHestonSimulation:
    def test_weak_euler_shape(self, default_params: HestonParams) -> None:
        s_T = heston_weak_euler_terminal(default_params, n_paths=1000, seed=42)
        assert s_T.shape == (1000,)
        assert np.all(s_T > 0)

    def test_strong_euler_shape(self, default_params: HestonParams) -> None:
        s_T = heston_strong_euler_terminal(default_params, n_paths=1000, seed=42)
        assert s_T.shape == (1000,)
        assert np.all(s_T > 0)

    def test_deterministic(self, default_params: HestonParams) -> None:
        s1 = heston_weak_euler_terminal(default_params, n_paths=100, seed=42)
        s2 = heston_weak_euler_terminal(default_params, n_paths=100, seed=42)
        np.testing.assert_array_equal(s1, s2)

    def test_mean_close_to_forward(self, default_params: HestonParams) -> None:
        """Mean terminal price should be close to S0 * exp(r*T)."""
        s_T = heston_weak_euler_terminal(default_params, n_paths=200_000, n_steps=200, seed=42)
        forward = default_params.s0 * np.exp(default_params.r * default_params.T)
        assert abs(np.mean(s_T) - forward) / forward < 0.02

    def test_weak_vs_strong_similar(self, default_params: HestonParams) -> None:
        """Weak and strong Euler should give similar means."""
        weak = heston_weak_euler_terminal(default_params, n_paths=100_000, seed=42)
        strong = heston_strong_euler_terminal(default_params, n_paths=100_000, seed=42)
        # Means should be close (same expected value)
        assert abs(np.mean(weak) - np.mean(strong)) / np.mean(weak) < 0.03


class TestHestonPricing:
    def test_call_price_positive(self, default_params: HestonParams) -> None:
        price, std_err = heston_european_price(
            default_params, k=100, is_call=True,
            n_paths=50_000, seed=42,
        )
        assert price > 0
        assert std_err > 0

    def test_put_price_positive(self, default_params: HestonParams) -> None:
        price, std_err = heston_european_price(
            default_params, k=100, is_call=False,
            n_paths=50_000, seed=42,
        )
        assert price > 0

    def test_put_call_parity(self, default_params: HestonParams) -> None:
        """C - P ~ S0 - K*exp(-rT) (approximate due to MC noise)."""
        call, _ = heston_european_price(default_params, k=100, is_call=True, n_paths=200_000, seed=42)
        put, _ = heston_european_price(default_params, k=100, is_call=False, n_paths=200_000, seed=42)
        parity = default_params.s0 - 100 * np.exp(-default_params.r * default_params.T)
        assert abs((call - put) - parity) < 1.0  # generous tolerance for MC


class TestHestonDistribution:
    def test_distribution_shape(self, default_params: HestonParams) -> None:
        dist = heston_terminal_distribution(
            default_params, n_qubits=3, n_paths=10_000, seed=42,
        )
        assert dist.n_states == 8
        assert dist.probabilities.shape == (8,)
        assert abs(dist.probabilities.sum() - 1.0) < 1e-10

    def test_positive_domain(self, default_params: HestonParams) -> None:
        dist = heston_terminal_distribution(
            default_params, n_qubits=3, n_paths=10_000, seed=42,
        )
        assert dist.low > 0


class TestResourceEstimates:
    def test_basic(self) -> None:
        res = resource_estimates(n_qubits_price=4, n_qubits_vol=4, n_steps=1)
        assert res["total_qubits"] > 0
        assert res["T_count"] > 0
        assert res["T_depth"] > 0
        assert res["cnot_count"] > 0

    def test_scaling(self) -> None:
        """More qubits/steps should increase resource counts."""
        r1 = resource_estimates(n_qubits_price=4, n_steps=1)
        r2 = resource_estimates(n_qubits_price=8, n_steps=1)
        assert r2["T_count"] > r1["T_count"]

        r3 = resource_estimates(n_qubits_price=4, n_steps=2)
        assert r3["T_count"] > r1["T_count"]
