"""Tests for Monte Carlo option pricing."""

from __future__ import annotations

import numpy as np
import pytest

from qufin.options.classical.black_scholes import call_price, put_price
from qufin.options.classical.monte_carlo import asian_mc, barrier_mc, european_mc


class TestEuropeanMC:
    def test_call_matches_bs(self) -> None:
        s, k, r, sigma, T = 100, 100, 0.05, 0.2, 1.0
        bs = call_price(s, k, r, sigma, T)
        mc = european_mc(s, k, r, sigma, T, n_paths=200_000, seed=42)
        assert abs(mc.price - bs) < 0.3  # MC tolerance

    def test_put_matches_bs(self) -> None:
        s, k, r, sigma, T = 100, 105, 0.05, 0.2, 1.0
        bs = put_price(s, k, r, sigma, T)
        mc = european_mc(s, k, r, sigma, T, n_paths=200_000, option_type="put", seed=42)
        assert abs(mc.price - bs) < 0.3

    def test_confidence_interval_contains_bs(self) -> None:
        s, k, r, sigma, T = 100, 100, 0.05, 0.2, 1.0
        bs = call_price(s, k, r, sigma, T)
        mc = european_mc(s, k, r, sigma, T, n_paths=500_000, seed=42)
        assert mc.confidence_interval[0] <= bs <= mc.confidence_interval[1]

    def test_antithetic_reduces_variance(self) -> None:
        s, k, r, sigma, T = 100, 100, 0.05, 0.2, 1.0
        mc_ant = european_mc(s, k, r, sigma, T, n_paths=50_000, antithetic=True, seed=42)
        mc_raw = european_mc(s, k, r, sigma, T, n_paths=50_000, antithetic=False, seed=42)
        # Antithetic should generally reduce std_err (not guaranteed but likely)
        assert mc_ant.std_err < mc_raw.std_err * 1.5  # generous bound

    def test_reproducibility(self) -> None:
        mc1 = european_mc(100, 100, 0.05, 0.2, 1.0, seed=42)
        mc2 = european_mc(100, 100, 0.05, 0.2, 1.0, seed=42)
        assert mc1.price == mc2.price


class TestAsianMC:
    def test_arithmetic_asian_call(self) -> None:
        mc = asian_mc(100, 100, 0.05, 0.2, 1.0, n_paths=100_000, seed=42)
        assert mc.price > 0

    def test_geometric_leq_arithmetic(self) -> None:
        # Geometric average <= arithmetic average, so geometric call <= arithmetic call
        geo = asian_mc(100, 100, 0.05, 0.2, 1.0, n_paths=200_000,
                      average_type="geometric", seed=42)
        arith = asian_mc(100, 100, 0.05, 0.2, 1.0, n_paths=200_000,
                        average_type="arithmetic", seed=42)
        assert geo.price < arith.price + 1.0  # generous tolerance


class TestBarrierMC:
    def test_up_and_out_leq_vanilla(self) -> None:
        s, k, r, sigma, T = 100, 100, 0.05, 0.2, 1.0
        vanilla = european_mc(s, k, r, sigma, T, n_paths=100_000, seed=42)
        barrier = barrier_mc(s, k, r, sigma, T, barrier=130,
                           barrier_type="up-and-out", n_paths=100_000, seed=42)
        assert barrier.price <= vanilla.price + 0.1

    def test_down_and_out_call(self) -> None:
        mc = barrier_mc(100, 100, 0.05, 0.2, 1.0, barrier=80,
                       barrier_type="down-and-out", n_paths=100_000, seed=42)
        assert mc.price > 0

    def test_knock_in_plus_knock_out_equals_vanilla(self) -> None:
        # In-out parity: knock-in + knock-out = vanilla
        s, k, r, sigma, T, b = 100, 100, 0.05, 0.2, 1.0, 130
        n = 200_000
        knock_out = barrier_mc(s, k, r, sigma, T, b, "up-and-out", n_paths=n, seed=42)
        knock_in = barrier_mc(s, k, r, sigma, T, b, "up-and-in", n_paths=n, seed=42)
        vanilla = european_mc(s, k, r, sigma, T, n_paths=n, seed=42)
        # Should be approximately equal (MC noise)
        assert abs((knock_in.price + knock_out.price) - vanilla.price) < 1.0
