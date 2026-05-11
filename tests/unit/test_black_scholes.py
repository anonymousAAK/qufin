"""Tests for Black-Scholes functional API."""

from __future__ import annotations

import numpy as np
import pytest

from qufin.options.classical.black_scholes import (
    call_price,
    delta,
    gamma,
    implied_volatility,
    price_and_greeks,
    put_price,
    rho,
    theta,
    vega,
)


class TestPricing:
    def test_atm_call(self) -> None:
        p = call_price(s=100, k=100, r=0.05, sigma=0.2, T=1.0)
        assert abs(p - 10.4506) < 0.01

    def test_atm_put(self) -> None:
        p = put_price(s=100, k=100, r=0.05, sigma=0.2, T=1.0)
        # Put-call parity: P = C - S + K*exp(-rT)
        c = call_price(s=100, k=100, r=0.05, sigma=0.2, T=1.0)
        expected = c - 100 + 100 * np.exp(-0.05)
        assert abs(p - expected) < 1e-10

    def test_put_call_parity(self) -> None:
        s, k, r, sigma, T = 110, 105, 0.03, 0.25, 0.5
        c = call_price(s, k, r, sigma, T)
        p = put_price(s, k, r, sigma, T)
        assert abs((c - p) - (s - k * np.exp(-r * T))) < 1e-10

    def test_with_dividend_yield(self) -> None:
        c = call_price(s=100, k=100, r=0.05, sigma=0.2, T=1.0, q=0.02)
        c_no_q = call_price(s=100, k=100, r=0.05, sigma=0.2, T=1.0, q=0.0)
        assert c < c_no_q  # dividend reduces call value

    def test_deep_otm_put_near_zero(self) -> None:
        p = put_price(s=200, k=100, r=0.05, sigma=0.2, T=0.5)
        assert p < 0.01

    def test_call_price_positive(self) -> None:
        assert call_price(s=100, k=100, r=0.05, sigma=0.2, T=1.0) > 0


class TestGreeks:
    def test_call_delta_range(self) -> None:
        d = delta(s=100, k=100, r=0.05, sigma=0.2, T=1.0, option_type="call")
        assert 0 < d < 1

    def test_put_delta_range(self) -> None:
        d = delta(s=100, k=100, r=0.05, sigma=0.2, T=1.0, option_type="put")
        assert -1 < d < 0

    def test_call_put_delta_relation(self) -> None:
        dc = delta(s=100, k=100, r=0.05, sigma=0.2, T=1.0, option_type="call")
        dp = delta(s=100, k=100, r=0.05, sigma=0.2, T=1.0, option_type="put")
        # Delta_call - Delta_put = exp(-qT) = 1 (q=0)
        assert abs(dc - dp - 1.0) < 1e-10

    def test_gamma_positive(self) -> None:
        g = gamma(s=100, k=100, r=0.05, sigma=0.2, T=1.0)
        assert g > 0

    def test_vega_positive(self) -> None:
        v = vega(s=100, k=100, r=0.05, sigma=0.2, T=1.0)
        assert v > 0

    def test_call_rho_positive(self) -> None:
        r_val = rho(s=100, k=100, r=0.05, sigma=0.2, T=1.0, option_type="call")
        assert r_val > 0

    def test_put_rho_negative(self) -> None:
        r_val = rho(s=100, k=100, r=0.05, sigma=0.2, T=1.0, option_type="put")
        assert r_val < 0


class TestImpliedVol:
    def test_roundtrip(self) -> None:
        sigma_true = 0.25
        price = call_price(s=100, k=100, r=0.05, sigma=sigma_true, T=1.0)
        sigma_impl = implied_volatility(price, s=100, k=100, r=0.05, T=1.0)
        assert abs(sigma_impl - sigma_true) < 1e-6

    def test_put_roundtrip(self) -> None:
        sigma_true = 0.30
        price = put_price(s=100, k=110, r=0.05, sigma=sigma_true, T=0.5)
        sigma_impl = implied_volatility(
            price, s=100, k=110, r=0.05, T=0.5, option_type="put"
        )
        assert abs(sigma_impl - sigma_true) < 1e-6


class TestPriceAndGreeks:
    def test_returns_all_fields(self) -> None:
        res = price_and_greeks(s=100, k=100, r=0.05, sigma=0.2, T=1.0)
        assert res.price > 0
        assert 0 < res.delta < 1
        assert res.gamma > 0
        assert res.vega > 0
        assert res.option_type == "call"
