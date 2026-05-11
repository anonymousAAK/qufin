"""Tests for European option pricing."""

from __future__ import annotations

import numpy as np

from qufin.options.european import EuropeanOption


class TestBlackScholes:
    """Test BS pricing against known reference values."""

    def test_atm_call(self) -> None:
        opt = EuropeanOption(s0=100, k=100, r=0.05, sigma=0.2, T=1.0, is_call=True)
        price = opt.bs_price()
        # Known BS value for ATM call: ~10.4506
        assert abs(price - 10.4506) < 0.01

    def test_put_call_parity(self) -> None:
        s0, k, r, sigma, T = 100.0, 105.0, 0.05, 0.2, 1.0
        call = EuropeanOption(s0=s0, k=k, r=r, sigma=sigma, T=T, is_call=True)
        put = EuropeanOption(s0=s0, k=k, r=r, sigma=sigma, T=T, is_call=False)
        # C - P = S - K*exp(-rT)
        lhs = call.bs_price() - put.bs_price()
        rhs = s0 - k * np.exp(-r * T)
        assert abs(lhs - rhs) < 1e-10

    def test_deep_itm_call(self) -> None:
        opt = EuropeanOption(s0=200, k=100, r=0.05, sigma=0.2, T=1.0, is_call=True)
        price = opt.bs_price()
        intrinsic = 200 - 100 * np.exp(-0.05)
        assert price > intrinsic

    def test_deep_otm_call(self) -> None:
        opt = EuropeanOption(s0=50, k=100, r=0.05, sigma=0.2, T=1.0, is_call=True)
        price = opt.bs_price()
        assert price > 0
        assert price < 1.0  # very small

    def test_delta_bounds(self) -> None:
        opt = EuropeanOption(s0=100, k=100, r=0.05, sigma=0.2, T=1.0, is_call=True)
        assert 0 <= opt.bs_delta() <= 1

    def test_gamma_positive(self) -> None:
        opt = EuropeanOption(s0=100, k=100, r=0.05, sigma=0.2, T=1.0, is_call=True)
        assert opt.bs_gamma() > 0

    def test_vega_positive(self) -> None:
        opt = EuropeanOption(s0=100, k=100, r=0.05, sigma=0.2, T=1.0, is_call=True)
        assert opt.bs_vega() > 0
