"""Property-based tests for Black-Scholes using Hypothesis."""

from __future__ import annotations

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from qufin.options.classical.black_scholes import call_price, put_price, vega


# Strategy for option parameters
spot = st.floats(min_value=10, max_value=500)
strike = st.floats(min_value=10, max_value=500)
rate = st.floats(min_value=0.0, max_value=0.15)
vol = st.floats(min_value=0.05, max_value=1.5)
expiry = st.floats(min_value=0.01, max_value=5.0)


@given(s=spot, k=strike, r=rate, sigma=vol, T=expiry)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_put_call_parity(s: float, k: float, r: float, sigma: float, T: float) -> None:
    """C - P = S - K * exp(-rT) must always hold."""
    c = call_price(s, k, r, sigma, T)
    p = put_price(s, k, r, sigma, T)
    rhs = s - k * np.exp(-r * T)
    assert abs((c - p) - rhs) < 1e-6, f"Parity violated: C-P={c-p}, S-Ke^(-rT)={rhs}"


@given(s=spot, k=strike, r=rate, sigma=vol, T=expiry)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_call_price_non_negative(s: float, k: float, r: float, sigma: float, T: float) -> None:
    """Call price is always non-negative."""
    assert call_price(s, k, r, sigma, T) >= -1e-10


@given(s=spot, k=strike, r=rate, sigma=vol, T=expiry)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_put_price_non_negative(s: float, k: float, r: float, sigma: float, T: float) -> None:
    """Put price is always non-negative."""
    assert put_price(s, k, r, sigma, T) >= -1e-10


@given(s=spot, k=strike, r=rate, T=expiry)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_call_monotone_in_volatility(s: float, k: float, r: float, T: float) -> None:
    """Call price is monotonically increasing in volatility."""
    c_low = call_price(s, k, r, 0.1, T)
    c_high = call_price(s, k, r, 0.5, T)
    assert c_high >= c_low - 1e-10


@given(k=strike, r=rate, sigma=vol, T=expiry)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_call_monotone_in_spot(k: float, r: float, sigma: float, T: float) -> None:
    """Call price is monotonically increasing in spot price."""
    c_low = call_price(50.0, k, r, sigma, T)
    c_high = call_price(200.0, k, r, sigma, T)
    assert c_high >= c_low - 1e-10


@given(s=spot, k=strike, r=rate, sigma=vol, T=expiry)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_vega_non_negative(s: float, k: float, r: float, sigma: float, T: float) -> None:
    """Vega is always non-negative for vanilla options."""
    assert vega(s, k, r, sigma, T) >= -1e-10
