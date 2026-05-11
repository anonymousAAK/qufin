"""Classical option pricing baselines."""

from __future__ import annotations

from qufin.options.classical import black_scholes, monte_carlo
from qufin.options.classical.binomial import crr_tree
from qufin.options.classical.black_scholes import (
    BSResult,
    call_price,
    implied_volatility,
    price_and_greeks,
    put_price,
)
from qufin.options.classical.monte_carlo import asian_mc, barrier_mc, european_mc

__all__ = [
    "BSResult",
    "asian_mc",
    "barrier_mc",
    "black_scholes",
    "call_price",
    "crr_tree",
    "european_mc",
    "implied_volatility",
    "monte_carlo",
    "price_and_greeks",
    "put_price",
]
