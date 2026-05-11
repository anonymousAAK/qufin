"""Black-Scholes closed-form pricing with full Greeks.

Standalone module providing functional API for BS pricing.
For an OOP interface, see `qufin.options.european.EuropeanOption`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.stats import norm


@dataclass
class BSResult:
    """Full Black-Scholes result with price and Greeks."""

    price: float
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float
    option_type: str


def _d1d2(
    s: float, k: float, r: float, q: float, sigma: float, T: float
) -> tuple[float, float]:
    """Compute d1 and d2."""
    d1 = (np.log(s / k) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def call_price(
    s: float, k: float, r: float, sigma: float, T: float, q: float = 0.0
) -> float:
    """Black-Scholes European call price.

    Parameters
    ----------
    s : float
        Spot price.
    k : float
        Strike price.
    r : float
        Risk-free rate.
    sigma : float
        Volatility.
    T : float
        Time to expiry (years).
    q : float
        Continuous dividend yield.
    """
    d1, d2 = _d1d2(s, k, r, q, sigma, T)
    return float(s * np.exp(-q * T) * norm.cdf(d1) - k * np.exp(-r * T) * norm.cdf(d2))


def put_price(
    s: float, k: float, r: float, sigma: float, T: float, q: float = 0.0
) -> float:
    """Black-Scholes European put price."""
    d1, d2 = _d1d2(s, k, r, q, sigma, T)
    return float(k * np.exp(-r * T) * norm.cdf(-d2) - s * np.exp(-q * T) * norm.cdf(-d1))


def delta(
    s: float, k: float, r: float, sigma: float, T: float,
    option_type: Literal["call", "put"] = "call", q: float = 0.0,
) -> float:
    """Black-Scholes delta."""
    d1, _ = _d1d2(s, k, r, q, sigma, T)
    if option_type == "call":
        return float(np.exp(-q * T) * norm.cdf(d1))
    return float(np.exp(-q * T) * (norm.cdf(d1) - 1))


def gamma(
    s: float, k: float, r: float, sigma: float, T: float, q: float = 0.0
) -> float:
    """Black-Scholes gamma (same for call and put)."""
    d1, _ = _d1d2(s, k, r, q, sigma, T)
    return float(np.exp(-q * T) * norm.pdf(d1) / (s * sigma * np.sqrt(T)))


def vega(
    s: float, k: float, r: float, sigma: float, T: float, q: float = 0.0
) -> float:
    """Black-Scholes vega (same for call and put)."""
    d1, _ = _d1d2(s, k, r, q, sigma, T)
    return float(s * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T))


def theta(
    s: float, k: float, r: float, sigma: float, T: float,
    option_type: Literal["call", "put"] = "call", q: float = 0.0,
) -> float:
    """Black-Scholes theta (per year)."""
    d1, d2 = _d1d2(s, k, r, q, sigma, T)
    term1 = -(s * np.exp(-q * T) * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    if option_type == "call":
        return float(
            term1 + q * s * np.exp(-q * T) * norm.cdf(d1)
            - r * k * np.exp(-r * T) * norm.cdf(d2)
        )
    return float(
        term1 - q * s * np.exp(-q * T) * norm.cdf(-d1)
        + r * k * np.exp(-r * T) * norm.cdf(-d2)
    )


def rho(
    s: float, k: float, r: float, sigma: float, T: float,
    option_type: Literal["call", "put"] = "call", q: float = 0.0,
) -> float:
    """Black-Scholes rho."""
    _, d2 = _d1d2(s, k, r, q, sigma, T)
    if option_type == "call":
        return float(k * T * np.exp(-r * T) * norm.cdf(d2))
    return float(-k * T * np.exp(-r * T) * norm.cdf(-d2))


def implied_volatility(
    market_price: float,
    s: float, k: float, r: float, T: float,
    option_type: Literal["call", "put"] = "call",
    q: float = 0.0,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> float:
    """Implied volatility via Newton-Raphson on BS price.

    Parameters
    ----------
    market_price : float
        Observed market price.
    tol : float
        Convergence tolerance.
    max_iter : int
        Maximum iterations.

    Returns
    -------
    float
        Implied volatility.

    Raises
    ------
    ValueError
        If no convergence.
    """
    price_fn = call_price if option_type == "call" else put_price
    sigma_est = 0.2  # initial guess

    for _ in range(max_iter):
        price = price_fn(s, k, r, sigma_est, T, q)
        v = vega(s, k, r, sigma_est, T, q)
        if v < 1e-12:
            break
        sigma_est -= (price - market_price) / v
        sigma_est = max(sigma_est, 1e-6)
        if abs(price - market_price) < tol:
            return sigma_est

    raise ValueError(f"Implied vol did not converge after {max_iter} iterations")


def price_and_greeks(
    s: float, k: float, r: float, sigma: float, T: float,
    option_type: Literal["call", "put"] = "call", q: float = 0.0,
) -> BSResult:
    """Compute BS price and all Greeks in one call."""
    price_fn = call_price if option_type == "call" else put_price
    return BSResult(
        price=price_fn(s, k, r, sigma, T, q),
        delta=delta(s, k, r, sigma, T, option_type, q),
        gamma=gamma(s, k, r, sigma, T, q),
        vega=vega(s, k, r, sigma, T, q),
        theta=theta(s, k, r, sigma, T, option_type, q),
        rho=rho(s, k, r, sigma, T, option_type, q),
        option_type=option_type,
    )
