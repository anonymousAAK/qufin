"""European option pricing: BS closed-form + QAE interface."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass
class EuropeanOption:
    """European option specification.

    Parameters
    ----------
    s0 : float
        Spot price.
    k : float
        Strike price.
    r : float
        Risk-free rate.
    sigma : float
        Volatility.
    T : float
        Time to expiry in years.
    is_call : bool
        True for call, False for put.
    """

    s0: float
    k: float
    r: float
    sigma: float
    T: float
    is_call: bool = True

    def bs_price(self) -> float:
        """Black-Scholes closed-form price."""
        d1 = (np.log(self.s0 / self.k) + (self.r + 0.5 * self.sigma**2) * self.T) / (
            self.sigma * np.sqrt(self.T)
        )
        d2 = d1 - self.sigma * np.sqrt(self.T)

        if self.is_call:
            return float(
                self.s0 * norm.cdf(d1) - self.k * np.exp(-self.r * self.T) * norm.cdf(d2)
            )
        else:
            return float(
                self.k * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.s0 * norm.cdf(-d1)
            )

    def bs_delta(self) -> float:
        """Black-Scholes delta."""
        d1 = (np.log(self.s0 / self.k) + (self.r + 0.5 * self.sigma**2) * self.T) / (
            self.sigma * np.sqrt(self.T)
        )
        if self.is_call:
            return float(norm.cdf(d1))
        return float(norm.cdf(d1) - 1)

    def bs_gamma(self) -> float:
        """Black-Scholes gamma."""
        d1 = (np.log(self.s0 / self.k) + (self.r + 0.5 * self.sigma**2) * self.T) / (
            self.sigma * np.sqrt(self.T)
        )
        return float(norm.pdf(d1) / (self.s0 * self.sigma * np.sqrt(self.T)))

    def bs_vega(self) -> float:
        """Black-Scholes vega."""
        d1 = (np.log(self.s0 / self.k) + (self.r + 0.5 * self.sigma**2) * self.T) / (
            self.sigma * np.sqrt(self.T)
        )
        return float(self.s0 * norm.pdf(d1) * np.sqrt(self.T))

    def bs_theta(self) -> float:
        """Black-Scholes theta."""
        d1 = (np.log(self.s0 / self.k) + (self.r + 0.5 * self.sigma**2) * self.T) / (
            self.sigma * np.sqrt(self.T)
        )
        d2 = d1 - self.sigma * np.sqrt(self.T)

        term1 = -(self.s0 * norm.pdf(d1) * self.sigma) / (2 * np.sqrt(self.T))
        if self.is_call:
            return float(term1 - self.r * self.k * np.exp(-self.r * self.T) * norm.cdf(d2))
        return float(term1 + self.r * self.k * np.exp(-self.r * self.T) * norm.cdf(-d2))
