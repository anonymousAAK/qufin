"""Classical VaR and Expected Shortfall implementations.

Provides historical, parametric (Gaussian), and Monte Carlo
Value-at-Risk (VaR) and Expected Shortfall (CVaR / ES).

References
----------
Jorion, "Value at Risk", 3rd ed., McGraw-Hill (2006).
McNeil, Frey, Embrechts, "Quantitative Risk Management", Princeton (2015).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class VaRResult:
    """Result from VaR / ES computation."""

    var: float
    expected_shortfall: float
    confidence_level: float
    method: str
    n_obs: int = 0
    portfolio_value: float = 1.0

    @property
    def var_dollar(self) -> float:
        """VaR in dollar terms."""
        return self.var * self.portfolio_value

    @property
    def es_dollar(self) -> float:
        """Expected shortfall in dollar terms."""
        return self.expected_shortfall * self.portfolio_value


def historical_var(
    returns: NDArray[np.float64],
    confidence: float = 0.95,
    portfolio_value: float = 1.0,
) -> VaRResult:
    """Historical simulation VaR and Expected Shortfall.

    Uses the empirical distribution of past returns directly.

    Parameters
    ----------
    returns : NDArray
        Historical portfolio returns, shape (T,) or (T, N) with weights summed.
    confidence : float
        Confidence level, e.g. 0.95 for 95% VaR.
    portfolio_value : float
        Portfolio notional for dollar VaR.
    """
    r = np.asarray(returns).flatten()
    alpha = 1 - confidence
    var = -float(np.percentile(r, 100 * alpha))
    tail = r[r <= -var]
    es = -float(np.mean(tail)) if len(tail) > 0 else var

    return VaRResult(
        var=var,
        expected_shortfall=es,
        confidence_level=confidence,
        method="historical",
        n_obs=len(r),
        portfolio_value=portfolio_value,
    )


def parametric_var(
    returns: NDArray[np.float64],
    confidence: float = 0.95,
    portfolio_value: float = 1.0,
) -> VaRResult:
    """Parametric (Gaussian) VaR and Expected Shortfall.

    Assumes returns are normally distributed.

    Parameters
    ----------
    returns : NDArray
        Historical portfolio returns, shape (T,).
    confidence : float
        Confidence level.
    portfolio_value : float
        Portfolio notional.
    """
    from scipy.stats import norm

    r = np.asarray(returns).flatten()
    mu = float(np.mean(r))
    sigma = float(np.std(r, ddof=1))

    alpha = 1 - confidence
    z = norm.ppf(1 - alpha)

    var = -(mu - z * sigma)

    # ES for normal: ES = mu + sigma * phi(z) / (1 - alpha)
    es = -(mu - sigma * norm.pdf(z) / alpha)

    return VaRResult(
        var=var,
        expected_shortfall=es,
        confidence_level=confidence,
        method="parametric_gaussian",
        n_obs=len(r),
        portfolio_value=portfolio_value,
    )


def monte_carlo_var(
    returns: NDArray[np.float64],
    confidence: float = 0.95,
    n_simulations: int = 100_000,
    horizon: int = 1,
    portfolio_value: float = 1.0,
    seed: int | None = 42,
) -> VaRResult:
    """Monte Carlo VaR and Expected Shortfall.

    Simulates future returns from a fitted multivariate normal.

    Parameters
    ----------
    returns : NDArray
        Historical returns, shape (T,) for single asset or (T, N).
    confidence : float
        Confidence level.
    n_simulations : int
        Number of simulated scenarios.
    horizon : int
        Holding period in same units as returns.
    portfolio_value : float
        Portfolio notional.
    seed : int or None
        Random seed.
    """
    rng = np.random.default_rng(seed)
    r = np.asarray(returns)

    if r.ndim == 1:
        mu = float(np.mean(r))
        sigma = float(np.std(r, ddof=1))
        sim_returns = rng.normal(mu * horizon, sigma * np.sqrt(horizon), n_simulations)
    else:
        mu = np.mean(r, axis=0)
        cov = np.cov(r, rowvar=False)
        # Equal-weight portfolio if no weights given
        n_assets = r.shape[1]
        w = np.ones(n_assets) / n_assets
        port_mu = float(w @ mu) * horizon
        port_var = float(w @ cov @ w) * horizon
        sim_returns = rng.normal(port_mu, np.sqrt(port_var), n_simulations)

    alpha = 1 - confidence
    var = -float(np.percentile(sim_returns, 100 * alpha))
    tail = sim_returns[sim_returns <= -var]
    es = -float(np.mean(tail)) if len(tail) > 0 else var

    return VaRResult(
        var=var,
        expected_shortfall=es,
        confidence_level=confidence,
        method="monte_carlo",
        n_obs=n_simulations,
        portfolio_value=portfolio_value,
    )


def portfolio_var(
    returns: NDArray[np.float64],
    weights: NDArray[np.float64],
    confidence: float = 0.95,
    method: str = "historical",
    portfolio_value: float = 1.0,
    **kwargs: Any,
) -> VaRResult:
    """Compute portfolio VaR given asset returns and weights.

    Parameters
    ----------
    returns : NDArray
        Asset returns, shape (T, N).
    weights : NDArray
        Portfolio weights, shape (N,).
    confidence : float
        Confidence level.
    method : str
        One of "historical", "parametric", "monte_carlo".
    portfolio_value : float
        Portfolio notional.
    """
    r = np.asarray(returns)
    w = np.asarray(weights)
    portfolio_returns = r @ w

    if method == "historical":
        return historical_var(portfolio_returns, confidence, portfolio_value)
    elif method == "parametric":
        return parametric_var(portfolio_returns, confidence, portfolio_value)
    elif method == "monte_carlo":
        return monte_carlo_var(
            portfolio_returns, confidence, portfolio_value=portfolio_value, **kwargs
        )
    else:
        msg = f"Unknown method: {method}. Use 'historical', 'parametric', or 'monte_carlo'."
        raise ValueError(msg)
