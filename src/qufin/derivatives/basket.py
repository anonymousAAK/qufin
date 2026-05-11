"""Multi-asset basket option pricing.

Classical Monte Carlo + QAE-ready problem construction for
basket options on correlated assets.

References
----------
Kashif et al., arXiv:2509.09432 — quantum basket option pricing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class BasketOptionSpec:
    """Basket option specification.

    Parameters
    ----------
    s0 : NDArray
        Initial prices, shape (n_assets,).
    k : float
        Strike price.
    r : float
        Risk-free rate.
    sigma : NDArray
        Volatilities, shape (n_assets,).
    corr : NDArray
        Correlation matrix, shape (n_assets, n_assets).
    T : float
        Time to expiry.
    weights : NDArray | None
        Asset weights in the basket (default: equal).
    is_call : bool
        True for call, False for put.
    """

    s0: NDArray[np.float64]
    k: float
    r: float
    sigma: NDArray[np.float64]
    corr: NDArray[np.float64]
    T: float
    weights: NDArray[np.float64] | None = None
    is_call: bool = True

    @property
    def n_assets(self) -> int:
        return len(self.s0)

    @property
    def basket_weights(self) -> NDArray[np.float64]:
        if self.weights is not None:
            return self.weights
        return np.ones(self.n_assets) / self.n_assets


@dataclass
class BasketResult:
    """Result from basket option pricing."""

    price: float
    std_err: float
    n_paths: int
    confidence_interval: tuple[float, float]


def basket_mc(
    spec: BasketOptionSpec,
    n_paths: int = 100_000,
    antithetic: bool = True,
    seed: int | None = 42,
) -> BasketResult:
    """Price a basket option via Monte Carlo.

    Simulates correlated GBM paths using Cholesky decomposition.

    Parameters
    ----------
    spec : BasketOptionSpec
        Option specification.
    n_paths : int
        Number of simulation paths.
    antithetic : bool
        Use antithetic variates.
    seed : int | None
        Random seed.

    Returns
    -------
    BasketResult
    """
    rng = np.random.default_rng(seed)
    n = spec.n_assets
    w = spec.basket_weights

    # Cholesky decomposition of correlation matrix
    L = np.linalg.cholesky(spec.corr)

    if antithetic:
        half = n_paths // 2
        z = rng.standard_normal((half, n))
        z = np.concatenate([z, -z], axis=0)
    else:
        z = rng.standard_normal((n_paths, n))

    # Correlated normals
    z_corr = z @ L.T

    # Terminal prices for each asset
    s_T = np.zeros((len(z_corr), n))
    for i in range(n):
        drift = (spec.r - 0.5 * spec.sigma[i] ** 2) * spec.T
        diffusion = spec.sigma[i] * np.sqrt(spec.T) * z_corr[:, i]
        s_T[:, i] = spec.s0[i] * np.exp(drift + diffusion)

    # Basket value at expiry: weighted sum
    basket_value = s_T @ w

    # Payoff
    if spec.is_call:
        payoffs = np.maximum(basket_value - spec.k, 0)
    else:
        payoffs = np.maximum(spec.k - basket_value, 0)

    discounted = np.exp(-spec.r * spec.T) * payoffs
    price = float(np.mean(discounted))
    std_err = float(np.std(discounted, ddof=1) / np.sqrt(len(discounted)))
    ci = (price - 1.96 * std_err, price + 1.96 * std_err)

    return BasketResult(
        price=price,
        std_err=std_err,
        n_paths=len(z_corr),
        confidence_interval=ci,
    )


def geometric_basket_closed_form(
    spec: BasketOptionSpec,
) -> float:
    """Closed-form price for geometric basket option.

    The geometric basket follows a log-normal distribution.

    Returns
    -------
    float
        Option price.
    """
    from scipy.stats import norm as norm_dist

    n = spec.n_assets
    w = spec.basket_weights

    # Geometric basket: B_T = prod(S_i^{w_i})
    # log(B_T) ~ Normal with:
    mu_g = np.sum(w * (np.log(spec.s0) + (spec.r - 0.5 * spec.sigma**2) * spec.T))
    var_g = 0.0
    for i in range(n):
        for j in range(n):
            var_g += w[i] * w[j] * spec.sigma[i] * spec.sigma[j] * spec.corr[i, j] * spec.T

    sigma_g = np.sqrt(var_g)

    # BS-like formula for log-normal
    F = np.exp(mu_g + 0.5 * var_g)  # forward
    d1 = (np.log(F / spec.k) + 0.5 * var_g) / sigma_g
    d2 = d1 - sigma_g

    discount = np.exp(-spec.r * spec.T)
    if spec.is_call:
        price = discount * (F * norm_dist.cdf(d1) - spec.k * norm_dist.cdf(d2))
    else:
        price = discount * (spec.k * norm_dist.cdf(-d2) - F * norm_dist.cdf(-d1))

    return float(price)
