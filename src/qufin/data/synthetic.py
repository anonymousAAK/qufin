"""Synthetic data generators: GBM, Heston, Merton jump-diffusion."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def gbm_paths(
    s0: float,
    mu: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int | None = None,
) -> NDArray[np.float64]:
    """Generate Geometric Brownian Motion sample paths.

    Parameters
    ----------
    s0 : float
        Initial price.
    mu : float
        Drift (annualized).
    sigma : float
        Volatility (annualized).
    T : float
        Time horizon in years.
    n_steps : int
        Number of time steps.
    n_paths : int
        Number of simulated paths.
    seed : int | None
        Random seed.

    Returns
    -------
    NDArray of shape (n_paths, n_steps + 1)
        Simulated price paths including the initial price.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    paths = np.zeros((n_paths, n_steps + 1), dtype=np.float64)
    paths[:, 0] = s0

    z = rng.standard_normal((n_paths, n_steps))
    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt) * z

    log_returns = drift + diffusion
    paths[:, 1:] = s0 * np.exp(np.cumsum(log_returns, axis=1))
    return paths


def heston_paths(
    s0: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    mu: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Generate Heston stochastic-volatility model paths.

    Uses Euler-Maruyama discretization with full truncation for the
    variance process to avoid negative variances.

    Parameters
    ----------
    s0, v0 : float
        Initial price and variance.
    kappa : float
        Mean-reversion speed.
    theta : float
        Long-run variance.
    xi : float
        Volatility of variance (vol-of-vol).
    rho : float
        Correlation between price and variance Brownian motions.
    mu : float
        Drift.
    T : float
        Time horizon.
    n_steps, n_paths : int
        Discretization and sample size.
    seed : int | None
        Random seed.

    Returns
    -------
    (prices, variances) : tuple of NDArray, each shape (n_paths, n_steps + 1)
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps

    prices = np.zeros((n_paths, n_steps + 1), dtype=np.float64)
    variances = np.zeros((n_paths, n_steps + 1), dtype=np.float64)
    prices[:, 0] = s0
    variances[:, 0] = v0

    z1 = rng.standard_normal((n_paths, n_steps))
    z2 = rng.standard_normal((n_paths, n_steps))
    w1 = z1
    w2 = rho * z1 + np.sqrt(1 - rho**2) * z2

    for t in range(n_steps):
        v = np.maximum(variances[:, t], 0.0)
        sqrt_v = np.sqrt(v)

        # Euler-Maruyama for variance with full truncation
        variances[:, t + 1] = np.maximum(
            v + kappa * (theta - v) * dt + xi * sqrt_v * np.sqrt(dt) * w2[:, t],
            0.0,
        )

        # Log-Euler for price
        prices[:, t + 1] = prices[:, t] * np.exp(
            (mu - 0.5 * v) * dt + sqrt_v * np.sqrt(dt) * w1[:, t]
        )

    return prices, variances


def merton_jump_paths(
    s0: float,
    mu: float,
    sigma: float,
    lam: float,
    jump_mean: float,
    jump_std: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int | None = None,
) -> NDArray[np.float64]:
    """Generate Merton jump-diffusion model paths.

    Parameters
    ----------
    s0 : float
        Initial price.
    mu, sigma : float
        Drift and volatility of the diffusion component.
    lam : float
        Jump intensity (expected jumps per year).
    jump_mean, jump_std : float
        Mean and std of log-jump size (normal distribution).
    T : float
        Time horizon.
    n_steps, n_paths : int
        Discretization and sample size.
    seed : int | None
        Random seed.

    Returns
    -------
    NDArray of shape (n_paths, n_steps + 1)
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps

    paths = np.zeros((n_paths, n_steps + 1), dtype=np.float64)
    paths[:, 0] = s0

    for t in range(n_steps):
        z = rng.standard_normal(n_paths)
        n_jumps = rng.poisson(lam * dt, n_paths)
        jump_sizes = np.array(
            [rng.normal(jump_mean, jump_std, n).sum() if n > 0 else 0.0 for n in n_jumps]
        )

        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt) * z
        paths[:, t + 1] = paths[:, t] * np.exp(drift + diffusion + jump_sizes)

    return paths
