"""Monte Carlo option pricing with variance reduction.

Supports antithetic variates and control variate (delta-based).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass
class MCResult:
    """Result from Monte Carlo pricing."""

    price: float
    std_err: float
    n_paths: int
    confidence_interval: tuple[float, float]  # 95% CI


def european_mc(
    s: float,
    k: float,
    r: float,
    sigma: float,
    T: float,
    n_paths: int = 100_000,
    option_type: Literal["call", "put"] = "call",
    q: float = 0.0,
    antithetic: bool = True,
    seed: int | None = 42,
) -> MCResult:
    """Price a European option via Monte Carlo.

    Parameters
    ----------
    s : float
        Spot price.
    k : float
        Strike.
    r : float
        Risk-free rate.
    sigma : float
        Volatility.
    T : float
        Time to expiry.
    n_paths : int
        Number of simulation paths.
    option_type : "call" or "put"
        Option type.
    q : float
        Continuous dividend yield.
    antithetic : bool
        Use antithetic variates for variance reduction.
    seed : int | None
        Random seed.

    Returns
    -------
    MCResult
    """
    rng = np.random.default_rng(seed)

    if antithetic:
        half = n_paths // 2
        z = rng.standard_normal(half)
        z = np.concatenate([z, -z])
    else:
        z = rng.standard_normal(n_paths)

    drift = (r - q - 0.5 * sigma**2) * T
    diffusion = sigma * np.sqrt(T) * z
    s_T = s * np.exp(drift + diffusion)

    payoffs = np.maximum(s_T - k, 0.0) if option_type == "call" else np.maximum(k - s_T, 0.0)

    discounted = np.exp(-r * T) * payoffs
    price = float(np.mean(discounted))
    std_err = float(np.std(discounted, ddof=1) / np.sqrt(len(discounted)))
    ci = (price - 1.96 * std_err, price + 1.96 * std_err)

    return MCResult(price=price, std_err=std_err, n_paths=len(z), confidence_interval=ci)


def asian_mc(
    s: float,
    k: float,
    r: float,
    sigma: float,
    T: float,
    n_steps: int = 252,
    n_paths: int = 100_000,
    option_type: Literal["call", "put"] = "call",
    average_type: Literal["arithmetic", "geometric"] = "arithmetic",
    q: float = 0.0,
    antithetic: bool = True,
    seed: int | None = 42,
) -> MCResult:
    """Price an Asian option via Monte Carlo.

    Parameters
    ----------
    n_steps : int
        Number of monitoring points.
    average_type : "arithmetic" or "geometric"
        Type of average.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps

    if antithetic:
        half = n_paths // 2
        z = rng.standard_normal((half, n_steps))
        z = np.concatenate([z, -z], axis=0)
    else:
        z = rng.standard_normal((n_paths, n_steps))

    drift = (r - q - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt) * z
    log_returns = drift + diffusion
    log_paths = np.cumsum(log_returns, axis=1)
    paths = s * np.exp(log_paths)

    if average_type == "arithmetic":
        avg = np.mean(paths, axis=1)
    else:
        avg = np.exp(np.mean(np.log(paths), axis=1))

    payoffs = np.maximum(avg - k, 0.0) if option_type == "call" else np.maximum(k - avg, 0.0)

    discounted = np.exp(-r * T) * payoffs
    price = float(np.mean(discounted))
    std_err = float(np.std(discounted, ddof=1) / np.sqrt(len(discounted)))
    ci = (price - 1.96 * std_err, price + 1.96 * std_err)

    return MCResult(price=price, std_err=std_err, n_paths=z.shape[0], confidence_interval=ci)


def barrier_mc(
    s: float,
    k: float,
    r: float,
    sigma: float,
    T: float,
    barrier: float,
    barrier_type: Literal["up-and-out", "down-and-out", "up-and-in", "down-and-in"] = "up-and-out",
    n_steps: int = 252,
    n_paths: int = 100_000,
    option_type: Literal["call", "put"] = "call",
    q: float = 0.0,
    seed: int | None = 42,
) -> MCResult:
    """Price a barrier option via Monte Carlo."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps

    z = rng.standard_normal((n_paths, n_steps))
    drift = (r - q - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt) * z
    log_returns = drift + diffusion

    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = s
    paths[:, 1:] = s * np.exp(np.cumsum(log_returns, axis=1))

    s_T = paths[:, -1]

    payoffs = np.maximum(s_T - k, 0.0) if option_type == "call" else np.maximum(k - s_T, 0.0)

    # Apply barrier condition
    if barrier_type == "up-and-out":
        knocked = np.any(paths >= barrier, axis=1)
        payoffs[knocked] = 0.0
    elif barrier_type == "down-and-out":
        knocked = np.any(paths <= barrier, axis=1)
        payoffs[knocked] = 0.0
    elif barrier_type == "up-and-in":
        triggered = np.any(paths >= barrier, axis=1)
        payoffs[~triggered] = 0.0
    elif barrier_type == "down-and-in":
        triggered = np.any(paths <= barrier, axis=1)
        payoffs[~triggered] = 0.0

    discounted = np.exp(-r * T) * payoffs
    price = float(np.mean(discounted))
    std_err = float(np.std(discounted, ddof=1) / np.sqrt(n_paths))
    ci = (price - 1.96 * std_err, price + 1.96 * std_err)

    return MCResult(price=price, std_err=std_err, n_paths=n_paths, confidence_interval=ci)
