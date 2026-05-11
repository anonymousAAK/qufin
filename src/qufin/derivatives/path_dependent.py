"""Path-dependent derivatives: floating-strike lookback and cliquet options.

Pricing is done via Monte Carlo simulation of GBM paths.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LookbackOptionSpec:
    """Floating-strike lookback option specification.

    Parameters
    ----------
    s0 : float
        Spot price.
    r : float
        Risk-free rate (annualised).
    sigma : float
        Volatility (annualised).
    T : float
        Time to expiry in years.
    is_call : bool
        True for call, False for put.
    n_steps : int
        Number of monitoring points for the path extremum.
    """

    s0: float
    r: float
    sigma: float
    T: float
    is_call: bool = True
    n_steps: int = 252


def _simulate_paths(
    s0: float,
    r: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return GBM paths of shape ``(n_paths, n_steps + 1)``."""
    dt = T / n_steps
    z = rng.standard_normal((n_paths, n_steps))
    increments = (r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z
    log_s = np.zeros((n_paths, n_steps + 1))
    log_s[:, 0] = np.log(s0)
    log_s[:, 1:] = np.log(s0) + np.cumsum(increments, axis=1)
    return np.exp(log_s)


def lookback_mc(
    spec: LookbackOptionSpec,
    n_paths: int = 100_000,
    seed: int | None = 42,
) -> float:
    """Price a floating-strike lookback option via Monte Carlo.

    For a call the payoff is ``S_T - min(S_t)``; for a put it is
    ``max(S_t) - S_T``.

    Parameters
    ----------
    spec : LookbackOptionSpec
        Option specification.
    n_paths : int
        Number of simulation paths.
    seed : int | None
        Random seed.

    Returns
    -------
    float
        Discounted Monte Carlo price.
    """
    rng = np.random.default_rng(seed)
    paths = _simulate_paths(spec.s0, spec.r, spec.sigma, spec.T, spec.n_steps, n_paths, rng)

    s_T = paths[:, -1]

    payoffs = s_T - np.min(paths, axis=1) if spec.is_call else np.max(paths, axis=1) - s_T

    discount = np.exp(-spec.r * spec.T)
    return float(discount * np.mean(payoffs))


def cliquet_mc(
    s0: float,
    r: float,
    sigma: float,
    T: float,
    n_periods: int = 12,
    cap: float = 0.05,
    floor: float = -0.05,
    n_paths: int = 100_000,
    seed: int | None = 42,
) -> float:
    """Price a cliquet (ratchet) option via Monte Carlo.

    The cliquet pays the sum of capped-and-floored periodic returns
    over ``n_periods`` equal sub-periods spanning ``[0, T]``.

    Parameters
    ----------
    s0 : float
        Spot price (used only for GBM drift/vol scaling).
    r : float
        Risk-free rate.
    sigma : float
        Volatility.
    T : float
        Total tenor in years.
    n_periods : int
        Number of reset periods.
    cap : float
        Maximum return per period.
    floor : float
        Minimum return per period.
    n_paths : int
        Number of Monte Carlo paths.
    seed : int | None
        Random seed.

    Returns
    -------
    float
        Discounted price of the cliquet.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_periods
    z = rng.standard_normal((n_paths, n_periods))

    # Periodic returns under GBM
    drift = (r - 0.5 * sigma ** 2) * dt
    diffusion = sigma * np.sqrt(dt) * z
    log_returns = drift + diffusion
    period_returns = np.exp(log_returns) - 1.0

    # Apply cap and floor
    capped = np.clip(period_returns, floor, cap)

    # Payoff is the sum of capped returns (floored at zero for the full sum)
    total_return = np.sum(capped, axis=1)
    payoffs = np.maximum(total_return, 0.0)

    discount = np.exp(-r * T)
    return float(discount * np.mean(payoffs) * s0)
