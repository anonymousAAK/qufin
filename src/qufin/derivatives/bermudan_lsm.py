"""Bermudan option pricing via Longstaff-Schwartz Monte Carlo.

Implements backward induction with polynomial regression (degree 3)
to estimate the continuation value at each exercise date.

References
----------
Longstaff & Schwartz (2001), *Valuing American Options by Simulation*.
"""

from __future__ import annotations

import numpy as np


def _simulate_gbm_paths(
    s0: float,
    r: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate GBM paths.

    Returns
    -------
    np.ndarray
        Shape ``(n_paths, n_steps + 1)`` with ``[:, 0] == s0``.
    """
    dt = T / n_steps
    z = rng.standard_normal((n_paths, n_steps))
    drift = (r - 0.5 * sigma ** 2) * dt
    diffusion = sigma * np.sqrt(dt) * z
    log_returns = drift + diffusion
    log_s = np.zeros((n_paths, n_steps + 1))
    log_s[:, 0] = np.log(s0)
    log_s[:, 1:] = np.log(s0) + np.cumsum(log_returns, axis=1)
    return np.exp(log_s)


def _intrinsic(s: np.ndarray, k: float, is_call: bool) -> np.ndarray:
    if is_call:
        return np.maximum(s - k, 0.0)
    return np.maximum(k - s, 0.0)


def lsm_price(
    s0: float,
    k: float,
    r: float,
    sigma: float,
    T: float,
    n_steps: int = 100,
    n_paths: int = 50_000,
    exercise_dates: list[float] | None = None,
    is_call: bool = False,
    seed: int | None = 42,
) -> dict[str, object]:
    """Price a Bermudan option with Longstaff-Schwartz Monte Carlo.

    Parameters
    ----------
    s0, k, r, sigma, T : float
        Standard option parameters.
    n_steps : int
        Number of time-steps in the GBM discretisation.
    n_paths : int
        Number of Monte Carlo paths.
    exercise_dates : list[float] | None
        Times (in years) when early exercise is allowed.
        If *None*, every step is an exercise opportunity (American).
    is_call : bool
        True for call, False for put.
    seed : int | None
        Random seed.

    Returns
    -------
    dict
        ``price``, ``std_err``, ``optimal_exercise_times``.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    paths = _simulate_gbm_paths(s0, r, sigma, T, n_steps, n_paths, rng)

    # Determine exercise step indices
    if exercise_dates is None:
        ex_steps = list(range(1, n_steps + 1))
    else:
        ex_steps = sorted({max(1, min(round(t / dt), n_steps)) for t in exercise_dates})

    # Cash-flow matrix: when each path is exercised and the payoff received
    cashflow = np.zeros(n_paths)
    exercise_time = np.full(n_paths, np.nan)

    # At the last exercise step, exercise if in the money
    last = ex_steps[-1]
    cashflow[:] = _intrinsic(paths[:, last], k, is_call)
    exercise_time[:] = last * dt

    # Backward induction
    for step in reversed(ex_steps[:-1]):
        t_step = step * dt
        s_step = paths[:, step]
        intrinsic = _intrinsic(s_step, k, is_call)
        itm = intrinsic > 0

        if np.sum(itm) < 5:
            # Too few in-the-money paths for regression; skip
            continue

        # Discount existing cash-flows back to this step
        time_ahead = exercise_time[itm] - t_step
        discounted_cf = cashflow[itm] * np.exp(-r * time_ahead)

        # Polynomial regression on in-the-money paths
        x = s_step[itm]
        x_norm = (x - np.mean(x)) / (np.std(x) + 1e-12)
        poly = np.column_stack([x_norm ** p for p in range(4)])  # degree 3
        coeffs, *_ = np.linalg.lstsq(poly, discounted_cf, rcond=None)
        continuation = poly @ coeffs

        # Exercise where intrinsic > estimated continuation
        exercise_mask = intrinsic[itm] > continuation
        itm_indices = np.where(itm)[0][exercise_mask]
        cashflow[itm_indices] = intrinsic[itm_indices]
        exercise_time[itm_indices] = t_step

    # Discount all cash-flows to t=0
    discount_factors = np.exp(-r * exercise_time)
    present_values = cashflow * discount_factors

    price = float(np.mean(present_values))
    std_err = float(np.std(present_values, ddof=1) / np.sqrt(n_paths))

    # Collect optimal exercise time distribution (ignoring zero-payoff paths)
    exercised = ~np.isnan(exercise_time) & (cashflow > 0)
    optimal_times = exercise_time[exercised].tolist() if np.any(exercised) else []

    return {
        "price": price,
        "std_err": std_err,
        "optimal_exercise_times": optimal_times,
    }
