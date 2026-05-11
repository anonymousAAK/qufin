"""Bermudan option pricing via CRR binomial tree.

Supports early exercise only at user-specified exercise dates,
interpolated to the nearest tree time-steps.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class BermudanOptionSpec:
    """Bermudan option specification.

    Parameters
    ----------
    s0 : float
        Spot price.
    k : float
        Strike price.
    r : float
        Risk-free rate (annualised).
    sigma : float
        Volatility (annualised).
    T : float
        Time to expiry in years.
    exercise_dates : list[float]
        Times (in years) at which early exercise is permitted.
        Must be in (0, T].
    is_call : bool
        True for call, False for put.
    n_steps : int
        Number of binomial tree time-steps.
    """

    s0: float
    k: float
    r: float
    sigma: float
    T: float
    exercise_dates: list[float] = field(default_factory=list)
    is_call: bool = True
    n_steps: int = 200


@dataclass
class BermudanResult:
    """Result of Bermudan option pricing.

    Attributes
    ----------
    price : float
        Option price at t=0.
    exercise_boundary : dict[int, float]
        Map from tree step index to critical stock price at which
        early exercise is optimal.
    """

    price: float
    exercise_boundary: dict[int, float]


def _payoff(s: np.ndarray, k: float, is_call: bool) -> np.ndarray:
    """Intrinsic value at given stock prices."""
    if is_call:
        return np.maximum(s - k, 0.0)
    return np.maximum(k - s, 0.0)


def bermudan_binomial(spec: BermudanOptionSpec) -> BermudanResult:
    """Price a Bermudan option using the CRR binomial tree.

    Early exercise is checked only at the tree steps closest to each
    element of ``spec.exercise_dates``.

    Parameters
    ----------
    spec : BermudanOptionSpec
        Option specification.

    Returns
    -------
    BermudanResult
        Pricing result with exercise boundary.
    """
    n = spec.n_steps
    dt = spec.T / n
    u = np.exp(spec.sigma * np.sqrt(dt))
    d = 1.0 / u
    disc = np.exp(-spec.r * dt)
    p = (np.exp(spec.r * dt) - d) / (u - d)

    # Map exercise dates to nearest step indices (always include final step)
    exercise_steps: set[int] = {n}
    for t_ex in spec.exercise_dates:
        step = round(t_ex / dt)
        step = max(1, min(step, n))
        exercise_steps.add(step)

    # Build terminal asset prices
    j = np.arange(n + 1)
    s_terminal = spec.s0 * u ** (n - j) * d ** j

    # Option values at maturity
    values = _payoff(s_terminal, spec.k, spec.is_call)

    exercise_boundary: dict[int, float] = {}

    # Backward induction
    for i in range(n - 1, -1, -1):
        s_nodes = spec.s0 * u ** (i - np.arange(i + 1)) * d ** np.arange(i + 1)
        continuation = disc * (p * values[: i + 1] + (1 - p) * values[1: i + 2])

        if i in exercise_steps:
            intrinsic = _payoff(s_nodes, spec.k, spec.is_call)
            exercise_mask = intrinsic > continuation
            values = np.where(exercise_mask, intrinsic, continuation)

            # Record boundary: first node (from the top) where exercise is optimal
            indices = np.where(exercise_mask)[0]
            if len(indices) > 0:
                exercise_boundary[i] = float(s_nodes[indices[0]])
        else:
            values = continuation

    return BermudanResult(price=float(values[0]), exercise_boundary=exercise_boundary)
