"""Benchmark metrics: quality, time-to-solution, approximation ratio."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def relative_error(computed: float, reference: float) -> float:
    """Relative error |computed - reference| / |reference|."""
    if abs(reference) < 1e-15:
        return abs(computed)
    return abs(computed - reference) / abs(reference)


def approximation_ratio(objective: float, best_known: float) -> float:
    """Approximation ratio: objective / best_known.

    For minimization, ratio >= 1.0 means worse than optimal.
    """
    if abs(best_known) < 1e-15:
        return float("inf")
    return objective / best_known


def solution_quality(
    weights: NDArray[np.float64],
    mu: NDArray[np.float64],
    cov: NDArray[np.float64],
    risk_free_rate: float = 0.0,
) -> dict[str, float]:
    """Compute portfolio quality metrics.

    Returns dict with: expected_return, volatility, sharpe_ratio.
    """
    ret = float(mu @ weights)
    vol = float(np.sqrt(weights @ cov @ weights))
    sharpe = (ret - risk_free_rate) / vol if vol > 1e-12 else 0.0
    return {
        "expected_return": ret,
        "volatility": vol,
        "sharpe_ratio": sharpe,
    }


def feasibility_rate(
    bitstrings: list[str],
    cardinality: int | None = None,
) -> float:
    """Fraction of sampled bitstrings that satisfy constraints."""
    if not bitstrings:
        return 0.0
    feasible = 0
    for bs in bitstrings:
        hw = sum(int(c) for c in bs)
        if cardinality is not None and hw != cardinality:
            continue
        feasible += 1
    return feasible / len(bitstrings)
