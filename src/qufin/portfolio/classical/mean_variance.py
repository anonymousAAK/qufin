"""Mean-variance portfolio optimization using CVXPY.

Implements Markowitz (1952) mean-variance optimization:
- Minimum variance
- Maximum return subject to variance constraint
- Maximum Sharpe ratio
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import cvxpy as cp
import numpy as np
from numpy.typing import NDArray


class Objective(str, Enum):
    MIN_VARIANCE = "min_variance"
    MAX_RETURN = "max_return"
    MAX_SHARPE = "max_sharpe"


@dataclass
class MVResult:
    """Result from mean-variance optimization."""

    weights: NDArray[np.float64]
    expected_return: float
    volatility: float
    sharpe_ratio: float


def mean_variance(
    mu: NDArray[np.float64],
    cov: NDArray[np.float64],
    objective: Objective = Objective.MIN_VARIANCE,
    risk_free_rate: float = 0.0,
    max_variance: float | None = None,
    min_return: float | None = None,
    long_only: bool = True,
    max_weight: float = 1.0,
    cardinality: int | None = None,
    sector_upper: dict[int, float] | None = None,
) -> MVResult:
    """Solve a mean-variance portfolio optimization.

    Parameters
    ----------
    mu : NDArray
        Expected returns vector, shape (N,).
    cov : NDArray
        Covariance matrix, shape (N, N).
    objective : Objective
        Which objective to optimize.
    risk_free_rate : float
        Risk-free rate for Sharpe calculation.
    max_variance : float | None
        Upper bound on portfolio variance (for MAX_RETURN).
    min_return : float | None
        Lower bound on expected return (for MIN_VARIANCE).
    long_only : bool
        If True, weights >= 0.
    max_weight : float
        Maximum weight per asset.
    cardinality : int | None
        Maximum number of assets (makes problem MIQP — slower).
    sector_upper : dict | None
        Mapping of asset index -> sector weight cap.

    Returns
    -------
    MVResult
    """
    n = len(mu)
    w = cp.Variable(n)
    ret = mu @ w
    risk = cp.quad_form(w, cov)

    constraints = [cp.sum(w) == 1]

    if long_only:
        constraints.append(w >= 0)
    constraints.append(w <= max_weight)

    if min_return is not None:
        constraints.append(ret >= min_return)
    if max_variance is not None:
        constraints.append(risk <= max_variance)

    if cardinality is not None:
        z = cp.Variable(n, boolean=True)
        constraints.append(w <= z * max_weight)
        if long_only:
            constraints.append(w >= 0)
        constraints.append(cp.sum(z) <= cardinality)

    if objective == Objective.MIN_VARIANCE:
        prob = cp.Problem(cp.Minimize(risk), constraints)
    elif objective == Objective.MAX_RETURN:
        prob = cp.Problem(cp.Maximize(ret), constraints)
    elif objective == Objective.MAX_SHARPE:
        # Sharpe via Cornuejols-Tutuncu transformation
        prob = cp.Problem(cp.Minimize(risk), constraints + [ret - risk_free_rate >= 1])
    else:
        raise ValueError(f"Unknown objective: {objective}")

    prob.solve(solver=cp.CLARABEL if cardinality is None else cp.SCIP)

    weights = np.array(w.value, dtype=np.float64).flatten()
    exp_ret = float(mu @ weights)
    vol = float(np.sqrt(weights @ cov @ weights))
    sharpe = (exp_ret - risk_free_rate) / vol if vol > 0 else 0.0

    return MVResult(
        weights=weights,
        expected_return=exp_ret,
        volatility=vol,
        sharpe_ratio=sharpe,
    )
