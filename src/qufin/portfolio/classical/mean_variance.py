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
        # Cornuejols-Tutuncu transformation: introduce y = w / kappa
        # where kappa = 1^T w (a scalar). The transformed problem is:
        #   min y^T cov y  s.t. (mu - rf)^T y = 1, y >= 0
        # Then w = y / sum(y).
        y = cp.Variable(n)
        sharpe_constraints = [(mu - risk_free_rate) @ y == 1]
        if long_only:
            sharpe_constraints.append(y >= 0)
        sharpe_constraints.append(y <= max_weight * cp.sum(y))
        sharpe_risk = cp.quad_form(y, cov)
        prob = cp.Problem(cp.Minimize(sharpe_risk), sharpe_constraints)
        prob.solve(solver=cp.CLARABEL)
        if y.value is not None:
            y_val = np.array(y.value, dtype=np.float64).flatten()
            y_sum = y_val.sum()
            if y_sum > 1e-12:
                w.value = (y_val / y_sum).reshape(w.shape)
            else:
                w.value = np.ones(n, dtype=np.float64).reshape(w.shape) / n
    else:
        raise ValueError(f"Unknown objective: {objective}")

    if objective != Objective.MAX_SHARPE:
        if cardinality is None:
            prob.solve(solver=cp.CLARABEL)
        else:
            # Try MIQP solvers in preference order
            solved = False
            for solver_name in [cp.SCIP, cp.GLPK_MI]:
                try:
                    prob.solve(solver=solver_name)
                    if w.value is not None:
                        solved = True
                        break
                except (cp.error.SolverError, Exception):
                    continue
            if not solved:
                # Fallback: solve without cardinality, then keep top-K
                prob_cont = cp.Problem(prob.objective, [cp.sum(w) == 1, w >= 0, w <= max_weight])
                prob_cont.solve(solver=cp.CLARABEL)
                if w.value is not None:
                    w_val = np.array(w.value).flatten()
                    top_k = np.argsort(w_val)[-cardinality:]
                    mask = np.zeros(n)
                    mask[top_k] = 1
                    w_val = w_val * mask
                    w_val = w_val / w_val.sum() if w_val.sum() > 0 else w_val
                    w.value = w_val.reshape(w.shape)

    if w.value is None:
        # All solvers failed — return equal weights as a safe fallback
        weights = np.ones(n, dtype=np.float64) / n
    else:
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
