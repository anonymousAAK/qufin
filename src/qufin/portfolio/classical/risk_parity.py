"""Risk parity (equal risk contribution) portfolio optimization.

Finds weights such that each asset contributes equally to total portfolio risk.
Also provides inverse-volatility weighting as a fast heuristic baseline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize


@dataclass
class RiskParityResult:
    """Result from risk parity optimization."""

    weights: NDArray[np.float64]
    risk_contributions: NDArray[np.float64]
    portfolio_volatility: float


def inverse_volatility_weights(
    cov: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Inverse-volatility portfolio weights.

    Each asset is weighted proportionally to 1 / sigma_i, then
    normalised to sum to 1.  This is a fast heuristic that ignores
    correlations.

    Parameters
    ----------
    cov : NDArray
        Covariance matrix, shape (N, N).

    Returns
    -------
    NDArray, shape (N,)
        Normalised inverse-volatility weights.
    """
    vols = np.sqrt(np.maximum(np.diag(cov), 1e-30))
    inv_vols = 1.0 / vols
    return (inv_vols / inv_vols.sum()).astype(np.float64)


def risk_parity(
    cov: NDArray[np.float64],
    budget: NDArray[np.float64] | None = None,
    max_iter: int = 1000,
) -> RiskParityResult:
    """Compute risk parity portfolio.

    Parameters
    ----------
    cov : NDArray
        Covariance matrix, shape (N, N).
    budget : NDArray | None
        Risk budget vector, shape (N,). Default is equal risk (1/N each).
    max_iter : int
        Max optimization iterations.

    Returns
    -------
    RiskParityResult
    """
    n = cov.shape[0]

    if budget is None:
        budget = np.ones(n) / n
    budget = budget / budget.sum()

    def _objective(w: NDArray[np.float64]) -> float:
        w = np.abs(w)  # ensure positive
        sigma = np.sqrt(w @ cov @ w)
        if sigma < 1e-12:
            return 0.0
        # Marginal risk contribution
        mrc = cov @ w / sigma
        # Risk contribution
        rc = w * mrc
        # Minimize sum of squared differences from target budget
        target_rc = budget * sigma
        return float(np.sum((rc - target_rc) ** 2))

    w0 = np.ones(n) / n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(1e-6, 1.0)] * n

    result = minimize(
        _objective,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": max_iter, "ftol": 1e-12},
    )

    weights = np.abs(result.x)
    weights = weights / weights.sum()

    sigma_p = np.sqrt(weights @ cov @ weights)
    mrc = cov @ weights / sigma_p if sigma_p > 1e-12 else np.zeros(n)
    rc = weights * mrc

    return RiskParityResult(
        weights=weights.astype(np.float64),
        risk_contributions=rc.astype(np.float64),
        portfolio_volatility=float(sigma_p),
    )
