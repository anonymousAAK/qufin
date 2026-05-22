"""Advanced tail risk measures.

Provides Entropic Value at Risk (EVaR), tail dependence coefficients,
expected tail loss beyond VaR, and spectral risk measures with
user-defined risk spectra.  All implementations are classical
(numpy / scipy).

References
----------
Ahmadi-Javid, J. Optim. Theory Appl. 155(3), 2012 -- EVaR.
Acerbi, J. Banking & Finance 26(7), 2002 -- Spectral risk measures.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray


def entropic_var(
    losses: NDArray[np.float64],
    alpha: float = 0.95,
) -> float:
    """Entropic Value at Risk at confidence level *alpha*.

    EVaR is the tightest upper bound on VaR / CVaR obtainable via the
    Chernoff inequality.  Computed by optimising over the free parameter
    *z > 0*: EVaR = inf_z { z^{-1} ln( M_L(z) / (1 - alpha) ) }.

    Parameters
    ----------
    losses : NDArray
        Loss samples (positive = loss).
    alpha : float
        Confidence level in (0, 1).

    Returns
    -------
    float
        EVaR estimate.
    """
    from scipy.optimize import minimize_scalar

    losses = np.asarray(losses, dtype=np.float64)
    if len(losses) == 0:
        return 0.0

    def _obj(z: float) -> float:
        if z <= 1e-12:
            return 1e30
        mgf = float(np.mean(np.exp(z * losses)))
        return np.log(mgf / (1.0 - alpha)) / z

    result = minimize_scalar(_obj, bounds=(1e-6, 10.0), method="bounded")
    return float(result.fun)


def tail_dependence_coefficient(
    u: NDArray[np.float64],
    v: NDArray[np.float64],
    threshold: float = 0.95,
) -> float:
    """Non-parametric upper tail dependence coefficient.

    Estimates lambda_U = P(V > q | U > q) at quantile *threshold* using
    rank-transformed data.

    Parameters
    ----------
    u, v : NDArray
        Two return series of equal length.
    threshold : float
        Quantile threshold in (0, 1).

    Returns
    -------
    float
        Estimated upper tail dependence coefficient in [0, 1].
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    n = len(u)
    if n < 2:
        return 0.0

    # Rank-transform to pseudo-uniform
    rank_u = np.argsort(np.argsort(u)) / (n - 1)
    rank_v = np.argsort(np.argsort(v)) / (n - 1)

    mask_u = rank_u > threshold
    if mask_u.sum() == 0:
        return 0.0
    return float(np.mean(rank_v[mask_u] > threshold))


def expected_tail_loss(
    losses: NDArray[np.float64],
    alpha: float = 0.95,
) -> float:
    """Expected tail loss (ETL) beyond VaR.

    Equivalent to CVaR / Expected Shortfall: E[L | L > VaR_alpha].

    Parameters
    ----------
    losses : NDArray
        Loss samples (positive = loss).
    alpha : float
        Confidence level in (0, 1).

    Returns
    -------
    float
        ETL estimate.
    """
    losses = np.asarray(losses, dtype=np.float64)
    if len(losses) == 0:
        return 0.0
    var = float(np.quantile(losses, alpha))
    tail = losses[losses >= var]
    if len(tail) == 0:
        return var
    return float(np.mean(tail))


def spectral_risk_measure(
    losses: NDArray[np.float64],
    phi: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = None,
) -> float:
    """Spectral risk measure with user-defined risk spectrum *phi*.

    M_phi = integral_0^1 phi(p) * q_p dp,  approximated by the
    discrete sum over sorted losses.  *phi* must be non-negative,
    non-decreasing, and integrate to 1.

    Parameters
    ----------
    losses : NDArray
        Loss samples (positive = loss).
    phi : callable | None
        Risk spectrum function mapping quantile levels (array in [0,1])
        to non-negative weights.  If *None*, uses an exponential
        spectrum phi(p) = c * exp(c * p) (risk-averse, c=2).

    Returns
    -------
    float
        Spectral risk measure estimate.
    """
    losses = np.asarray(losses, dtype=np.float64)
    n = len(losses)
    if n == 0:
        return 0.0

    sorted_losses = np.sort(losses)
    p = (np.arange(1, n + 1) - 0.5) / n  # midpoint quantile levels

    if phi is None:
        c = 2.0
        weights = c * np.exp(c * p)
    else:
        weights = phi(p)

    weights = np.maximum(weights, 0.0)
    total = float(np.sum(weights))
    if total < 1e-15:
        return 0.0
    weights = weights / total

    return float(np.dot(weights, sorted_losses))
