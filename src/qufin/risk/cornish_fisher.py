"""Cornish-Fisher VaR expansion and Kelly criterion utilities.

Provides modified Value-at-Risk via the Cornish-Fisher expansion that
accounts for non-normal skewness and kurtosis, as well as Kelly-criterion
position sizing.

References
----------
Cornish & Fisher, "Moments and Cumulants in the Specification of
    Distributions", Revue de l'Institut International de Statistique (1938).
Kelly, "A New Interpretation of Information Rate", Bell System Technical
    Journal (1956).
Maillard, "A User's Guide to the Cornish-Fisher Expansion" (2018).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sp_stats


def cornish_fisher_var(
    returns: NDArray[np.float64],
    confidence: float = 0.99,
) -> dict[str, float]:
    """Modified VaR using the Cornish-Fisher expansion.

    Adjusts the Gaussian quantile for empirical skewness and excess
    kurtosis of the return distribution.

    Args:
        returns: Array of asset returns.
        confidence: Confidence level (e.g. 0.99 for 99 % VaR).

    Returns:
        Dict with keys:

        * ``"var"`` — Cornish-Fisher modified VaR (positive = loss).
        * ``"var_gaussian"`` — Standard Gaussian VaR for comparison.
        * ``"skewness"`` — Sample skewness.
        * ``"excess_kurtosis"`` — Sample excess kurtosis.
        * ``"z_cf"`` — Adjusted quantile used in the expansion.
    """
    r = np.asarray(returns, dtype=np.float64).ravel()
    mu = float(np.mean(r))
    sigma = float(np.std(r, ddof=1))

    s = float(sp_stats.skew(r, bias=False))
    k = float(sp_stats.kurtosis(r, bias=False))  # excess kurtosis

    z = float(sp_stats.norm.ppf(1.0 - confidence))

    # Cornish-Fisher expansion
    z_cf = (
        z
        + (z ** 2 - 1.0) * s / 6.0
        + (z ** 3 - 3.0 * z) * k / 24.0
        - (2.0 * z ** 3 - 5.0 * z) * s ** 2 / 36.0
    )

    var_cf = -(mu + z_cf * sigma)
    var_gauss = -(mu + z * sigma)

    return {
        "var": float(var_cf),
        "var_gaussian": float(var_gauss),
        "skewness": s,
        "excess_kurtosis": k,
        "z_cf": float(z_cf),
    }


def cornish_fisher_es(
    returns: NDArray[np.float64],
    confidence: float = 0.99,
) -> float:
    """Expected Shortfall using the Cornish-Fisher VaR threshold.

    Approximated as the mean of returns that fall below the negative
    of the Cornish-Fisher VaR.

    Args:
        returns: Array of asset returns.
        confidence: Confidence level.

    Returns:
        Expected shortfall (positive = loss).
    """
    r = np.asarray(returns, dtype=np.float64).ravel()
    var_cf = cornish_fisher_var(r, confidence)["var"]

    tail = r[r <= -var_cf]
    if len(tail) == 0:
        # If no observations exceed the VaR, fall back to VaR itself
        return var_cf
    return float(-np.mean(tail))


def kelly_criterion(
    returns: NDArray[np.float64],
    risk_free_rate: float = 0.0,
) -> dict[str, float]:
    """Kelly criterion for optimal position sizing.

    Computes the full Kelly fraction ``f* = (mu - rf) / sigma^2`` and
    the more practical half-Kelly.

    Args:
        returns: Array of asset returns.
        risk_free_rate: Risk-free rate over the same period as *returns*.

    Returns:
        Dict with keys ``"full_kelly"``, ``"half_kelly"``,
        ``"expected_return"``, ``"volatility"``, ``"sharpe"``.
    """
    r = np.asarray(returns, dtype=np.float64).ravel()
    mu = float(np.mean(r))
    sigma = float(np.std(r, ddof=1))
    sigma2 = sigma ** 2

    excess = mu - risk_free_rate
    full_kelly = excess / sigma2 if sigma2 > 0 else 0.0
    sharpe = excess / sigma if sigma > 0 else 0.0

    return {
        "full_kelly": full_kelly,
        "half_kelly": full_kelly / 2.0,
        "expected_return": mu,
        "volatility": sigma,
        "sharpe": sharpe,
    }


def fractional_kelly(
    returns: NDArray[np.float64],
    risk_free_rate: float = 0.0,
    fraction: float = 0.5,
    estimation_error_adjustment: bool = True,
) -> dict[str, float]:
    """Fractional Kelly with optional estimation-error adjustment.

    Args:
        returns: Array of asset returns.
        risk_free_rate: Risk-free rate over the same period as *returns*.
        fraction: Kelly fraction to use (0.5 = half-Kelly).
        estimation_error_adjustment: If ``True``, shrink the Kelly
            fraction by ``(n - 1) / (n + 1)`` where *n* is the sample
            size, accounting for parameter uncertainty.

    Returns:
        Dict with keys:

        * ``"fraction"`` — requested fraction.
        * ``"kelly_fraction"`` — ``fraction * full_kelly``.
        * ``"adjusted_kelly"`` — after estimation-error shrinkage
          (equals *kelly_fraction* when adjustment is off).
        * ``"confidence_band"`` — tuple ``(lower, upper)`` Kelly at
          +/- 1 standard error of the mean estimate.
    """
    r = np.asarray(returns, dtype=np.float64).ravel()
    n = len(r)
    mu = float(np.mean(r))
    sigma = float(np.std(r, ddof=1))
    sigma2 = sigma ** 2

    excess = mu - risk_free_rate
    full_k = excess / sigma2 if sigma2 > 0 else 0.0
    frac_k = fraction * full_k

    if estimation_error_adjustment and n > 1:
        shrinkage = (n - 1) / (n + 1)
        adjusted_k = frac_k * shrinkage
    else:
        adjusted_k = frac_k

    # Confidence band: Kelly at mean +/- 1 SE
    se_mu = sigma / np.sqrt(n) if n > 0 else 0.0
    lower_mu = excess - se_mu
    upper_mu = excess + se_mu
    if sigma2 > 0:
        lower_k = fraction * (lower_mu / sigma2)
        upper_k = fraction * (upper_mu / sigma2)
    else:
        lower_k = 0.0
        upper_k = 0.0

    return {
        "fraction": fraction,
        "kelly_fraction": frac_k,
        "adjusted_kelly": adjusted_k,
        "confidence_band": (float(lower_k), float(upper_k)),
    }
