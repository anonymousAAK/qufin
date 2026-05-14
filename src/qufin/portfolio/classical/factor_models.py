"""Fama-French factor model integration for portfolio optimization.

Estimates factor exposures via OLS, constructs factor-model covariance
matrices, and decomposes portfolio risk into systematic and idiosyncratic
components.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class FactorExposureResult:
    """Result of factor exposure estimation.

    Attributes:
        betas: Factor loadings, shape (n_assets, n_factors).
        alpha: Regression intercepts, shape (n_assets,).
        r_squared: Coefficient of determination per asset, shape (n_assets,).
        residual_cov: Diagonal residual covariance, shape (n_assets, n_assets).
        factor_names: Human-readable factor labels.
    """

    betas: NDArray[np.float64]
    alpha: NDArray[np.float64]
    r_squared: NDArray[np.float64]
    residual_cov: NDArray[np.float64]
    factor_names: list[str]


@dataclass
class FactorModelResult:
    """Full factor model output.

    Attributes:
        expected_returns: Factor-implied expected returns, shape (n_assets,).
        factor_cov: Factor covariance matrix, shape (n_factors, n_factors).
        cov: Full asset covariance via factor model, shape (n_assets, n_assets).
        exposures: Underlying exposure estimation result.
    """

    expected_returns: NDArray[np.float64]
    factor_cov: NDArray[np.float64]
    cov: NDArray[np.float64]
    exposures: FactorExposureResult


def estimate_factor_exposures(
    returns: NDArray[np.float64],
    factor_returns: NDArray[np.float64],
    window: int | None = None,
    factor_names: list[str] | None = None,
) -> FactorExposureResult:
    """Estimate factor exposures via OLS regression.

    Runs per-asset regressions:  r_i = alpha_i + beta_i^T * f + epsilon_i

    Args:
        returns: Asset return series, shape (T, n_assets).
        factor_returns: Factor return series, shape (T, n_factors).
        window: If provided, use only the last *window* observations.
        factor_names: Optional factor labels. Defaults to
            ``["factor_0", "factor_1", ...]``.

    Returns:
        FactorExposureResult with estimated betas, alphas, R-squared values,
        and diagonal residual covariance.
    """
    returns = np.asarray(returns, dtype=np.float64)
    factor_returns = np.asarray(factor_returns, dtype=np.float64)

    if returns.ndim == 1:
        returns = returns[:, np.newaxis]
    if factor_returns.ndim == 1:
        factor_returns = factor_returns[:, np.newaxis]

    if window is not None:
        returns = returns[-window:]
        factor_returns = factor_returns[-window:]

    t_obs, _n_assets = returns.shape
    n_factors = factor_returns.shape[1]

    if factor_names is None:
        factor_names = [f"factor_{i}" for i in range(n_factors)]

    # Design matrix: [1, f1, f2, ..., fK]
    X = np.column_stack([np.ones(t_obs), factor_returns])  # (T, 1 + n_factors)

    # OLS via least-squares: X @ coeffs = returns
    coeffs, _, _, _ = np.linalg.lstsq(X, returns, rcond=None)  # (1+K, N)

    alpha = coeffs[0, :].astype(np.float64)  # (n_assets,)
    betas = coeffs[1:, :].T.astype(np.float64)  # (n_assets, n_factors)

    # Residuals and R-squared
    fitted = X @ coeffs  # (T, n_assets)
    residuals = returns - fitted  # (T, n_assets)

    ss_res = np.sum(residuals ** 2, axis=0)
    ss_tot = np.sum((returns - returns.mean(axis=0)) ** 2, axis=0)
    # Guard against constant series (ss_tot == 0)
    r_squared = np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, 0.0).astype(np.float64)

    # Diagonal residual covariance (unbiased estimate)
    dof = max(t_obs - n_factors - 1, 1)
    residual_var = ss_res / dof
    residual_cov = np.diag(residual_var).astype(np.float64)

    return FactorExposureResult(
        betas=betas,
        alpha=alpha,
        r_squared=r_squared,
        residual_cov=residual_cov,
        factor_names=factor_names,
    )


def factor_model_cov(
    exposures: FactorExposureResult,
    factor_cov: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute full asset covariance via factor model.

    Sigma = B @ F @ B^T + D

    Args:
        exposures: Factor exposure result containing betas and residual_cov.
        factor_cov: Factor covariance matrix, shape (n_factors, n_factors).

    Returns:
        Asset covariance matrix, shape (n_assets, n_assets).
    """
    B = exposures.betas
    D = exposures.residual_cov
    cov: NDArray[np.float64] = (B @ factor_cov @ B.T + D).astype(np.float64)
    return cov


def factor_expected_returns(
    exposures: FactorExposureResult,
    factor_premium: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute expected returns from factor model.

    E[r] = alpha + B @ factor_premium

    Args:
        exposures: Factor exposure result containing alphas and betas.
        factor_premium: Expected factor risk premia, shape (n_factors,).

    Returns:
        Expected asset returns, shape (n_assets,).
    """
    factor_premium = np.asarray(factor_premium, dtype=np.float64)
    mu = exposures.alpha + exposures.betas @ factor_premium
    return mu.astype(np.float64)


def build_factor_model(
    returns: NDArray[np.float64],
    factor_returns: NDArray[np.float64],
    window: int | None = None,
    factor_names: list[str] | None = None,
) -> FactorModelResult:
    """Build a complete factor model from return data.

    Convenience wrapper that estimates exposures, computes factor covariance
    from the factor return series, and assembles the full asset covariance
    and expected return vector.

    Args:
        returns: Asset return series, shape (T, n_assets).
        factor_returns: Factor return series, shape (T, n_factors).
        window: If provided, use only the last *window* observations for
            both exposure estimation and factor covariance.
        factor_names: Optional factor labels.

    Returns:
        FactorModelResult with expected returns, factor covariance,
        full asset covariance, and exposure details.
    """
    factor_returns = np.asarray(factor_returns, dtype=np.float64)
    if factor_returns.ndim == 1:
        factor_returns = factor_returns[:, np.newaxis]

    exposures = estimate_factor_exposures(
        returns, factor_returns, window=window, factor_names=factor_names,
    )

    # Factor covariance from (possibly windowed) factor returns
    fr = factor_returns[-window:] if window is not None else factor_returns
    f_cov = np.cov(fr, rowvar=False, ddof=1)
    # np.cov returns scalar for single factor; ensure 2-D
    f_cov = np.atleast_2d(f_cov).astype(np.float64)

    # Factor premium defaults to mean of factor returns
    factor_premium = fr.mean(axis=0)

    cov = factor_model_cov(exposures, f_cov)
    expected_ret = factor_expected_returns(exposures, factor_premium)

    return FactorModelResult(
        expected_returns=expected_ret,
        factor_cov=f_cov,
        cov=cov,
        exposures=exposures,
    )


def risk_decomposition(
    weights: NDArray[np.float64],
    exposures: FactorExposureResult,
    factor_cov: NDArray[np.float64],
) -> dict[str, float | NDArray[np.float64]]:
    """Decompose portfolio variance into systematic and idiosyncratic parts.

    Args:
        weights: Portfolio weights, shape (n_assets,).
        exposures: Factor exposure result with betas and residual_cov.
        factor_cov: Factor covariance matrix, shape (n_factors, n_factors).

    Returns:
        Dictionary with keys:

        - ``total_variance``: Total portfolio variance (float).
        - ``systematic_variance``: Variance from factor exposures (float).
        - ``idiosyncratic_variance``: Variance from residuals (float).
        - ``systematic_pct``: Fraction of variance that is systematic (float).
        - ``factor_contributions``: Per-factor variance contribution,
            shape (n_factors,).
    """
    weights = np.asarray(weights, dtype=np.float64)
    factor_cov = np.asarray(factor_cov, dtype=np.float64)

    B = exposures.betas  # (n_assets, n_factors)
    D = exposures.residual_cov  # (n_assets, n_assets)

    # Portfolio factor exposure: (n_factors,)
    port_beta = B.T @ weights

    # Systematic variance: w^T B F B^T w = port_beta^T F port_beta
    systematic_var = float(port_beta @ factor_cov @ port_beta)

    # Idiosyncratic variance: w^T D w
    idiosyncratic_var = float(weights @ D @ weights)

    total_var = systematic_var + idiosyncratic_var

    # Per-factor marginal contributions
    # Factor k contribution = (port_beta_k)^2 * F_kk + cross-terms
    # Use: contribution_k = port_beta_k * (F @ port_beta)_k
    factor_contributions = (port_beta * (factor_cov @ port_beta)).astype(np.float64)

    systematic_pct = systematic_var / total_var if total_var > 0 else 0.0

    return {
        "total_variance": total_var,
        "systematic_variance": systematic_var,
        "idiosyncratic_variance": idiosyncratic_var,
        "systematic_pct": systematic_pct,
        "factor_contributions": factor_contributions,
    }
