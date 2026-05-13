"""Robust covariance estimation for portfolio optimization.

Implements shrinkage estimators that produce well-conditioned covariance
matrices, especially useful when the number of observations is small
relative to the number of assets.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class CovEstimateResult:
    """Result from covariance estimation.

    Attributes:
        cov: Estimated covariance matrix, shape (p, p).
        method: Name of the estimator used.
        shrinkage_intensity: Shrinkage coefficient alpha in [0, 1], or None
            if no shrinkage was applied.
        n_obs: Number of observations used in estimation.
    """

    cov: NDArray[np.float64]
    method: str
    shrinkage_intensity: float | None
    n_obs: int


def ledoit_wolf(returns: NDArray[np.float64]) -> CovEstimateResult:
    """Ledoit-Wolf linear shrinkage toward scaled identity.

    Implements the analytical formula from Ledoit & Wolf (2004)
    "A well-conditioned estimator for large-dimensional covariance matrices".
    The shrinkage target is F = (trace(S) / p) * I.

    Args:
        returns: Asset returns matrix, shape (n, p) where n is the number
            of observations and p is the number of assets.

    Returns:
        CovEstimateResult with the shrunk covariance matrix.
    """
    returns = np.asarray(returns, dtype=np.float64)
    n, p = returns.shape

    # Sample covariance (using 1/n normalization as in Ledoit-Wolf 2004)
    mean = returns.mean(axis=0)
    x = returns - mean
    sample_cov = (x.T @ x) / n

    # Shrinkage target: scaled identity
    mu = np.trace(sample_cov) / p
    target = mu * np.eye(p)

    # --- Compute optimal shrinkage intensity analytically ---
    # delta = Frobenius norm squared of (S - mu*I), normalized
    delta = np.sum((sample_cov - target) ** 2) / p

    # Estimate beta (sum of asymptotic variances of sample cov entries)
    # beta_hat = (1/n^2) * sum_k || x_k x_k^T - S ||_F^2 / p
    x_outer_minus_s = np.array(
        [np.outer(x[k], x[k]) - sample_cov for k in range(n)]
    )
    beta = np.sum(x_outer_minus_s ** 2) / (n * n * p)

    # Optimal shrinkage: alpha* = beta / delta, clamped to [0, 1]
    alpha = min(1.0, max(0.0, beta / delta)) if delta > 0 else 1.0

    cov_est = alpha * target + (1.0 - alpha) * sample_cov

    return CovEstimateResult(
        cov=cov_est.astype(np.float64),
        method="ledoit_wolf",
        shrinkage_intensity=float(alpha),
        n_obs=n,
    )


def oracle_approx_shrinkage(returns: NDArray[np.float64]) -> CovEstimateResult:
    """Oracle Approximating Shrinkage (OAS) estimator.

    Implements the formula from Chen, Wiesel, Eldar, Hero (2010)
    "Shrinkage Algorithms for MMSE Covariance Estimation". Shrinks toward
    scaled identity with an improved shrinkage formula that often
    outperforms Ledoit-Wolf.

    Args:
        returns: Asset returns matrix, shape (n, p) where n is the number
            of observations and p is the number of assets.

    Returns:
        CovEstimateResult with the shrunk covariance matrix.
    """
    returns = np.asarray(returns, dtype=np.float64)
    n, p = returns.shape

    # Sample covariance (1/(n-1) normalization, standard unbiased)
    mean = returns.mean(axis=0)
    x = returns - mean
    sample_cov = (x.T @ x) / (n - 1) if n > 1 else (x.T @ x)

    # Shrinkage target: scaled identity
    trace_s = np.trace(sample_cov)
    mu = trace_s / p
    target = mu * np.eye(p)

    # OAS shrinkage coefficient
    trace_s2 = np.sum(sample_cov * sample_cov)  # tr(S^2)
    trace_s_sq = trace_s ** 2  # tr(S)^2

    numerator = (1.0 - 2.0 / p) * trace_s2 + trace_s_sq
    denominator = (n + 1.0 - 2.0 / p) * (trace_s2 - trace_s_sq / p)

    if abs(denominator) < 1e-15:
        rho = 1.0
    else:
        rho = numerator / denominator

    rho = min(1.0, max(0.0, rho))

    cov_est = rho * target + (1.0 - rho) * sample_cov

    return CovEstimateResult(
        cov=cov_est.astype(np.float64),
        method="oas",
        shrinkage_intensity=float(rho),
        n_obs=n,
    )


def constant_correlation(returns: NDArray[np.float64]) -> CovEstimateResult:
    """Shrinkage toward constant-correlation matrix.

    The target matrix preserves individual asset variances but replaces
    all pairwise correlations with the average sample correlation.
    Optimal shrinkage intensity is computed analytically following
    Ledoit & Wolf (2004).

    Args:
        returns: Asset returns matrix, shape (n, p) where n is the number
            of observations and p is the number of assets.

    Returns:
        CovEstimateResult with the shrunk covariance matrix.
    """
    returns = np.asarray(returns, dtype=np.float64)
    n, p = returns.shape

    # Sample covariance (1/n normalization)
    mean = returns.mean(axis=0)
    x = returns - mean
    sample_cov = (x.T @ x) / n

    # Standard deviations and correlation matrix
    std = np.sqrt(np.diag(sample_cov))
    std_safe = np.where(std < 1e-15, 1e-15, std)
    std_outer = np.outer(std_safe, std_safe)
    corr = sample_cov / std_outer

    # Average off-diagonal correlation
    mask = ~np.eye(p, dtype=bool)
    if p > 1:
        r_bar = corr[mask].sum() / (p * (p - 1))
    else:
        r_bar = 0.0

    # Constant correlation target F
    target = r_bar * std_outer
    np.fill_diagonal(target, np.diag(sample_cov))

    # --- Optimal shrinkage intensity ---
    # Following Ledoit-Wolf (2004) approach for this target
    # Compute pi_hat: sum of asymptotic variances of s_ij
    y = x ** 2
    phi_mat = (y.T @ y) / n - sample_cov ** 2  # element-wise

    pi_hat = np.sum(phi_mat)

    # Compute gamma_hat: Frobenius norm of (S - F)
    gamma_hat = np.sum((sample_cov - target) ** 2)

    # Compute rho_hat: sum of asymptotic covariances between s_ij and f_ij
    # For constant correlation target, this is more involved
    rho_diag = np.sum(np.diag(phi_mat))  # diagonal terms: same as pi

    # Off-diagonal terms
    rho_off = 0.0
    if p > 1:
        for i in range(p):
            for j in range(p):
                if i != j:
                    # Derivative of f_ij w.r.t. s_kl terms
                    # f_ij = r_bar * sqrt(s_ii * s_jj)
                    # Approximate asymptotic covariance
                    theta_ij = 0.0
                    for k in range(n):
                        term_ij = x[k, i] * x[k, j] - sample_cov[i, j]
                        # Contribution from s_ii
                        if std[i] > 1e-15 and std[j] > 1e-15:
                            e_ii = x[k, i] ** 2 - sample_cov[i, i]
                            e_jj = x[k, j] ** 2 - sample_cov[j, j]
                            f_deriv = r_bar * 0.5 * (
                                std[j] / std_safe[i] * e_ii
                                + std[i] / std_safe[j] * e_jj
                            )
                        else:
                            f_deriv = 0.0
                        theta_ij += term_ij * f_deriv
                    rho_off += theta_ij / n

    rho_hat = rho_diag + rho_off

    # Optimal shrinkage
    kappa = (pi_hat - rho_hat) / gamma_hat if gamma_hat > 0 else 0.0
    alpha = max(0.0, min(1.0, kappa / n))

    cov_est = alpha * target + (1.0 - alpha) * sample_cov

    return CovEstimateResult(
        cov=cov_est.astype(np.float64),
        method="constant_correlation",
        shrinkage_intensity=float(alpha),
        n_obs=n,
    )


def select_estimator(
    returns: NDArray[np.float64],
    method: str = "auto",
) -> CovEstimateResult:
    """Select and apply a covariance estimator.

    Args:
        returns: Asset returns matrix, shape (n, p) where n is the number
            of observations and p is the number of assets.
        method: Estimator to use. One of ``"ledoit_wolf"``, ``"oas"``,
            ``"constant_correlation"``, ``"sample"``, or ``"auto"``.
            When ``"auto"``, uses Ledoit-Wolf if n/p < 10 and sample
            covariance otherwise.

    Returns:
        CovEstimateResult with the estimated covariance matrix.

    Raises:
        ValueError: If *method* is not recognized.
    """
    returns = np.asarray(returns, dtype=np.float64)
    n, p = returns.shape

    if method == "auto":
        method = "ledoit_wolf" if (n / p) < 10 else "sample"

    estimators = {
        "ledoit_wolf": ledoit_wolf,
        "oas": oracle_approx_shrinkage,
        "constant_correlation": constant_correlation,
    }

    if method == "sample":
        cov = np.cov(returns, rowvar=False)
        # np.cov returns a scalar for 1-d input
        if cov.ndim == 0:
            cov = cov.reshape(1, 1)
        return CovEstimateResult(
            cov=cov.astype(np.float64),
            method="sample",
            shrinkage_intensity=None,
            n_obs=n,
        )

    if method not in estimators:
        raise ValueError(
            f"Unknown method {method!r}. Choose from "
            f"{sorted([*estimators.keys(), 'sample', 'auto'])}."
        )

    return estimators[method](returns)
