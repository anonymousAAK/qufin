"""Black-Litterman model for portfolio allocation.

Combines market equilibrium returns with investor views
to produce a posterior expected return vector.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class BLResult:
    """Black-Litterman posterior result."""

    posterior_mu: NDArray[np.float64]
    posterior_cov: NDArray[np.float64]
    equilibrium_mu: NDArray[np.float64]


def black_litterman(
    cov: NDArray[np.float64],
    market_caps: NDArray[np.float64],
    risk_aversion: float = 2.5,
    tau: float = 0.05,
    P: NDArray[np.float64] | None = None,
    Q: NDArray[np.float64] | None = None,
    omega: NDArray[np.float64] | None = None,
) -> BLResult:
    """Compute Black-Litterman posterior expected returns.

    Parameters
    ----------
    cov : NDArray
        Covariance matrix of returns, shape (N, N).
    market_caps : NDArray
        Market capitalizations, shape (N,). Used to derive equilibrium weights.
    risk_aversion : float
        Risk aversion coefficient (lambda). Typical range: 1-4.
    tau : float
        Scalar controlling uncertainty in the prior (equilibrium). Typical: 0.01-0.1.
    P : NDArray | None
        Views matrix, shape (K, N). Each row is one view linking assets.
    Q : NDArray | None
        Views vector, shape (K,). Expected return of each view.
    omega : NDArray | None
        Uncertainty of views, shape (K, K). If None, uses proportional-to-variance
        heuristic: omega_ii = tau * P_i @ cov @ P_i^T.

    Returns
    -------
    BLResult
    """
    len(market_caps)

    # Equilibrium weights from market cap
    w_eq = market_caps / market_caps.sum()

    # Implied equilibrium excess returns: pi = lambda * cov @ w_eq
    pi = risk_aversion * cov @ w_eq

    if P is None or Q is None:
        # No views: posterior = prior
        return BLResult(
            posterior_mu=pi,
            posterior_cov=cov,
            equilibrium_mu=pi,
        )

    P = np.atleast_2d(P)
    Q = np.atleast_1d(Q)

    if omega is None:
        # Proportional to variance heuristic (Idzorek, 2007)
        omega = np.diag(np.diag(tau * P @ cov @ P.T))

    tau_cov = tau * cov
    tau_cov_inv = np.linalg.inv(tau_cov)
    omega_inv = np.linalg.inv(omega)

    # Posterior precision = prior precision + views precision
    posterior_cov_inv = tau_cov_inv + P.T @ omega_inv @ P
    posterior_cov = np.linalg.inv(posterior_cov_inv)

    # Posterior mean
    posterior_mu = posterior_cov @ (tau_cov_inv @ pi + P.T @ omega_inv @ Q)

    return BLResult(
        posterior_mu=posterior_mu.astype(np.float64),
        posterior_cov=posterior_cov.astype(np.float64),
        equilibrium_mu=pi.astype(np.float64),
    )
