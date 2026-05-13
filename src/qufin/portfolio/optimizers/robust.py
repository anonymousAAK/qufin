"""Robust portfolio optimization with worst-case CVaR under uncertainty.

Implements a robust counterpart of the Markowitz mean-variance model
using ellipsoidal uncertainty sets for expected returns. The worst-case
expected return under the uncertainty set is maximized, yielding a
minimax formulation that guards against estimation error.

The QUBO formulation embeds the robust counterpart so it can be solved
on quantum hardware via QAOA or VQE, while a classical comparison is
provided via CVXPY's second-order cone programming.

References
----------
Goldfarb & Iyengar, "Robust Portfolio Selection Problems,"
    Mathematics of Operations Research 28(1):1-38 (2003).
Tutuncu & Koenig, "Robust Asset Allocation,"
    Annals of Operations Research 132:157-187 (2004).
Ben-Tal, El Ghaoui, Nemirovski, "Robust Optimization,"
    Princeton University Press (2009).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from qufin.portfolio.qubo import PortfolioQUBO
from qufin.utils.results import Result


@dataclass
class EllipsoidalUncertaintySet:
    """Ellipsoidal uncertainty set for expected returns.

    The true mean mu lies in the set:
        { mu : (mu - mu_hat)^T Sigma_mu_inv (mu - mu_hat) <= epsilon^2 }

    where mu_hat is the estimated mean, Sigma_mu is the uncertainty
    shape matrix (typically proportional to the covariance), and
    epsilon controls the size of the uncertainty region.

    Parameters
    ----------
    mu_hat : NDArray
        Estimated (nominal) expected returns, shape (N,).
    sigma_mu : NDArray
        Uncertainty shape matrix, shape (N, N). Positive semidefinite.
    epsilon : float
        Uncertainty radius. 0 means no uncertainty (standard MVO).
    """

    mu_hat: NDArray[np.float64]
    sigma_mu: NDArray[np.float64]
    epsilon: float

    def __post_init__(self) -> None:
        n = len(self.mu_hat)
        if self.sigma_mu.shape != (n, n):
            raise ValueError(
                f"sigma_mu shape {self.sigma_mu.shape} doesn't match "
                f"mu_hat length {n}"
            )
        if self.epsilon < 0:
            raise ValueError(f"epsilon must be non-negative, got {self.epsilon}")


def build_ellipsoidal_uncertainty(
    returns: NDArray[np.float64],
    epsilon: float = 0.1,
    shrinkage: float = 0.0,
) -> EllipsoidalUncertaintySet:
    """Construct an ellipsoidal uncertainty set from historical returns.

    The shape matrix is derived from the sample covariance of returns,
    scaled by 1/T (the estimation uncertainty of the sample mean).
    An optional Ledoit-Wolf-style shrinkage can be applied.

    Parameters
    ----------
    returns : NDArray
        Historical return matrix, shape (T, N) where T is number of
        observations and N is number of assets.
    epsilon : float
        Uncertainty radius controlling conservatism. Larger values
        yield more robust (conservative) portfolios.
    shrinkage : float
        Shrinkage intensity in [0, 1]. 0 = sample covariance,
        1 = diagonal (uncorrelated) target.

    Returns
    -------
    EllipsoidalUncertaintySet
    """
    if returns.ndim != 2:
        raise ValueError(f"returns must be 2-D, got shape {returns.shape}")
    T, N = returns.shape
    if T < 2:
        raise ValueError(f"Need at least 2 observations, got {T}")

    mu_hat = np.mean(returns, axis=0)
    cov = np.cov(returns, rowvar=False, ddof=1)

    # Uncertainty of the sample mean: Sigma_mu = Sigma / T
    sigma_mu = cov / T

    if shrinkage > 0:
        target = np.diag(np.diag(sigma_mu))
        sigma_mu = (1 - shrinkage) * sigma_mu + shrinkage * target

    return EllipsoidalUncertaintySet(
        mu_hat=mu_hat,
        sigma_mu=sigma_mu,
        epsilon=epsilon,
    )


@dataclass
class RobustPortfolioResult(Result):
    """Result from robust portfolio optimization.

    Attributes
    ----------
    best_bitstring : str
        Best binary solution found (quantum solver only).
    best_objective : float
        Objective value of the best solution.
    weights : NDArray[np.float64]
        Portfolio weights.
    worst_case_return : float
        Worst-case expected return under the uncertainty set.
    nominal_return : float
        Nominal expected return (at center of uncertainty set).
    risk : float
        Portfolio variance (w^T Sigma w).
    feasible : bool
        Whether constraints are satisfied.
    """

    best_bitstring: str = ""
    best_objective: float = float("inf")
    weights: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    worst_case_return: float = 0.0
    nominal_return: float = 0.0
    risk: float = 0.0
    feasible: bool = False


class RobustPortfolioOptimizer:
    """Worst-case CVaR portfolio optimizer with QUBO formulation.

    Solves the robust Markowitz problem:

        min  gamma * w^T Sigma w  -  min_{mu in U} mu^T w
        s.t. sum(w) = 1, w >= 0

    For the ellipsoidal uncertainty set U, the inner min has a
    closed-form worst case:

        min_{mu in U} mu^T w = mu_hat^T w - epsilon * sqrt(w^T Sigma_mu w)

    The QUBO formulation linearizes the sqrt term via a first-order
    Taylor expansion around equal weights, yielding a penalty that
    can be embedded into the standard QUBO matrix.

    Parameters
    ----------
    uncertainty : EllipsoidalUncertaintySet
        The uncertainty set for expected returns.
    cov : NDArray
        Asset return covariance matrix, shape (N, N).
    gamma : float
        Risk aversion parameter.
    cardinality : int | None
        If set, select exactly this many assets.
    budget_penalty : float | None
        Penalty for budget constraint. Auto-scaled if None.
    """

    def __init__(
        self,
        uncertainty: EllipsoidalUncertaintySet,
        cov: NDArray[np.float64],
        gamma: float = 1.0,
        cardinality: int | None = None,
        budget_penalty: float | None = None,
    ) -> None:
        self.uncertainty = uncertainty
        self.cov = cov
        self.gamma = gamma
        self.cardinality = cardinality
        self.budget_penalty = budget_penalty
        self._n_assets = len(uncertainty.mu_hat)

    @property
    def n_assets(self) -> int:
        return self._n_assets

    def _compute_robust_mu(self) -> NDArray[np.float64]:
        """Compute the effective (worst-case adjusted) expected returns.

        For binary decision variables x_i in {0, 1} with equal-weight
        decoding (w_i = x_i / sum(x)), the worst-case return penalty
        is linearized around the assumption that K assets are selected
        with equal weight 1/K.

        The adjusted return for asset i is:
            mu_adj_i = mu_hat_i - epsilon * sqrt(e_i^T Sigma_mu e_i) / sqrt(K)

        where e_i is the i-th standard basis vector. This simplifies to:
            mu_adj_i = mu_hat_i - epsilon * sqrt(Sigma_mu[i,i]) / sqrt(K)

        For the QUBO without cardinality, we use K = N (all assets).
        """
        n = self._n_assets
        K = self.cardinality if self.cardinality is not None else n
        eps = self.uncertainty.epsilon
        sigma_mu = self.uncertainty.sigma_mu

        mu_adj = self.uncertainty.mu_hat.copy()
        if eps > 0 and K > 0:
            # Per-asset worst-case penalty from diagonal of uncertainty matrix
            for i in range(n):
                mu_adj[i] -= eps * np.sqrt(max(0.0, sigma_mu[i, i])) / np.sqrt(K)

            # Cross-correlation penalty: additional quadratic adjustment
            # This captures the off-diagonal elements of sigma_mu
            # via a first-order correction to the per-asset penalty
            L = np.linalg.cholesky(
                sigma_mu + 1e-12 * np.eye(n)  # regularize for PSD
            )
            # The norm of each column of L gives the marginal contribution
            col_norms = np.sqrt(np.sum(L**2, axis=0))
            for i in range(n):
                correction = col_norms[i] - np.sqrt(max(0.0, sigma_mu[i, i]))
                mu_adj[i] -= eps * correction / np.sqrt(K)

        return mu_adj

    def build_qubo(self) -> PortfolioQUBO:
        """Build a QUBO with robust (worst-case) adjusted returns.

        Returns
        -------
        PortfolioQUBO
            A standard QUBO object with mu replaced by the worst-case
            adjusted returns.
        """
        mu_adj = self._compute_robust_mu()
        return PortfolioQUBO(
            mu=mu_adj,
            cov=self.cov,
            gamma=self.gamma,
            cardinality=self.cardinality,
            budget_penalty=self.budget_penalty,
        )

    def build_matrix(self) -> NDArray[np.float64]:
        """Build the robust QUBO Q matrix directly.

        Returns
        -------
        NDArray of shape (n_assets, n_assets)
        """
        return self.build_qubo().build_matrix()

    def solve_exhaustive(self) -> RobustPortfolioResult:
        """Solve the robust QUBO by exhaustive enumeration.

        Only practical for small problems (N <= 20).

        Returns
        -------
        RobustPortfolioResult
        """
        start = time.perf_counter()
        qubo = self.build_qubo()
        Q = qubo.build_matrix()
        n = self._n_assets

        best_obj = float("inf")
        best_bs = "0" * n

        for k in range(1, 2**n):
            bs = format(k, f"0{n}b")
            x = np.array([int(c) for c in bs], dtype=np.float64)
            obj = float(x @ Q @ x)
            if obj < best_obj:
                best_obj = obj
                best_bs = bs

        weights = qubo.decode_weights(best_bs)
        feasibility = qubo.feasibility_check(best_bs)
        nominal_ret = float(self.uncertainty.mu_hat @ weights)
        wc_ret = float(self._compute_robust_mu() @ weights)
        risk = float(weights @ self.cov @ weights)
        wall_time = time.perf_counter() - start

        return RobustPortfolioResult(
            value=best_obj,
            wall_time_s=wall_time,
            best_bitstring=best_bs,
            best_objective=best_obj,
            weights=weights,
            worst_case_return=wc_ret,
            nominal_return=nominal_ret,
            risk=risk,
            feasible=all(feasibility.values()) if feasibility else True,
        )


def robust_classical(
    mu: NDArray[np.float64],
    cov: NDArray[np.float64],
    uncertainty: EllipsoidalUncertaintySet,
    gamma: float = 1.0,
    long_only: bool = True,
    max_weight: float = 1.0,
) -> RobustPortfolioResult:
    """Solve the robust portfolio problem classically via CVXPY.

    Solves the second-order cone program (SOCP):

        min  gamma * w^T Sigma w  -  mu_hat^T w  +  epsilon * ||L^T w||_2
        s.t. sum(w) = 1
             w >= 0 (if long_only)
             w <= max_weight

    where L is the Cholesky factor of sigma_mu.

    Parameters
    ----------
    mu : NDArray
        Nominal expected returns.
    cov : NDArray
        Return covariance matrix.
    uncertainty : EllipsoidalUncertaintySet
        Uncertainty set specification.
    gamma : float
        Risk aversion parameter.
    long_only : bool
        If True, enforce w >= 0.
    max_weight : float
        Maximum weight per asset.

    Returns
    -------
    RobustPortfolioResult
    """
    import cvxpy as cp

    start = time.perf_counter()
    n = len(mu)

    w = cp.Variable(n)
    ret_term = uncertainty.mu_hat @ w

    # Robust penalty: epsilon * ||L^T w||_2
    sigma_mu_reg = uncertainty.sigma_mu + 1e-12 * np.eye(n)
    L = np.linalg.cholesky(sigma_mu_reg)
    robust_penalty = uncertainty.epsilon * cp.norm(L.T @ w, 2)

    # Risk term
    risk_term = gamma * cp.quad_form(w, cp.psd_wrap(cov))

    objective = cp.Minimize(risk_term - ret_term + robust_penalty)

    constraints = [cp.sum(w) == 1]
    if long_only:
        constraints.append(w >= 0)
    if max_weight < 1.0:
        constraints.append(w <= max_weight)

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS, verbose=False)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        # Fallback: try ECOS
        prob.solve(solver=cp.ECOS, verbose=False)

    w_val = w.value
    if w_val is None:
        w_val = np.ones(n) / n  # fallback to equal weight

    w_val = np.maximum(w_val, 0) if long_only else w_val
    w_val = w_val / np.sum(w_val)  # re-normalize

    nominal_ret = float(mu @ w_val)
    wc_ret = float(mu @ w_val - uncertainty.epsilon * np.sqrt(
        float(w_val @ uncertainty.sigma_mu @ w_val)
    ))
    risk = float(w_val @ cov @ w_val)
    obj_val = float(gamma * risk - wc_ret)
    wall_time = time.perf_counter() - start

    return RobustPortfolioResult(
        value=obj_val,
        wall_time_s=wall_time,
        weights=w_val,
        worst_case_return=wc_ret,
        nominal_return=nominal_ret,
        risk=risk,
        feasible=True,
    )
