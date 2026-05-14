"""Gaussian copula credit model.

Implements the one-factor Gaussian copula model for portfolio credit risk.
Used as the classical baseline for quantum credit risk (Egger et al.).

The model assumes each obligor i defaults when its latent variable
X_i = sqrt(rho_i)*Z + sqrt(1-rho_i)*eps_i falls below a threshold,
where Z ~ N(0,1) is the systemic factor and eps_i ~ N(0,1) is idiosyncratic.

References
----------
Li, "On Default Correlation: A Copula Function Approach",
Journal of Fixed Income 9(4):43-54 (2000).

Vasicek, "The Distribution of Loan Portfolio Value",
Risk Magazine (2002).

Egger et al., "Credit Risk Analysis using Quantum Computers",
IEEE Trans. Computers 70(12) (2021), arXiv:1907.03044.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm


@dataclass
class CreditPortfolio:
    """Credit portfolio specification.

    Parameters
    ----------
    n_obligors : int
        Number of obligors (counterparties).
    default_probs : NDArray[np.float64]
        Marginal default probabilities, shape (n_obligors,).
    correlations : NDArray[np.float64]
        Asset correlation with systemic factor, shape (n_obligors,).
        Single float broadcast to all obligors.
    exposures : NDArray[np.float64]
        Loss-given-default per obligor, shape (n_obligors,).
    recovery_rates : NDArray[np.float64]
        Recovery rates per obligor, shape (n_obligors,).
    """

    n_obligors: int
    default_probs: NDArray[np.float64]
    correlations: NDArray[np.float64]
    exposures: NDArray[np.float64]
    recovery_rates: NDArray[np.float64] = field(default_factory=lambda: np.array([]))

    def __post_init__(self) -> None:
        self.default_probs = np.asarray(self.default_probs, dtype=np.float64)
        self.correlations = np.asarray(self.correlations, dtype=np.float64)
        self.exposures = np.asarray(self.exposures, dtype=np.float64)

        if self.correlations.ndim == 0:
            self.correlations = np.full(self.n_obligors, float(self.correlations))

        if len(self.recovery_rates) == 0:
            self.recovery_rates = np.zeros(self.n_obligors)
        else:
            self.recovery_rates = np.asarray(self.recovery_rates, dtype=np.float64)

    @property
    def lgd(self) -> NDArray[np.float64]:
        """Loss given default = exposure * (1 - recovery_rate)."""
        return self.exposures * (1 - self.recovery_rates)

    @property
    def default_thresholds(self) -> NDArray[np.float64]:
        """Default thresholds: Phi^{-1}(PD_i)."""
        return norm.ppf(self.default_probs)


@dataclass
class CreditRiskResult:
    """Result from credit risk analysis."""

    expected_loss: float
    unexpected_loss: float
    var_99: float
    es_99: float
    loss_distribution: NDArray[np.float64]
    n_simulations: int
    method: str


def gaussian_copula_mc(
    portfolio: CreditPortfolio,
    n_simulations: int = 100_000,
    confidence: float = 0.99,
    seed: int | None = 42,
) -> CreditRiskResult:
    """Monte Carlo simulation under the one-factor Gaussian copula.

    Parameters
    ----------
    portfolio : CreditPortfolio
        Credit portfolio specification.
    n_simulations : int
        Number of MC scenarios.
    confidence : float
        VaR/ES confidence level.
    seed : int or None
        Random seed.
    """
    rng = np.random.default_rng(seed)

    n = portfolio.n_obligors
    thresholds = portfolio.default_thresholds
    rho = portfolio.correlations
    lgd = portfolio.lgd

    # Simulate systemic factor Z
    z = rng.standard_normal(n_simulations)

    # Simulate idiosyncratic factors
    eps = rng.standard_normal((n_simulations, n))

    # Latent variables: X_i = sqrt(rho_i)*Z + sqrt(1-rho_i)*eps_i
    x = np.sqrt(rho)[None, :] * z[:, None] + np.sqrt(1 - rho)[None, :] * eps

    # Default indicator: X_i < threshold_i
    defaults = (x < thresholds[None, :]).astype(np.float64)

    # Portfolio losses
    losses = defaults @ lgd

    # Statistics
    expected_loss = float(np.mean(losses))
    unexpected_loss = float(np.std(losses))
    var = float(np.percentile(losses, 100 * confidence))
    tail = losses[losses >= var]
    es = float(np.mean(tail)) if len(tail) > 0 else var

    return CreditRiskResult(
        expected_loss=expected_loss,
        unexpected_loss=unexpected_loss,
        var_99=var,
        es_99=es,
        loss_distribution=losses,
        n_simulations=n_simulations,
        method="gaussian_copula_mc",
    )


def vasicek_analytical(
    pd: float,
    rho: float,
    lgd: float = 1.0,
    confidence: float = 0.99,
) -> dict[str, float]:
    """Vasicek single-factor analytical approximation.

    For a homogeneous portfolio (all obligors have the same PD and rho),
    the loss distribution has a closed-form CDF (Vasicek 2002).

    Parameters
    ----------
    pd : float
        Common default probability.
    rho : float
        Common asset correlation.
    lgd : float
        Common loss-given-default.
    confidence : float
        Confidence level for VaR.

    Returns
    -------
    dict with 'expected_loss', 'var', 'conditional_pd'.
    """
    # Conditional PD at the VaR level
    # Use the adverse quantile: Z at the (1-conf) percentile (bad state)
    z_alpha = norm.ppf(1 - confidence)
    threshold = norm.ppf(pd)
    conditional_pd = float(norm.cdf(
        (threshold - np.sqrt(rho) * z_alpha) / np.sqrt(1 - rho)
    ))

    expected_loss = pd * lgd
    var = conditional_pd * lgd

    return {
        "expected_loss": expected_loss,
        "var": var,
        "conditional_pd": conditional_pd,
    }
