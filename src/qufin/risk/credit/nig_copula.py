"""Normal-Inverse Gaussian copula for CDO pricing (arXiv:2008.04110).

Implements the NIG distribution as an alternative to the Gaussian
copula for modeling heavy-tailed default dependence in CDO tranches.

The NIG distribution has four parameters (alpha, beta, mu, delta)
and captures skewness and kurtosis better than the Gaussian.

References
----------
Barndorff-Nielsen, "Normal Inverse Gaussian Distributions and
Stochastic Volatility Modelling", Scand. J. Stat. 24:1-13 (1997).

Egger et al., "Quantum Computing for Finance: State-of-the-Art
and Future Prospects", IEEE Trans. QE 2:3101720 (2021).

Braun et al., arXiv:2008.04110 — NIG copula for credit with QAE.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.special import kv as bessel_k
from scipy.stats import norm


@dataclass
class NIGParams:
    """Normal-Inverse Gaussian distribution parameters.

    Parameters
    ----------
    alpha : float
        Tail heaviness (alpha > 0). Larger = lighter tails.
    beta : float
        Asymmetry (-alpha < beta < alpha). beta=0 is symmetric.
    mu : float
        Location parameter.
    delta : float
        Scale parameter (delta > 0).
    """

    alpha: float = 1.5
    beta: float = 0.0
    mu: float = 0.0
    delta: float = 1.0

    def __post_init__(self) -> None:
        if self.alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {self.alpha}")
        if self.delta <= 0:
            raise ValueError(f"delta must be > 0, got {self.delta}")
        if abs(self.beta) >= self.alpha:
            raise ValueError(f"|beta| must be < alpha, got |{self.beta}| >= {self.alpha}")

    @property
    def gamma(self) -> float:
        """gamma = sqrt(alpha^2 - beta^2)."""
        return float(np.sqrt(self.alpha**2 - self.beta**2))

    @property
    def mean(self) -> float:
        return self.mu + self.delta * self.beta / self.gamma

    @property
    def variance(self) -> float:
        return self.delta * self.alpha**2 / self.gamma**3

    @property
    def skewness(self) -> float:
        return 3 * self.beta / (self.alpha * np.sqrt(self.delta * self.gamma))

    @property
    def kurtosis_excess(self) -> float:
        return 3 * (1 + 4 * self.beta**2 / self.alpha**2) / (self.delta * self.gamma)


def nig_pdf(
    x: NDArray[np.float64],
    params: NIGParams,
) -> NDArray[np.float64]:
    """NIG probability density function.

    f(x) = (alpha * delta / pi) * exp(delta*gamma + beta*(x-mu))
            * K_1(alpha * sqrt(delta^2 + (x-mu)^2))
            / sqrt(delta^2 + (x-mu)^2)

    where K_1 is the modified Bessel function of the second kind.
    """
    x = np.asarray(x, dtype=np.float64)
    a, b, mu, d = params.alpha, params.beta, params.mu, params.delta
    g = params.gamma

    q = np.sqrt(d**2 + (x - mu) ** 2)
    log_pdf = (
        np.log(a * d / np.pi)
        + d * g
        + b * (x - mu)
        + np.log(bessel_k(1, a * q))
        - np.log(q)
    )
    return np.exp(log_pdf)


def nig_copula_mc(
    n_obligors: int,
    default_probs: NDArray[np.float64],
    nig_params: NIGParams,
    exposures: NDArray[np.float64],
    n_simulations: int = 100_000,
    confidence: float = 0.99,
    seed: int | None = 42,
) -> dict[str, float]:
    """Monte Carlo simulation with NIG copula.

    Uses a one-factor NIG model: each obligor's latent variable is
    X_i = Z + eps_i where Z follows a NIG distribution (systemic factor)
    and eps_i ~ N(0,1) (idiosyncratic).

    Parameters
    ----------
    n_obligors : int
        Number of obligors.
    default_probs : NDArray
        Marginal default probabilities.
    nig_params : NIGParams
        Parameters for the NIG systemic factor.
    exposures : NDArray
        Loss given default per obligor.
    n_simulations : int
        Number of simulations.
    confidence : float
        Confidence level for VaR.
    seed : int or None
        Random seed.
    """
    rng = np.random.default_rng(seed)
    default_probs = np.asarray(default_probs)
    exposures = np.asarray(exposures)
    thresholds = norm.ppf(default_probs)

    # Simulate NIG systemic factor via normal-mean variance mixture:
    # Z = mu + beta*V + sqrt(V)*W, where V ~ IG(delta, gamma), W ~ N(0,1)
    _a, b, mu, d = nig_params.alpha, nig_params.beta, nig_params.mu, nig_params.delta
    g = nig_params.gamma

    # Inverse Gaussian samples: V ~ IG(delta/gamma, delta^2)
    # Using Wald distribution parameterization
    ig_mu = d / g
    ig_lambda = d**2
    v_samples = rng.wald(ig_mu, ig_lambda, n_simulations)

    w_samples = rng.standard_normal(n_simulations)
    z_samples = mu + b * v_samples + np.sqrt(v_samples) * w_samples

    # Transform Z to uniform via NIG CDF (approximate with empirical)
    z_sorted = np.sort(z_samples)
    z_ranks = np.searchsorted(z_sorted, z_samples) / n_simulations

    # Map uniform to standard normal for threshold comparison
    z_normal = norm.ppf(np.clip(z_ranks, 1e-10, 1 - 1e-10))

    # Idiosyncratic noise
    eps = rng.standard_normal((n_simulations, n_obligors))

    # Latent: simple additive factor model
    rho = 0.3  # default correlation strength
    x = np.sqrt(rho) * z_normal[:, None] + np.sqrt(1 - rho) * eps

    defaults = (x < thresholds[None, :]).astype(np.float64)
    losses = defaults @ exposures

    expected_loss = float(np.mean(losses))
    var_val = float(np.percentile(losses, 100 * confidence))
    tail = losses[losses >= var_val]
    es_val = float(np.mean(tail)) if len(tail) > 0 else var_val

    return {
        "expected_loss": expected_loss,
        "var": var_val,
        "es": es_val,
        "n_simulations": n_simulations,
        "method": "nig_copula_mc",
    }
