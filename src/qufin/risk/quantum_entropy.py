"""Quantum entropy-based risk measures for portfolio analysis.

Computes Von Neumann entropy of portfolio correlation matrices and
quantum relative entropy for distribution divergence, providing
information-theoretic risk measures grounded in quantum mechanics.

The Von Neumann entropy of a correlation matrix captures portfolio
diversification: a maximally diversified portfolio has maximum entropy
(proportional to log N), while a concentrated portfolio has near-zero
entropy.

References
----------
Bouchaud & Potters, "Theory of Financial Risk and Derivative Pricing" (2003).
Nielsen & Chuang, "Quantum Computation and Quantum Information" (2010).
Pichler, Puhr & Wurm, arXiv:1811.00787 -- risk via quantum entropy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class EntropyResult:
    """Result of an entropy-based risk measure computation.

    Parameters
    ----------
    entropy : float
        The computed entropy value.
    normalised_entropy : float
        Entropy normalised by log(N) to lie in [0, 1].
    effective_rank : float
        exp(entropy), the effective number of independent components.
    eigenvalues : NDArray
        Eigenvalues of the density matrix.
    method : str
        Computation method ("von_neumann", "relative", "shannon").
    """

    entropy: float = 0.0
    normalised_entropy: float = 0.0
    effective_rank: float = 1.0
    eigenvalues: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(0)
    )
    method: str = "von_neumann"


@dataclass
class RelativeEntropyResult:
    """Result from quantum relative entropy computation.

    Parameters
    ----------
    divergence : float
        S(rho || sigma) = Tr[rho (log rho - log sigma)].
    is_finite : bool
        True if the divergence is finite (supp(rho) in supp(sigma)).
    symmetrised : float
        Symmetrised divergence: (S(rho||sigma) + S(sigma||rho)) / 2.
    """

    divergence: float = 0.0
    is_finite: bool = True
    symmetrised: float = 0.0


def _to_density_matrix(cov: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert a covariance or correlation matrix to a density matrix.

    The density matrix is rho = C / Tr(C) where C is the positive
    semi-definite input.  Negative eigenvalues are clamped to zero.
    """
    cov = np.asarray(cov, dtype=np.float64)
    # Symmetrise
    cov = (cov + cov.T) / 2.0
    # Eigendecompose and clamp
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 0.0)
    # Reconstruct PSD matrix
    psd = (eigvecs * eigvals) @ eigvecs.T
    tr = np.trace(psd)
    if tr < 1e-30:
        n = cov.shape[0]
        return np.eye(n) / n
    return psd / tr


def von_neumann_entropy(
    cov: NDArray[np.float64],
) -> EntropyResult:
    """Compute the Von Neumann entropy of a portfolio covariance matrix.

    The matrix is normalised to a density matrix rho = C / Tr(C)
    and the entropy is S(rho) = -Tr[rho log rho].

    Parameters
    ----------
    cov : NDArray, shape (n, n)
        Covariance or correlation matrix.

    Returns
    -------
    EntropyResult
    """
    rho = _to_density_matrix(cov)
    n = rho.shape[0]

    eigvals = np.linalg.eigvalsh(rho)
    eigvals = eigvals[eigvals > 1e-30]

    entropy = -float(np.sum(eigvals * np.log(eigvals)))
    max_entropy = np.log(n) if n > 1 else 1.0
    normalised = entropy / max_entropy if max_entropy > 0 else 0.0
    eff_rank = float(np.exp(entropy))

    return EntropyResult(
        entropy=entropy,
        normalised_entropy=normalised,
        effective_rank=eff_rank,
        eigenvalues=eigvals,
        method="von_neumann",
    )


def quantum_relative_entropy(
    cov_p: NDArray[np.float64],
    cov_q: NDArray[np.float64],
) -> RelativeEntropyResult:
    """Compute quantum relative entropy S(rho || sigma).

    S(rho || sigma) = Tr[rho (log rho - log sigma)].
    Returns infinity when the support of rho is not contained
    in the support of sigma.

    Parameters
    ----------
    cov_p : NDArray, shape (n, n)
        First covariance / correlation matrix (rho).
    cov_q : NDArray, shape (n, n)
        Second covariance / correlation matrix (sigma).

    Returns
    -------
    RelativeEntropyResult
    """
    rho = _to_density_matrix(cov_p)
    sigma = _to_density_matrix(cov_q)

    # Matrix logarithms
    eigvals_r, eigvecs_r = np.linalg.eigh(rho)
    eigvals_s, eigvecs_s = np.linalg.eigh(sigma)

    # Check support condition
    supp_sigma = eigvals_s > 1e-12

    # Project rho onto sigma's support
    proj_sigma = eigvecs_s[:, supp_sigma] @ eigvecs_s[:, supp_sigma].T
    rho_proj = proj_sigma @ rho @ proj_sigma
    trace_loss = abs(np.trace(rho) - np.trace(rho_proj))
    if trace_loss > 1e-8:
        return RelativeEntropyResult(
            divergence=float("inf"),
            is_finite=False,
            symmetrised=float("inf"),
        )

    # Compute log(rho) and log(sigma) via eigendecomposition
    log_rho = _matrix_log(rho, eigvals_r, eigvecs_r)
    log_sigma = _matrix_log(sigma, eigvals_s, eigvecs_s)

    divergence = float(np.real(np.trace(rho @ (log_rho - log_sigma))))
    divergence = max(divergence, 0.0)  # numerical floor

    # Symmetrised version
    log_rho_2 = log_rho
    log_sigma_2 = log_sigma
    reverse = float(np.real(np.trace(sigma @ (log_sigma_2 - log_rho_2))))
    reverse = max(reverse, 0.0)
    symmetrised = (divergence + reverse) / 2.0

    return RelativeEntropyResult(
        divergence=divergence,
        is_finite=True,
        symmetrised=symmetrised,
    )


def _matrix_log(
    M: NDArray[np.float64],
    eigvals: NDArray[np.float64],
    eigvecs: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute matrix logarithm from eigendecomposition."""
    safe_eigvals = np.where(eigvals > 1e-30, eigvals, 1e-30)
    log_eigvals = np.log(safe_eigvals)
    # Zero out contributions from near-zero eigenvalues
    log_eigvals = np.where(eigvals > 1e-30, log_eigvals, 0.0)
    return (eigvecs * log_eigvals) @ eigvecs.T


# ---------------------------------------------------------------------------
# Classical baseline: Shannon entropy
# ---------------------------------------------------------------------------


def shannon_entropy(
    weights: NDArray[np.float64],
) -> EntropyResult:
    """Compute Shannon entropy of a discrete probability / weight vector.

    H(w) = -sum_i w_i log(w_i) for w_i > 0.

    Parameters
    ----------
    weights : NDArray, shape (n,)
        Non-negative weight vector (need not sum to 1).

    Returns
    -------
    EntropyResult
    """
    w = np.asarray(weights, dtype=np.float64)
    w = np.abs(w)
    w_sum = w.sum()
    if w_sum < 1e-30:
        n = len(w)
        return EntropyResult(
            entropy=0.0,
            normalised_entropy=0.0,
            effective_rank=1.0,
            eigenvalues=w,
            method="shannon",
        )

    p = w / w_sum
    nonzero = p > 1e-30
    entropy = -float(np.sum(p[nonzero] * np.log(p[nonzero])))
    n = len(w)
    max_entropy = np.log(n) if n > 1 else 1.0
    normalised = entropy / max_entropy if max_entropy > 0 else 0.0
    eff_rank = float(np.exp(entropy))

    return EntropyResult(
        entropy=entropy,
        normalised_entropy=normalised,
        effective_rank=eff_rank,
        eigenvalues=p,
        method="shannon",
    )


def portfolio_diversification_score(
    cov: NDArray[np.float64],
    weights: NDArray[np.float64] | None = None,
) -> dict[str, float]:
    """Compute diversification scores using both quantum and classical entropy.

    Parameters
    ----------
    cov : NDArray, shape (n, n)
        Covariance matrix.
    weights : NDArray, shape (n,) | None
        Portfolio weights. If None, equal weights are used.

    Returns
    -------
    dict with keys 'von_neumann', 'shannon', 'effective_rank',
    'normalised_vn', 'normalised_shannon'.
    """
    n = cov.shape[0]
    if weights is None:
        weights = np.ones(n) / n

    vn = von_neumann_entropy(cov)
    sh = shannon_entropy(weights)

    return {
        "von_neumann": vn.entropy,
        "shannon": sh.entropy,
        "effective_rank": vn.effective_rank,
        "normalised_vn": vn.normalised_entropy,
        "normalised_shannon": sh.normalised_entropy,
    }
