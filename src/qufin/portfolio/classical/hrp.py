"""Hierarchical Risk Parity (Lopez de Prado, 2016).

Tree-based portfolio construction that doesn't require covariance
matrix inversion, making it more robust than mean-variance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform


@dataclass
class HRPResult:
    """Result from HRP allocation."""

    weights: NDArray[np.float64]
    cluster_order: list[int]


def hrp(
    returns: NDArray[np.float64],
    linkage_method: str = "single",
) -> HRPResult:
    """Compute Hierarchical Risk Parity weights.

    Parameters
    ----------
    returns : NDArray
        Return matrix, shape (T, N).
    linkage_method : str
        Linkage method for hierarchical clustering: 'single', 'complete',
        'average', 'ward'.

    Returns
    -------
    HRPResult
    """
    cov = np.cov(returns, rowvar=False)
    corr = np.corrcoef(returns, rowvar=False)
    cov.shape[0]

    # Step 1: Tree clustering
    dist = _correlation_distance(corr)  # type: ignore[arg-type]
    dist_condensed = squareform(dist, checks=False)
    link = linkage(dist_condensed, method=linkage_method)
    sort_idx = list(leaves_list(link))

    # Step 2: Quasi-diagonalization (already done by leaves_list)
    # Step 3: Recursive bisection
    weights = _recursive_bisection(cov, sort_idx)

    return HRPResult(weights=weights, cluster_order=sort_idx)


def _correlation_distance(corr: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert correlation matrix to distance matrix."""
    return np.sqrt(0.5 * (1 - corr))


def _cluster_variance(cov: NDArray[np.float64], indices: list[int]) -> float:
    """Compute inverse-variance portfolio variance for a cluster."""
    sub_cov = cov[np.ix_(indices, indices)]
    ivp = 1.0 / np.diag(sub_cov)
    ivp = ivp / ivp.sum()
    return float(ivp @ sub_cov @ ivp)


def _recursive_bisection(
    cov: NDArray[np.float64], sort_idx: list[int]
) -> NDArray[np.float64]:
    """Recursively bisect the sorted assets and allocate risk."""
    n = cov.shape[0]
    weights = np.ones(n, dtype=np.float64)
    clusters = [sort_idx]

    while clusters:
        new_clusters = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            mid = len(cluster) // 2
            left = cluster[:mid]
            right = cluster[mid:]

            var_left = _cluster_variance(cov, left)
            var_right = _cluster_variance(cov, right)

            alpha = 1.0 - var_left / (var_left + var_right) if (var_left + var_right) > 0 else 0.5

            for i in left:
                weights[i] *= alpha
            for i in right:
                weights[i] *= 1 - alpha

            new_clusters.append(left)
            new_clusters.append(right)

        clusters = new_clusters

    # Normalize
    weights = weights / weights.sum()
    return weights  # type: ignore[no-any-return]
