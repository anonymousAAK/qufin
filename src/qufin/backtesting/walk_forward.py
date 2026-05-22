"""Walk-forward validation utilities.

Monte Carlo permutation test for strategy significance and
combinatorially symmetric cross-validation (CSCV) for detecting
backtest overfitting.

References
----------
White, Econometrica 68(5), 2000 -- Reality Check bootstrap.
Bailey et al., Notices AMS 61(5), 2014 -- CSCV for backtest overfitting.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from numpy.typing import NDArray


def permutation_test(
    strategy_returns: NDArray[np.float64],
    benchmark_returns: NDArray[np.float64],
    n_perms: int = 10_000,
    seed: int | None = 42,
) -> float:
    """Monte Carlo permutation test for strategy significance.

    Tests H0: the strategy has no skill beyond the benchmark.
    Randomly reassigns strategy / benchmark labels and computes the
    fraction of permuted mean-excess-return >= observed.

    Parameters
    ----------
    strategy_returns : NDArray
        Strategy return series.
    benchmark_returns : NDArray
        Benchmark return series (same length).
    n_perms : int
        Number of permutations.
    seed : int | None
        Random seed.

    Returns
    -------
    float
        One-sided p-value (lower is more significant).
    """
    strategy_returns = np.asarray(strategy_returns, dtype=np.float64)
    benchmark_returns = np.asarray(benchmark_returns, dtype=np.float64)
    n = len(strategy_returns)

    observed = float(np.mean(strategy_returns) - np.mean(benchmark_returns))
    pooled = np.stack([strategy_returns, benchmark_returns], axis=0)

    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perms):
        perm = rng.integers(0, 2, size=n)
        perm_strat = np.where(perm == 0, pooled[0], pooled[1])
        perm_bench = np.where(perm == 0, pooled[1], pooled[0])
        if np.mean(perm_strat) - np.mean(perm_bench) >= observed:
            count += 1

    return count / n_perms


@dataclass
class CSCVResult:
    """Result of combinatorially symmetric cross-validation."""

    pbo: float
    logits: NDArray[np.float64]
    n_combinations: int


def cscv_backtest_overfitting(
    returns_matrix: NDArray[np.float64],
    n_partitions: int = 8,
) -> CSCVResult:
    """Combinatorially symmetric cross-validation (CSCV).

    Splits the return matrix into *n_partitions* time blocks, then
    evaluates every S/2-combination of blocks as in-sample vs
    out-of-sample.  Reports the probability of backtest overfitting
    (PBO): the fraction of combinations where the IS-optimal model
    under-performs the median OOS.

    Parameters
    ----------
    returns_matrix : NDArray, shape (T, S)
        T time periods x S strategy variants (e.g. different hyper-
        parameter settings of the same strategy).
    n_partitions : int
        Number of contiguous time blocks (must be even, >= 4).

    Returns
    -------
    CSCVResult
    """
    returns_matrix = np.asarray(returns_matrix, dtype=np.float64)
    T, _S = returns_matrix.shape

    if n_partitions % 2 != 0:
        n_partitions += 1
    n_partitions = max(n_partitions, 4)

    block_size = T // n_partitions
    if block_size < 1:
        return CSCVResult(pbo=0.0, logits=np.array([]), n_combinations=0)

    # Trim to exact multiple
    T_trim = block_size * n_partitions
    returns_matrix = returns_matrix[:T_trim]

    blocks = np.array_split(returns_matrix, n_partitions, axis=0)
    half = n_partitions // 2
    all_indices = list(range(n_partitions))

    logits: list[float] = []
    for is_indices in combinations(all_indices, half):
        oos_indices = [i for i in all_indices if i not in is_indices]

        is_data = np.concatenate([blocks[i] for i in is_indices], axis=0)
        oos_data = np.concatenate([blocks[i] for i in oos_indices], axis=0)

        # Mean return per strategy
        is_mean = np.mean(is_data, axis=0)
        oos_mean = np.mean(oos_data, axis=0)

        best_is = int(np.argmax(is_mean))
        oos_best_perf = oos_mean[best_is]
        oos_median = float(np.median(oos_mean))

        # Logit: positive means OOS-optimal underperforms median
        logit = oos_median - oos_best_perf
        logits.append(logit)

    logits_arr = np.array(logits, dtype=np.float64)
    pbo = float(np.mean(logits_arr > 0)) if len(logits_arr) > 0 else 0.0

    return CSCVResult(
        pbo=pbo,
        logits=logits_arr,
        n_combinations=len(logits_arr),
    )
