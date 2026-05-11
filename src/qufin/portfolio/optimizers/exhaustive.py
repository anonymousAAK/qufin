"""Exhaustive (brute-force) QUBO solver for small problems.

Enumerates all 2^n bitstrings and returns the optimal one.
Only practical for n <= 20 qubits.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from itertools import product

import numpy as np
from numpy.typing import NDArray

from qufin.portfolio.qubo import PortfolioQUBO
from qufin.utils.results import Result


@dataclass
class ExhaustiveResult(Result):
    best_bitstring: str = ""
    best_objective: float = float("inf")
    weights: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    n_evaluated: int = 0
    feasible: bool = False
    all_objectives: list[tuple[str, float]] | None = None


def exhaustive_solve(
    qubo: PortfolioQUBO,
    return_all: bool = False,
) -> ExhaustiveResult:
    """Brute-force solve: enumerate all 2^n bitstrings.

    Parameters
    ----------
    qubo : PortfolioQUBO
        The QUBO to solve.
    return_all : bool
        If True, return all (bitstring, objective) pairs in `all_objectives`.

    Raises
    ------
    ValueError
        If n_qubits > 20 (too large for brute force).
    """
    n = qubo.n_qubits
    if n > 20:
        raise ValueError(f"Exhaustive search too large: {n} qubits (max 20)")

    start = time.perf_counter()
    Q = qubo.build_matrix()

    best_bs = ""
    best_obj = float("inf")
    all_objs: list[tuple[str, float]] | None = [] if return_all else None
    n_evaluated = 0

    for bits in product([0, 1], repeat=n):
        x = np.array(bits, dtype=np.float64)
        obj = float(x @ Q @ x)
        bs = "".join(str(b) for b in bits)
        n_evaluated += 1

        if return_all and all_objs is not None:
            all_objs.append((bs, obj))

        if obj < best_obj:
            best_obj = obj
            best_bs = bs

    wall_time = time.perf_counter() - start
    weights = qubo.decode_weights(best_bs)
    feasibility = qubo.feasibility_check(best_bs)

    return ExhaustiveResult(
        value=best_obj,
        wall_time_s=wall_time,
        best_bitstring=best_bs,
        best_objective=best_obj,
        weights=weights,
        n_evaluated=n_evaluated,
        feasible=all(feasibility.values()) if feasibility else True,
        all_objectives=all_objs,
    )
