"""Warm-start strategies: classical relaxation -> QAOA/VQE seed.

Implements the Egger et al. warm-start approach where a continuous
relaxation (SDP / LP) is solved classically, then used to initialize
quantum variational parameters closer to a good solution.

Also provides CVaR-aware warm starting and multi-start (run from k
different warm-start points, return the best).

References
----------
Egger et al., Quantum 5, 479 (2021) — Warm-starting quantum optimization.
Goemans & Williamson, JACM 42(6), 1995 — SDP relaxation for MAX-CUT.
Barkoutsos et al., Quantum 4, 256 (2020) — CVaR optimization.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from qufin.portfolio.qubo import PortfolioQUBO


@dataclass
class WarmStartResult:
    """Result of warm-start parameter initialization."""

    initial_params: NDArray[np.float64]
    relaxed_solution: NDArray[np.float64]
    rounded_bitstring: str
    relaxed_objective: float


@dataclass
class MultiStartResult:
    """Result of multi-start warm starting."""

    best: WarmStartResult
    all_results: list[WarmStartResult]
    best_index: int


def continuous_relaxation(qubo: PortfolioQUBO) -> NDArray[np.float64]:
    """Solve the continuous relaxation of the QUBO (0 <= x <= 1).

    Minimizes x^T Q x subject to 0 <= x_i <= 1.
    If cardinality is set, also constrains sum(x) = K.
    """
    from scipy.optimize import minimize

    Q = qubo.build_matrix()
    n = Q.shape[0]

    def obj(x: NDArray[np.float64]) -> float:
        return float(x @ Q @ x)

    def grad(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return 2 * Q @ x

    bounds = [(0.0, 1.0)] * n
    constraints = []
    if qubo.cardinality is not None:
        constraints.append({
            "type": "eq",
            "fun": lambda x: np.sum(x) - qubo.cardinality,
        })

    x0 = np.full(n, 0.5)
    if qubo.cardinality is not None:
        x0 = np.full(n, qubo.cardinality / n)

    result = minimize(
        obj,
        x0,
        jac=grad,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    return np.clip(result.x, 0.0, 1.0)


def round_solution(
    x_relaxed: NDArray[np.float64],
    cardinality: int | None = None,
) -> str:
    """Round continuous relaxation to a binary solution.

    Uses deterministic rounding: if cardinality K is set, select the
    top-K variables by relaxed value. Otherwise threshold at 0.5.
    """
    n = len(x_relaxed)
    if cardinality is not None:
        # Select top-K by relaxed value
        top_k = np.argsort(x_relaxed)[-cardinality:]
        bits = np.zeros(n, dtype=int)
        bits[top_k] = 1
    else:
        bits = (x_relaxed >= 0.5).astype(int)
    return "".join(str(b) for b in bits)


def warm_start_qaoa(
    qubo: PortfolioQUBO,
    p: int = 3,
    seed: int | None = 42,
) -> WarmStartResult:
    """Generate warm-start parameters for QAOA from continuous relaxation.

    Solves the continuous relaxation, then maps the solution to initial
    QAOA parameters (gammas, betas) using the Egger et al. heuristic:
    - gammas initialized near 0 (problem unitary close to identity)
    - betas initialized to rotate toward the relaxed solution

    Parameters
    ----------
    qubo : PortfolioQUBO
        The QUBO problem.
    p : int
        Number of QAOA layers.
    seed : int | None
        Random seed for perturbations.

    Returns
    -------
    WarmStartResult
        Contains initial_params (gammas then betas, shape 2*p),
        relaxed_solution, rounded_bitstring, and relaxed_objective.
    """
    rng = np.random.default_rng(seed)

    # Solve continuous relaxation
    x_relaxed = continuous_relaxation(qubo)
    Q = qubo.build_matrix()
    relaxed_obj = float(x_relaxed @ Q @ x_relaxed)

    # Round to bitstring
    rounded_bs = round_solution(x_relaxed, qubo.cardinality)

    # Egger et al. warm-start heuristic:
    # Map relaxed solution to rotation angles.
    # For variables near 0 or 1, the mixer rotation should be small.
    # For variables near 0.5, the mixer should explore more.
    # theta_i = arcsin(sqrt(x_i)) maps [0,1] -> [0, pi/2]
    thetas = np.arcsin(np.sqrt(np.clip(x_relaxed, 0, 1)))
    avg_theta = float(np.mean(thetas))

    # Initialize betas based on average relaxed angle
    # Small betas = stay close to initialization
    betas = np.full(p, avg_theta * 0.3)
    # Add small perturbation for symmetry breaking
    betas += rng.normal(0, 0.05, p)

    # Gammas: small values, scaled by problem energy scale
    energy_scale = float(np.max(np.abs(Q))) if np.max(np.abs(Q)) > 0 else 1.0
    gammas = np.full(p, 0.1 / energy_scale)
    gammas += rng.normal(0, 0.02 / energy_scale, p)

    initial_params = np.concatenate([gammas, betas])

    return WarmStartResult(
        initial_params=initial_params,
        relaxed_solution=x_relaxed,
        rounded_bitstring=rounded_bs,
        relaxed_objective=relaxed_obj,
    )


def warm_start_vqe(
    qubo: PortfolioQUBO,
    n_params: int,
    seed: int | None = 42,
) -> WarmStartResult:
    """Generate warm-start parameters for VQE from continuous relaxation.

    Solves the continuous relaxation, then initializes VQE rotation
    angles to bias the ansatz toward the relaxed solution.

    Parameters
    ----------
    qubo : PortfolioQUBO
        The QUBO problem.
    n_params : int
        Total number of VQE variational parameters.
    seed : int | None
        Random seed.

    Returns
    -------
    WarmStartResult
    """
    rng = np.random.default_rng(seed)

    x_relaxed = continuous_relaxation(qubo)
    Q = qubo.build_matrix()
    relaxed_obj = float(x_relaxed @ Q @ x_relaxed)
    rounded_bs = round_solution(x_relaxed, qubo.cardinality)

    # Map relaxed solution to initial angles
    # theta_i = 2*arcsin(sqrt(x_i)) maps [0,1] -> [0, pi]
    # This biases Ry rotations toward |1> for x_i near 1
    thetas = 2 * np.arcsin(np.sqrt(np.clip(x_relaxed, 0, 1)))

    # Tile the per-qubit angles across all VQE parameters
    n_qubits = len(x_relaxed)
    params = np.zeros(n_params)
    # Fill rotation params with repeated per-qubit bias
    for i in range(n_params):
        qubit_idx = i % n_qubits
        params[i] = thetas[qubit_idx]
    # Add small perturbation for symmetry breaking
    params += rng.normal(0, 0.1, n_params)

    return WarmStartResult(
        initial_params=params,
        relaxed_solution=x_relaxed,
        rounded_bitstring=rounded_bs,
        relaxed_objective=relaxed_obj,
    )


def cvar_warm_start(
    qubo: PortfolioQUBO,
    alpha: float = 0.2,
    p: int = 3,
    seed: int | None = 42,
) -> WarmStartResult:
    """CVaR-aware warm start: solve a CVaR-weighted continuous relaxation.

    Uses the alpha-worst-case samples from a randomized rounding of the
    continuous relaxation to bias QAOA initial parameters toward solutions
    that minimize tail risk.

    Parameters
    ----------
    qubo : PortfolioQUBO
        The QUBO problem.
    alpha : float
        CVaR confidence level in (0, 1]. Smaller = more tail-focused.
    p : int
        Number of QAOA layers.
    seed : int | None
        Random seed.

    Returns
    -------
    WarmStartResult
    """
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")

    rng = np.random.default_rng(seed)
    Q = qubo.build_matrix()
    n = Q.shape[0]

    # Solve continuous relaxation
    x_relaxed = continuous_relaxation(qubo)

    # Generate randomized roundings and keep alpha-worst (highest cost)
    n_samples = 200
    samples: list[tuple[float, NDArray[np.float64]]] = []
    for _ in range(n_samples):
        probs = np.clip(x_relaxed, 0.01, 0.99)
        bits_f = (rng.random(n) < probs).astype(np.float64)
        obj = float(bits_f @ Q @ bits_f)
        samples.append((obj, bits_f))

    samples.sort(key=lambda s: s[0])
    cutoff = max(1, int(n_samples * alpha))
    tail_samples = samples[:cutoff]

    # Average the tail samples to get CVaR-biased relaxed solution
    cvar_relaxed = np.mean([s[1] for s in tail_samples], axis=0)
    cvar_relaxed = np.clip(cvar_relaxed, 0.0, 1.0)
    cvar_obj = float(np.mean([s[0] for s in tail_samples]))

    rounded_bs = round_solution(cvar_relaxed, qubo.cardinality)

    # Map to QAOA parameters (same heuristic as warm_start_qaoa)
    thetas = np.arcsin(np.sqrt(np.clip(cvar_relaxed, 0, 1)))
    avg_theta = float(np.mean(thetas))
    betas = np.full(p, avg_theta * 0.3) + rng.normal(0, 0.05, p)
    energy_scale = float(np.max(np.abs(Q))) if np.max(np.abs(Q)) > 0 else 1.0
    gammas = np.full(p, 0.1 / energy_scale) + rng.normal(0, 0.02 / energy_scale, p)

    return WarmStartResult(
        initial_params=np.concatenate([gammas, betas]),
        relaxed_solution=cvar_relaxed,
        rounded_bitstring=rounded_bs,
        relaxed_objective=cvar_obj,
    )


def multi_start_qaoa(
    qubo: PortfolioQUBO,
    k: int = 5,
    p: int = 3,
    seed: int | None = 42,
) -> MultiStartResult:
    """Run QAOA warm-start from k different starting points.

    Generates k warm-start parameter sets with different random seeds
    and returns the one with the best (lowest) relaxed objective, along
    with all results for analysis.

    Parameters
    ----------
    qubo : PortfolioQUBO
        The QUBO problem.
    k : int
        Number of starting points.
    p : int
        Number of QAOA layers.
    seed : int | None
        Base random seed (each start uses seed + i).

    Returns
    -------
    MultiStartResult
    """
    if k < 1:
        raise ValueError("k must be >= 1")

    results: list[WarmStartResult] = []
    for i in range(k):
        s = (seed + i) if seed is not None else None
        res = warm_start_qaoa(qubo, p=p, seed=s)
        results.append(res)

    # Select the start whose rounded bitstring gives the best QUBO objective
    Q = qubo.build_matrix()
    objectives = []
    for res in results:
        bits = np.array([int(c) for c in res.rounded_bitstring], dtype=float)
        objectives.append(float(bits @ Q @ bits))

    best_idx = int(np.argmin(objectives))

    return MultiStartResult(
        best=results[best_idx],
        all_results=results,
        best_index=best_idx,
    )
