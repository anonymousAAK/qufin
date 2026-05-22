"""Simulated Quantum Annealing for cardinality-constrained portfolio optimization.

Formulates the Markowitz mean-variance objective with a cardinality constraint
as a QUBO and solves it using a simulated quantum annealing schedule.  The
transverse-field term decays exponentially over the anneal, mimicking the
quantum tunnelling effect that allows escape from local minima.

A classical random-restart hill-climbing baseline is included for comparison.

References
----------
Kadowaki & Nishimori, Phys. Rev. E 58, 5355 (1998).
Finnila et al., Chem. Phys. Lett. 219, 343 (1994).
Brandhofer et al., arXiv:2207.10555 -- portfolio QAOA benchmarking.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from qufin.backends.base import Backend
from qufin.portfolio.qubo import PortfolioQUBO
from qufin.utils.results import Result


@dataclass
class SQAConfig:
    """Configuration for Simulated Quantum Annealing.

    Parameters
    ----------
    n_sweeps : int
        Number of Monte Carlo sweeps (outer iterations).
    n_replicas : int
        Number of Suzuki-Trotter replicas (imaginary-time slices).
    beta : float
        Inverse temperature.
    gamma_start : float
        Initial transverse-field strength.
    gamma_end : float
        Final transverse-field strength.
    seed : int | None
        Random seed for reproducibility.
    """

    n_sweeps: int = 200
    n_replicas: int = 8
    beta: float = 2.0
    gamma_start: float = 3.0
    gamma_end: float = 0.01
    seed: int | None = 42


@dataclass
class SQAResult(Result):
    """Result from Simulated Quantum Annealing."""

    best_bitstring: str = ""
    best_objective: float = float("inf")
    weights: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    feasible: bool = False
    energy_history: list[float] = field(default_factory=list)
    n_sweeps_completed: int = 0


def _qubo_energy(x: NDArray[np.float64], Q: NDArray[np.float64]) -> float:
    """Compute QUBO energy x^T Q x."""
    return float(x @ Q @ x)


def simulated_quantum_annealing(
    qubo: PortfolioQUBO,
    config: SQAConfig,
    backend: Backend | None = None,
) -> SQAResult:
    """Run simulated quantum annealing on a portfolio QUBO.

    Uses the Suzuki-Trotter decomposition of the transverse-field
    Ising model.  Replicas are coupled along the imaginary-time
    axis; the transverse field decays from ``gamma_start`` to
    ``gamma_end`` on an exponential schedule.

    Parameters
    ----------
    qubo : PortfolioQUBO
        Portfolio QUBO formulation.
    config : SQAConfig
        Annealing schedule and parameters.
    backend : Backend | None
        Optional quantum backend (stored in result metadata).

    Returns
    -------
    SQAResult
    """
    start = time.perf_counter()
    rng = np.random.default_rng(config.seed)
    n = qubo.n_qubits
    Q = qubo.build_matrix()
    R = config.n_replicas

    # Initialise replicas randomly
    spins = rng.integers(0, 2, size=(R, n)).astype(np.float64)

    best_obj = float("inf")
    best_bs = "0" * n
    energy_history: list[float] = []

    for sweep in range(config.n_sweeps):
        # Exponential decay of transverse field
        progress = sweep / max(config.n_sweeps - 1, 1)
        gamma = config.gamma_start * (config.gamma_end / config.gamma_start) ** progress
        J_perp = -0.5 * config.beta / R * np.log(np.tanh(gamma * config.beta / R + 1e-30))

        for r in range(R):
            for i in rng.permutation(n):
                # Classical energy change for flipping spin i
                x = spins[r].copy()
                e_before = _qubo_energy(x, Q)
                x[i] = 1.0 - x[i]
                e_after = _qubo_energy(x, Q)
                delta_classical = (e_after - e_before) / R

                # Coupling to adjacent replicas
                r_prev = (r - 1) % R
                r_next = (r + 1) % R
                s_i = 2.0 * spins[r, i] - 1.0  # {0,1} -> {-1,+1}
                s_prev = 2.0 * spins[r_prev, i] - 1.0
                s_next = 2.0 * spins[r_next, i] - 1.0
                delta_coupling = 2.0 * J_perp * s_i * (s_prev + s_next)

                delta_total = delta_classical + delta_coupling

                # Metropolis acceptance
                if delta_total <= 0 or rng.random() < np.exp(-config.beta * delta_total):
                    spins[r, i] = 1.0 - spins[r, i]

        # Track best across replicas
        for r in range(R):
            obj = _qubo_energy(spins[r], Q)
            if obj < best_obj:
                best_obj = obj
                best_bs = "".join(str(int(b)) for b in spins[r])
        energy_history.append(best_obj)

    wall_time = time.perf_counter() - start
    weights = qubo.decode_weights(best_bs)
    feasibility = qubo.feasibility_check(best_bs)

    return SQAResult(
        value=best_obj,
        wall_time_s=wall_time,
        backend_id=backend.backend_id if backend else "classical_sqa",
        seed=config.seed,
        best_bitstring=best_bs,
        best_objective=best_obj,
        weights=weights,
        feasible=all(feasibility.values()) if feasibility else True,
        energy_history=energy_history,
        n_sweeps_completed=config.n_sweeps,
    )


# ---------------------------------------------------------------------------
# Classical baseline: random-restart hill climbing
# ---------------------------------------------------------------------------


def random_restart_hill_climbing(
    qubo: PortfolioQUBO,
    n_restarts: int = 50,
    max_steps: int = 200,
    seed: int | None = 42,
) -> SQAResult:
    """Classical random-restart hill climbing for comparison.

    Parameters
    ----------
    qubo : PortfolioQUBO
        Portfolio QUBO formulation.
    n_restarts : int
        Number of random restarts.
    max_steps : int
        Maximum single-bit-flip steps per restart.
    seed : int | None
        Random seed.

    Returns
    -------
    SQAResult
    """
    start = time.perf_counter()
    rng = np.random.default_rng(seed)
    n = qubo.n_qubits
    Q = qubo.build_matrix()

    best_obj = float("inf")
    best_bs = "0" * n
    energy_history: list[float] = []

    for _ in range(n_restarts):
        x = rng.integers(0, 2, size=n).astype(np.float64)
        obj = _qubo_energy(x, Q)

        for _step in range(max_steps):
            improved = False
            for i in rng.permutation(n):
                x[i] = 1.0 - x[i]
                new_obj = _qubo_energy(x, Q)
                if new_obj < obj:
                    obj = new_obj
                    improved = True
                else:
                    x[i] = 1.0 - x[i]
            if not improved:
                break

        if obj < best_obj:
            best_obj = obj
            best_bs = "".join(str(int(b)) for b in x)
        energy_history.append(best_obj)

    wall_time = time.perf_counter() - start
    weights = qubo.decode_weights(best_bs)
    feasibility = qubo.feasibility_check(best_bs)

    return SQAResult(
        value=best_obj,
        wall_time_s=wall_time,
        backend_id="classical_hill_climb",
        seed=seed,
        best_bitstring=best_bs,
        best_objective=best_obj,
        weights=weights,
        feasible=all(feasibility.values()) if feasibility else True,
        energy_history=energy_history,
        n_sweeps_completed=n_restarts,
    )
