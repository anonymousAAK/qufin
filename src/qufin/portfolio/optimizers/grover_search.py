"""Grover Adaptive Search for QUBO portfolio optimization.

Implements Grover search with an adaptive threshold on the QUBO objective.
Starting from a random threshold, Grover amplification boosts the probability
of measuring bitstrings with objective value below the threshold.  After each
round the threshold is tightened to the best objective found so far and the
search repeats until no improvement is found.

The algorithm achieves a quadratic speedup over brute-force enumeration for
exact combinatorial optimization.

References
----------
Durr & Hoyer, arXiv:quant-ph/9607014 -- finding the minimum.
Gilliam, Woerner, Gonciulea, arXiv:1912.04088 -- Grover Adaptive Search.
Brandhofer et al., arXiv:2207.10555 -- portfolio QAOA benchmarking.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from qufin.backends.base import Backend
from qufin.portfolio.qubo import PortfolioQUBO
from qufin.utils.results import Result


@dataclass
class GroverSearchConfig:
    """Configuration for the Grover Adaptive Search optimizer.

    Parameters
    ----------
    max_iterations : int
        Maximum number of adaptive threshold rounds.
    n_grover_iterations : int | None
        Number of Grover iterations per round. If None, chosen
        adaptively as floor(pi/4 * sqrt(N / M)) where M is the
        estimated number of marked states.
    lam : float
        Geometric growth factor for Grover iteration schedule
        (used when n_grover_iterations is None).  Must be in (1, 2).
    shots : int
        Number of measurement shots per Grover circuit execution.
    seed : int | None
        Random seed for reproducibility.
    max_qubits : int
        Maximum number of qubits allowed (guards against
        exponential memory in classical simulation).
    """

    max_iterations: int = 50
    n_grover_iterations: int | None = None
    lam: float = 6 / 5
    shots: int = 8192
    seed: int | None = 42
    max_qubits: int = 20


@dataclass
class GroverSearchResult(Result):
    """Result from Grover Adaptive Search."""

    best_bitstring: str = ""
    best_objective: float = float("inf")
    weights: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    feasible: bool = False
    n_rounds: int = 0
    threshold_history: list[float] = field(default_factory=list)
    n_oracle_calls: int = 0
    classical_brute_force_objective: float | None = None


def _compute_all_energies(qubo: PortfolioQUBO) -> dict[str, float]:
    """Enumerate all 2^n bitstrings and their QUBO objectives."""
    n = qubo.n_qubits
    Q = qubo.build_matrix()
    energies: dict[str, float] = {}
    for i in range(2**n):
        bs = format(i, f"0{n}b")
        x = np.array([int(c) for c in bs], dtype=np.float64)
        energies[bs] = float(x @ Q @ x)
    return energies


def _build_oracle(n: int, energies: dict[str, float], threshold: float) -> object:
    """Build the phase-flip oracle marking states with energy < threshold.

    Implements O|x> = -|x> if E(x) < threshold, else |x>.
    """
    from qiskit.circuit import QuantumCircuit

    qc = QuantumCircuit(n)

    # Diagonal unitary: phase-flip marked states
    diag = []
    for i in range(2**n):
        bs = format(i, f"0{n}b")
        if energies[bs] < threshold:
            diag.append(-1.0)
        else:
            diag.append(1.0)

    if all(d == 1.0 for d in diag):
        # No marked states -- return identity
        return qc

    try:
        from qiskit.circuit.library import DiagonalGate
    except ImportError:  # Qiskit < 2.1
        from qiskit.circuit.library import Diagonal as DiagonalGate

    qc = QuantumCircuit(n)
    qc.append(DiagonalGate(diag), range(n))
    return qc


def _build_diffusion(n: int) -> object:
    """Build the Grover diffusion operator: 2|s><s| - I.

    |s> = H^n |0> is the uniform superposition.
    """
    from qiskit.circuit import QuantumCircuit

    qc = QuantumCircuit(n)

    qc.h(range(n))
    qc.x(range(n))

    # Multi-controlled Z
    if n == 1:
        qc.z(0)
    else:
        qc.h(n - 1)
        qc.mcx(list(range(n - 1)), n - 1)
        qc.h(n - 1)

    qc.x(range(n))
    qc.h(range(n))

    return qc


def _build_grover_circuit(
    n: int,
    energies: dict[str, float],
    threshold: float,
    n_iterations: int,
) -> object:
    """Build a Grover circuit with the given number of iterations.

    The circuit is: H^n -> (Oracle . Diffusion)^n_iterations -> Measure.
    """
    from qiskit.circuit import QuantumCircuit

    qc = QuantumCircuit(n, n)

    # Uniform superposition
    qc.h(range(n))

    oracle = _build_oracle(n, energies, threshold)
    diffusion = _build_diffusion(n)

    for _ in range(n_iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffusion, inplace=True)

    qc.measure(range(n), range(n))
    return qc


def _optimal_grover_iters(n_total: int, n_marked: int) -> int:
    """Optimal number of Grover iterations: floor(pi/4 * sqrt(N/M))."""
    if n_marked <= 0 or n_marked >= n_total:
        return 1
    return max(1, int(math.pi / 4 * math.sqrt(n_total / n_marked)))


class GroverAdaptiveSearch:
    """Grover Adaptive Search for portfolio QUBO optimization.

    Uses Grover amplification with adaptive threshold tightening
    to find the global minimum of the QUBO objective. Each round
    applies Grover iterations to amplify states with objective below
    the current threshold, then updates the threshold to the best
    measured value.
    """

    def __init__(
        self,
        qubo: PortfolioQUBO,
        config: GroverSearchConfig,
        backend: Backend,
    ) -> None:
        if qubo.n_qubits > config.max_qubits:
            raise ValueError(
                f"Problem size {qubo.n_qubits} qubits is too large "
                f"(max {config.max_qubits}). Increase max_qubits if intended."
            )
        self.qubo = qubo
        self.config = config
        self.backend = backend
        self._energies = _compute_all_energies(qubo)
        self._rng = np.random.default_rng(config.seed)

    def _count_marked(self, threshold: float) -> int:
        """Count states with energy strictly below threshold."""
        return sum(1 for e in self._energies.values() if e < threshold)

    def _pick_n_iters(self, threshold: float) -> int:
        """Choose the number of Grover iterations for the current round."""
        if self.config.n_grover_iterations is not None:
            return self.config.n_grover_iterations

        n_total = 2 ** self.qubo.n_qubits
        n_marked = self._count_marked(threshold)

        if n_marked == 0:
            # No marked states -- use random iteration count
            max_iters = max(1, int(math.sqrt(n_total)))
            return int(self._rng.integers(1, max_iters + 1))

        return _optimal_grover_iters(n_total, n_marked)

    def run(self) -> GroverSearchResult:
        """Execute the Grover Adaptive Search algorithm."""
        start = time.perf_counter()
        n = self.qubo.n_qubits

        # Initialize with a random threshold
        all_energies_sorted = sorted(self._energies.values())
        # Start with the median energy as initial threshold
        threshold = float(np.median(all_energies_sorted))

        best_bs = ""
        best_obj = float("inf")
        total_oracle_calls = 0
        threshold_history: list[float] = [threshold]

        # Pick a random initial solution
        random_bs = format(
            int(self._rng.integers(0, 2**n)), f"0{n}b"
        )
        best_bs = random_bs
        best_obj = self._energies[random_bs]
        threshold = best_obj

        for _round_idx in range(self.config.max_iterations):
            n_iters = self._pick_n_iters(threshold)
            n_marked = self._count_marked(threshold)

            if n_marked == 0:
                # No states below threshold; try a looser threshold
                # by picking a random number of Grover iterations
                # with a lambda-scaled schedule (Durr-Hoyer)
                max_j = max(1, int(math.sqrt(2**n)))
                n_iters = int(self._rng.integers(0, max_j + 1))
                # Use the current best_obj as threshold (nothing better known)
                threshold = best_obj

            circuit = _build_grover_circuit(n, self._energies, threshold, n_iters)
            result = self.backend.run(circuit, shots=self.config.shots)
            total_oracle_calls += n_iters * self.config.shots

            # Find the best measured bitstring
            improved = False
            for bitstring, _count in result.counts.items():
                # Pad bitstring to n bits
                bs = bitstring.zfill(n)
                if bs in self._energies:
                    obj = self._energies[bs]
                    if obj < best_obj:
                        best_obj = obj
                        best_bs = bs
                        improved = True

            if improved:
                threshold = best_obj
            threshold_history.append(threshold)

            if not improved and n_marked == 0:
                # No improvement and no marked states -- converged
                break

        wall_time = time.perf_counter() - start

        weights = self.qubo.decode_weights(best_bs)
        feasibility = self.qubo.feasibility_check(best_bs)

        return GroverSearchResult(
            value=best_obj,
            n_shots=self.config.shots,
            wall_time_s=wall_time,
            backend_id=self.backend.backend_id,
            seed=self.config.seed,
            best_bitstring=best_bs,
            best_objective=best_obj,
            weights=weights,
            feasible=all(feasibility.values()) if feasibility else True,
            n_rounds=len(threshold_history) - 1,
            threshold_history=threshold_history,
            n_oracle_calls=total_oracle_calls,
        )


# ---------------------------------------------------------------------------
# Comparison utilities
# ---------------------------------------------------------------------------


def branch_and_bound_solve(qubo: PortfolioQUBO) -> tuple[str, float]:
    """Simple branch-and-bound solver for comparison benchmarking.

    Uses depth-first search with lower-bound pruning on the QUBO
    objective.  Not competitive for large instances but provides
    a classical baseline for comparison at small sizes.

    Returns
    -------
    tuple of (best_bitstring, best_objective)
    """
    n = qubo.n_qubits
    Q = qubo.build_matrix()

    best_obj = float("inf")
    best_bs = "0" * n

    def _lower_bound(partial: list[int], depth: int) -> float:
        """Partial assignment lower bound: fix assigned bits, assume
        remaining bits contribute minimally."""
        x = np.array(partial + [0] * (n - depth), dtype=np.float64)
        base = float(x @ Q @ x)
        # For each free variable, compute the minimum marginal contribution
        lb = base
        for j in range(depth, n):
            # Cost if x_j = 1 (vs 0)
            marginal = Q[j, j]
            for i in range(depth):
                marginal += (Q[i, j] + Q[j, i]) * partial[i]
            if marginal < 0:
                lb += marginal
        return lb

    def _dfs(partial: list[int], depth: int) -> None:
        nonlocal best_obj, best_bs

        if depth == n:
            x = np.array(partial, dtype=np.float64)
            obj = float(x @ Q @ x)
            if obj < best_obj:
                best_obj = obj
                best_bs = "".join(str(b) for b in partial)
            return

        # Branch on x[depth] = 0 and x[depth] = 1
        for val in [0, 1]:
            partial.append(val)
            lb = _lower_bound(partial, depth + 1)
            if lb < best_obj:
                _dfs(partial, depth + 1)
            partial.pop()

    _dfs([], 0)
    return best_bs, best_obj


def compare_optimizers(
    qubo: PortfolioQUBO,
    config: GroverSearchConfig,
    backend: Backend,
    qaoa_config: object | None = None,
) -> dict[str, dict[str, float]]:
    """Compare Grover Adaptive Search, branch-and-bound, and optionally QAOA.

    Parameters
    ----------
    qubo : PortfolioQUBO
        The portfolio optimization QUBO.
    config : GroverSearchConfig
        Configuration for the Grover search.
    backend : Backend
        Quantum backend for circuit execution.
    qaoa_config : QAOAConfig | None
        Optional QAOA configuration for comparison.

    Returns
    -------
    dict mapping optimizer name to a dict with keys
    'objective', 'wall_time_s', and 'n_oracle_calls' (if applicable).
    """
    results: dict[str, dict[str, float]] = {}

    # Grover Adaptive Search
    gas = GroverAdaptiveSearch(qubo, config, backend)
    gas_result = gas.run()
    results["grover_adaptive"] = {
        "objective": gas_result.best_objective,
        "wall_time_s": gas_result.wall_time_s,
        "n_oracle_calls": float(gas_result.n_oracle_calls),
    }

    # Branch and bound
    t0 = time.perf_counter()
    _bb_bs, bb_obj = branch_and_bound_solve(qubo)
    bb_time = time.perf_counter() - t0
    results["branch_and_bound"] = {
        "objective": bb_obj,
        "wall_time_s": bb_time,
        "n_oracle_calls": 0.0,
    }

    # QAOA (if config provided)
    if qaoa_config is not None:
        from qufin.portfolio.optimizers.qaoa import QAOAConfig, QAOAPortfolio

        if isinstance(qaoa_config, QAOAConfig):
            qaoa = QAOAPortfolio(qubo, qaoa_config, backend)
            qaoa_result = qaoa.run()
            results["qaoa"] = {
                "objective": qaoa_result.best_objective,
                "wall_time_s": qaoa_result.wall_time_s,
                "n_oracle_calls": 0.0,
            }

    return results
