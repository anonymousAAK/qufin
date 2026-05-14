"""Szegedy quantum walk optimizer for portfolio optimization.

Implements a discrete-time quantum walk on the portfolio state graph,
where vertices represent candidate portfolios (bitstrings) and transition
probabilities are derived from the QUBO objective via a Boltzmann-like
weighting. Low-energy states are marked and amplified through the quantum
walk search framework.

References
----------
Szegedy, FOCS 2004 -- Quantum speed-up of Markov chain based algorithms.
Magniez et al., SIAM J. Comput. 40(4):1220-1240, 2011 -- Search via quantum walk.
Portugal, Quantum Walks and Search Algorithms, Springer (2013).
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
class SzegedyWalkConfig:
    """Configuration for the Szegedy quantum walk optimizer.

    Parameters
    ----------
    n_walk_steps : int
        Number of quantum walk iterations (analogous to Grover iterations).
    temperature : float
        Boltzmann temperature for converting QUBO energies to transition
        probabilities. Lower values sharpen the distribution toward
        low-energy states.
    energy_threshold : float | None
        States with energy below this threshold are marked. If None,
        the median energy across all states is used.
    shots : int
        Number of measurement shots for the final circuit.
    seed : int | None
        Random seed for reproducibility.
    """

    n_walk_steps: int = 3
    temperature: float = 1.0
    energy_threshold: float | None = None
    shots: int = 8192
    seed: int | None = 42


@dataclass
class SzegedyWalkResult(Result):
    """Result from Szegedy quantum walk portfolio optimization."""

    best_bitstring: str = ""
    best_objective: float = float("inf")
    weights: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    walk_unitary_dim: int = 0
    n_walk_steps: int = 0
    marked_count: int = 0
    feasible: bool = False


def _enumerate_bitstrings(n: int) -> list[str]:
    """Generate all n-bit bitstrings in lexicographic order."""
    return [format(i, f"0{n}b") for i in range(2**n)]


def compute_qubo_energies(qubo: PortfolioQUBO) -> dict[str, float]:
    """Evaluate QUBO objective for every bitstring.

    Parameters
    ----------
    qubo : PortfolioQUBO
        Portfolio QUBO formulation.

    Returns
    -------
    dict mapping bitstring -> objective value.
    """
    n = qubo.n_qubits
    Q = qubo.build_matrix()
    energies: dict[str, float] = {}
    for bs in _enumerate_bitstrings(n):
        x = np.array([int(c) for c in bs], dtype=np.float64)
        energies[bs] = float(x @ Q @ x)
    return energies


def build_transition_matrix(
    energies: dict[str, float],
    temperature: float = 1.0,
) -> NDArray[np.float64]:
    """Build a doubly-stochastic transition matrix from QUBO energies.

    Uses a Metropolis-Hastings-like construction: the proposal distribution
    is uniform over all states, and acceptance is governed by Boltzmann
    weights. The resulting matrix is then symmetrized to ensure double
    stochasticity.

    Parameters
    ----------
    energies : dict[str, float]
        Mapping from bitstring to objective value.
    temperature : float
        Boltzmann temperature. Lower values make the walk prefer
        lower-energy states more strongly.

    Returns
    -------
    NDArray of shape (N, N) -- doubly-stochastic transition matrix P
        where P[i, j] is the probability of transitioning from state i to j.
    """
    bitstrings = sorted(energies.keys())
    N = len(bitstrings)
    e_vals = np.array([energies[bs] for bs in bitstrings])

    # Boltzmann weights
    # Shift energies to avoid overflow: subtract minimum
    e_shifted = e_vals - e_vals.min()
    # Guard against zero temperature
    temp = max(temperature, 1e-12)
    boltzmann = np.exp(-e_shifted / temp)

    # Build Metropolis-Hastings transition matrix
    P = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            # Acceptance probability: min(1, pi_j / pi_i)
            ratio = boltzmann[j] / max(boltzmann[i], 1e-30)
            P[i, j] = min(1.0, ratio) / (N - 1)

        # Self-loop: remaining probability
        P[i, i] = 1.0 - P[i, :].sum() + P[i, i]

    # Symmetrize to make doubly stochastic via Sinkhorn iteration:
    # alternately normalize rows and columns until convergence.
    P_ds = (P + P.T) / 2.0
    for _ in range(100):
        # Normalize rows
        row_sums = P_ds.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        P_ds = P_ds / row_sums
        # Normalize columns
        col_sums = P_ds.sum(axis=0, keepdims=True)
        col_sums = np.where(col_sums > 0, col_sums, 1.0)
        P_ds = P_ds / col_sums

    # Final row normalization for exact stochasticity
    row_sums = P_ds.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    P_ds = P_ds / row_sums

    return P_ds


def build_walk_unitary(P: NDArray[np.float64]) -> NDArray[np.complex128]:
    """Build the Szegedy walk unitary from a doubly-stochastic matrix.

    The Szegedy walk operator acts on a doubled Hilbert space
    H_A x H_B (each of dimension N). The walk operator is W = S * (2*Pi - I)
    where Pi projects onto the column space of |psi_x> states, and S is
    the swap operator.

    For computational tractability this constructs the full unitary matrix
    of dimension N^2 x N^2.

    Parameters
    ----------
    P : NDArray of shape (N, N)
        Doubly-stochastic transition matrix.

    Returns
    -------
    NDArray of shape (N^2, N^2) -- the walk unitary W.
    """
    N = P.shape[0]
    dim = N * N

    # Build the |psi_x> states: |psi_x> = |x> (sum_y sqrt(P[x,y]) |y>)
    # in the NxN product space, |psi_x> has nonzero entries at positions x*N+y
    sqrt_P_elem = np.sqrt(np.maximum(P, 0.0))

    # Projector Pi = sum_x |psi_x><psi_x|
    Pi = np.zeros((dim, dim), dtype=np.complex128)
    for x in range(N):
        psi_x = np.zeros(dim, dtype=np.complex128)
        for y in range(N):
            psi_x[x * N + y] = sqrt_P_elem[x, y]
        # Normalize (should already be normalized if P is stochastic)
        norm = np.linalg.norm(psi_x)
        if norm > 1e-12:
            psi_x /= norm
        Pi += np.outer(psi_x, psi_x.conj())

    # Reflection: R = 2*Pi - I
    R = 2.0 * Pi - np.eye(dim, dtype=np.complex128)

    # Swap operator S: S|x,y> = |y,x>
    S = np.zeros((dim, dim), dtype=np.complex128)
    for x in range(N):
        for y in range(N):
            S[y * N + x, x * N + y] = 1.0

    # Walk operator: W = S @ R
    W = S @ R

    return W


def build_marked_oracle(
    energies: dict[str, float],
    threshold: float,
) -> NDArray[np.complex128]:
    """Build a phase oracle that marks low-energy states.

    Applies a phase flip (-1) to any basis state |x,y> where x is a
    marked (low-energy) state.

    Parameters
    ----------
    energies : dict[str, float]
        Mapping from bitstring to objective value.
    threshold : float
        States with energy <= threshold are marked.

    Returns
    -------
    NDArray of shape (N^2, N^2) -- diagonal oracle matrix.
    """
    bitstrings = sorted(energies.keys())
    N = len(bitstrings)
    dim = N * N

    oracle = np.eye(dim, dtype=np.complex128)
    for idx, bs in enumerate(bitstrings):
        if energies[bs] <= threshold:
            # Mark all |x, y> where x = idx
            for y in range(N):
                oracle[idx * N + y, idx * N + y] = -1.0

    return oracle


def szegedy_walk_search(
    qubo: PortfolioQUBO,
    config: SzegedyWalkConfig,
) -> tuple[NDArray[np.complex128], list[str], dict[str, float]]:
    """Execute Szegedy walk search via matrix simulation.

    Constructs the walk operator and oracle, then applies walk steps
    to amplify marked (low-energy) states.

    Parameters
    ----------
    qubo : PortfolioQUBO
        Portfolio QUBO formulation.
    config : SzegedyWalkConfig
        Walk configuration.

    Returns
    -------
    tuple of (final_state, bitstrings, energies)
        final_state : statevector after walk iterations
        bitstrings : ordered list of bitstrings (vertex labels)
        energies : dict of bitstring -> objective value
    """
    energies = compute_qubo_energies(qubo)
    bitstrings = sorted(energies.keys())
    N = len(bitstrings)

    # Build transition matrix
    P = build_transition_matrix(energies, config.temperature)

    # Build walk unitary
    W = build_walk_unitary(P)

    # Determine threshold
    e_vals = np.array([energies[bs] for bs in bitstrings])
    if config.energy_threshold is not None:
        threshold = config.energy_threshold
    else:
        threshold = float(np.median(e_vals))

    # Build oracle
    oracle = build_marked_oracle(energies, threshold)

    # Initial state: uniform superposition over |psi_x> states
    dim = N * N
    sqrt_P_elem = np.sqrt(np.maximum(P, 0.0))
    state = np.zeros(dim, dtype=np.complex128)
    for x in range(N):
        for y in range(N):
            state[x * N + y] = sqrt_P_elem[x, y]
    norm = np.linalg.norm(state)
    if norm > 1e-12:
        state /= norm

    # Apply walk steps: each step is Oracle @ Walk
    for _ in range(config.n_walk_steps):
        state = oracle @ state
        state = W @ state

    return state, bitstrings, energies


def classical_random_walk(
    qubo: PortfolioQUBO,
    n_steps: int = 1000,
    temperature: float = 1.0,
    seed: int | None = 42,
) -> tuple[str, float, list[float]]:
    """Classical random walk baseline on the same Markov chain.

    Performs a classical Metropolis-Hastings walk on the QUBO energy
    landscape and tracks the best state found.

    Parameters
    ----------
    qubo : PortfolioQUBO
        Portfolio QUBO formulation.
    n_steps : int
        Number of random walk steps.
    temperature : float
        Boltzmann temperature for acceptance probabilities.
    seed : int | None
        Random seed.

    Returns
    -------
    tuple of (best_bitstring, best_energy, energy_trace)
    """
    rng = np.random.default_rng(seed)
    n = qubo.n_qubits
    Q = qubo.build_matrix()

    # Start at random bitstring
    current = rng.integers(0, 2, size=n)
    x_curr = current.astype(np.float64)
    e_curr = float(x_curr @ Q @ x_curr)

    best_bs = "".join(str(b) for b in current)
    best_energy = e_curr
    trace: list[float] = [e_curr]

    temp = max(temperature, 1e-12)

    for _ in range(n_steps):
        # Propose: flip one random bit
        proposal = current.copy()
        flip_idx = rng.integers(0, n)
        proposal[flip_idx] = 1 - proposal[flip_idx]

        x_prop = proposal.astype(np.float64)
        e_prop = float(x_prop @ Q @ x_prop)

        # Metropolis acceptance
        delta = e_prop - e_curr
        if delta <= 0 or rng.random() < np.exp(-delta / temp):
            current = proposal
            e_curr = e_prop

        trace.append(e_curr)

        bs = "".join(str(b) for b in current)
        if e_curr < best_energy:
            best_energy = e_curr
            best_bs = bs

    return best_bs, best_energy, trace


class SzegedyWalkOptimizer:
    """Szegedy quantum walk optimizer for portfolio QUBO problems.

    Uses a discrete-time quantum walk on the portfolio state graph to
    search for low-energy portfolio configurations. The walk operates
    on a Markov chain whose stationary distribution is biased toward
    low-energy states via Boltzmann weighting.

    Parameters
    ----------
    qubo : PortfolioQUBO
        Portfolio QUBO formulation.
    config : SzegedyWalkConfig
        Walk optimizer configuration.
    backend : Backend
        Quantum backend (used for metadata; walk is simulated via
        matrix operations for small instances).
    """

    def __init__(
        self,
        qubo: PortfolioQUBO,
        config: SzegedyWalkConfig,
        backend: Backend,
    ) -> None:
        self.qubo = qubo
        self.config = config
        self.backend = backend

        n = qubo.n_qubits
        if n > 10:
            raise ValueError(
                f"Szegedy walk requires 2^(2n) memory; n={n} is too large. "
                f"Use n <= 10."
            )

    def run(self) -> SzegedyWalkResult:
        """Run the Szegedy walk optimizer and return the best portfolio.

        Returns
        -------
        SzegedyWalkResult
            Optimization result including best bitstring, objective,
            and decoded portfolio weights.
        """
        start = time.perf_counter()

        state, bitstrings, energies = szegedy_walk_search(
            self.qubo, self.config,
        )

        N = len(bitstrings)

        # Extract marginal probabilities over the first register
        probs = np.abs(state) ** 2
        marginal = np.zeros(N, dtype=np.float64)
        for x in range(N):
            for y in range(N):
                marginal[x] += probs[x * N + y]

        # Determine threshold for counting marked states
        e_vals = np.array([energies[bs] for bs in bitstrings])
        if self.config.energy_threshold is not None:
            threshold = self.config.energy_threshold
        else:
            threshold = float(np.median(e_vals))

        marked_count = sum(1 for bs in bitstrings if energies[bs] <= threshold)

        # Find best state: highest marginal probability among candidates,
        # then break ties by energy
        best_idx = int(np.argmax(marginal))
        best_bs = bitstrings[best_idx]
        best_obj = energies[best_bs]

        # Also check: among top-k most probable states, pick lowest energy
        top_k = min(N, max(5, marked_count))
        top_indices = np.argsort(marginal)[::-1][:top_k]
        for idx in top_indices:
            bs = bitstrings[idx]
            if energies[bs] < best_obj:
                best_obj = energies[bs]
                best_bs = bs

        weights = self.qubo.decode_weights(best_bs)
        feasibility = self.qubo.feasibility_check(best_bs)
        wall_time = time.perf_counter() - start

        return SzegedyWalkResult(
            value=best_obj,
            n_shots=self.config.shots,
            wall_time_s=wall_time,
            backend_id=self.backend.backend_id,
            seed=self.config.seed,
            best_bitstring=best_bs,
            best_objective=best_obj,
            weights=weights,
            walk_unitary_dim=N * N,
            n_walk_steps=self.config.n_walk_steps,
            marked_count=marked_count,
            feasible=all(feasibility.values()) if feasibility else True,
        )
