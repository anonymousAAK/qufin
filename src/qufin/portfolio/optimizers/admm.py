"""ADMM-based QUBO decomposition for large portfolio problems.

Splits a large QUBO (>50 assets) into sub-problems of manageable size
and solves them iteratively via the Alternating Direction Method of
Multipliers (ADMM) with consensus constraints.

References
----------
Boyd et al., "Distributed Optimization and Statistical Learning via
    the Alternating Direction Method of Multipliers," Found. & Trends
    in Machine Learning 3(1):1-122 (2011).
Gambella et al., "Multi-block ADMM Heuristics for Mixed-Binary
    Optimization on Classical and Quantum Computers," IEEE Trans.
    Quantum Eng. 1:1-15 (2020).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from itertools import product

import numpy as np
from numpy.typing import NDArray

from qufin.backends.base import Backend
from qufin.portfolio.qubo import PortfolioQUBO
from qufin.utils.results import Result


@dataclass
class ADMMConfig:
    """Configuration for the ADMM portfolio optimizer.

    Parameters
    ----------
    sub_problem_size : int
        Maximum number of qubits per sub-problem.
    max_iterations : int
        Maximum number of ADMM outer iterations.
    rho : float
        Augmented Lagrangian penalty parameter.
    rho_update : bool
        Whether to adaptively update rho based on residual balance.
    tol_primal : float
        Convergence tolerance for the primal residual.
    tol_dual : float
        Convergence tolerance for the dual residual.
    sub_solver : str
        Solver for sub-problems: ``"qaoa"`` or ``"exhaustive"``.
    qaoa_depth : int
        QAOA circuit depth (p parameter) when using the QAOA sub-solver.
    shots : int
        Number of measurement shots per sub-problem evaluation.
    seed : int
        Random seed for reproducibility.
    """

    sub_problem_size: int = 20
    max_iterations: int = 50
    rho: float = 1.0
    rho_update: bool = True
    tol_primal: float = 1e-4
    tol_dual: float = 1e-4
    sub_solver: str = "qaoa"
    qaoa_depth: int = 2
    shots: int = 4096
    seed: int = 42


@dataclass
class ADMMResult(Result):
    """Result of the ADMM portfolio optimization.

    Attributes
    ----------
    best_bitstring : str
        Best binary solution found.
    weights : NDArray[np.float64]
        Decoded portfolio weights from the best bitstring.
    objective : float
        QUBO objective value of the best solution.
    n_iterations : int
        Number of ADMM iterations executed.
    primal_residuals : list[float]
        Primal residual history across iterations.
    dual_residuals : list[float]
        Dual residual history across iterations.
    sub_problem_objectives : list[list[float]]
        Per-iteration, per-sub-problem objective values.
    converged : bool
        Whether ADMM converged within tolerance.
    """

    best_bitstring: str = ""
    weights: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    objective: float = float("inf")
    n_iterations: int = 0
    primal_residuals: list[float] = field(default_factory=list)
    dual_residuals: list[float] = field(default_factory=list)
    sub_problem_objectives: list[list[float]] = field(default_factory=list)
    converged: bool = False


class ADMMPortfolio:
    """ADMM solver that decomposes large portfolio QUBOs into sub-problems.

    The full QUBO variable vector is partitioned into blocks of at most
    ``config.sub_problem_size`` variables.  Each ADMM iteration solves
    the blocks independently (x-update), enforces consensus (z-update),
    and adjusts dual variables (u-update).  The penalty parameter rho
    can be adapted automatically to balance primal and dual convergence.

    Parameters
    ----------
    qubo : PortfolioQUBO
        The portfolio QUBO to solve.
    config : ADMMConfig
        ADMM hyper-parameters.
    backend : Backend
        Quantum backend used for QAOA sub-problems.
    """

    def __init__(
        self,
        qubo: PortfolioQUBO,
        config: ADMMConfig,
        backend: Backend,
    ) -> None:
        self.qubo = qubo
        self.config = config
        self.backend = backend
        self._Q = qubo.build_matrix()
        self._n = self._Q.shape[0]
        self._rng = np.random.default_rng(config.seed)

    # ------------------------------------------------------------------
    # Partitioning
    # ------------------------------------------------------------------

    def _partition_variables(self) -> list[list[int]]:
        """Split variable indices into groups of at most *sub_problem_size*."""
        sz = self.config.sub_problem_size
        return [
            list(range(start, min(start + sz, self._n)))
            for start in range(0, self._n, sz)
        ]

    # ------------------------------------------------------------------
    # Sub-QUBO construction
    # ------------------------------------------------------------------

    def _build_sub_qubo(
        self,
        indices: list[int],
        z: NDArray[np.float64],
        u: NDArray[np.float64],
        rho: float,
    ) -> NDArray[np.float64]:
        """Build the augmented sub-QUBO matrix for a block of variables.

        The sub-problem objective for block *indices* is:

            x_i^T Q_ii x_i  +  c^T x_i  +  (rho/2) ||x_i - z_i + u_i||^2

        where ``c`` encodes the cross-terms with variables outside the
        block (fixed at their current *z* values).

        Because the solver works with a QUBO matrix (x^T M x), linear
        terms are folded onto the diagonal (valid since x_j^2 = x_j for
        binary variables).

        Parameters
        ----------
        indices : list[int]
            Variable indices belonging to this block.
        z : NDArray
            Current consensus variable vector (full size).
        u : NDArray
            Current scaled dual variable vector (full size).
        rho : float
            Current augmented Lagrangian penalty.

        Returns
        -------
        NDArray of shape (len(indices), len(indices))
        """
        idx = np.array(indices)
        m = len(idx)

        # Diagonal block of Q
        sub_Q = self._Q[np.ix_(idx, idx)].copy()

        # Cross-terms: contribution from variables outside this block
        # c_i = sum_{j not in block} (Q[i,j] + Q[j,i]) * z[j]
        all_idx = np.arange(self._n)
        mask = np.ones(self._n, dtype=bool)
        mask[idx] = False
        ext_idx = all_idx[mask]

        if len(ext_idx) > 0:
            # Q[block, external] and Q[external, block]^T give cross couplings
            cross = self._Q[np.ix_(idx, ext_idx)] + self._Q[np.ix_(ext_idx, idx)].T
            linear_cross = cross @ z[ext_idx]  # shape (m,)
        else:
            linear_cross = np.zeros(m, dtype=np.float64)

        # Augmented Lagrangian: (rho/2) * ||x_i - z_i + u_i||^2
        # Expand: (rho/2)(x_i^2 - 2 x_i (z_i - u_i) + const)
        # For binary x_i^2 = x_i, so diagonal gets +rho/2
        # and linear term is -rho*(z_i - u_i)
        z_block = z[idx]
        u_block = u[idx]

        linear_aug = -rho * (z_block - u_block)  # shape (m,)
        diag_aug = rho / 2.0  # scalar added to each diagonal entry

        # Fold all linear terms onto the diagonal (x_i^2 = x_i)
        for k in range(m):
            sub_Q[k, k] += linear_cross[k] + linear_aug[k] + diag_aug

        return sub_Q

    # ------------------------------------------------------------------
    # Sub-problem solver
    # ------------------------------------------------------------------

    def _solve_sub_problem(
        self,
        sub_Q: NDArray[np.float64],
        n_qubits: int,
    ) -> str:
        """Solve a sub-QUBO, returning the best bitstring.

        Parameters
        ----------
        sub_Q : NDArray
            The (augmented) QUBO matrix for the sub-problem.
        n_qubits : int
            Number of binary variables in the sub-problem.

        Returns
        -------
        str
            Best bitstring found.
        """
        if self.config.sub_solver == "exhaustive":
            return self._exhaustive_sub(sub_Q, n_qubits)
        return self._qaoa_sub(sub_Q, n_qubits)

    def _exhaustive_sub(
        self,
        sub_Q: NDArray[np.float64],
        n_qubits: int,
    ) -> str:
        """Brute-force enumeration for small sub-problems."""
        best_bs = "0" * n_qubits
        best_obj = float("inf")

        for bits in product([0, 1], repeat=n_qubits):
            x = np.array(bits, dtype=np.float64)
            obj = float(x @ sub_Q @ x)
            if obj < best_obj:
                best_obj = obj
                best_bs = "".join(str(b) for b in bits)

        return best_bs

    def _qaoa_sub(
        self,
        sub_Q: NDArray[np.float64],
        n_qubits: int,
    ) -> str:
        """Solve a sub-problem with QAOA via a synthetic PortfolioQUBO."""
        # Build a minimal PortfolioQUBO wrapper so QAOAPortfolio can be used.
        # We create a dummy QUBO whose build_matrix() returns our sub_Q.
        from qufin.portfolio.optimizers.qaoa import QAOAConfig, QAOAPortfolio

        dummy_qubo = PortfolioQUBO(
            mu=np.zeros(n_qubits, dtype=np.float64),
            cov=np.zeros((n_qubits, n_qubits), dtype=np.float64),
            gamma=1.0,
            encoding="one_hot",
        )

        config = QAOAConfig(
            p=self.config.qaoa_depth,
            mixer="x",
            shots=self.config.shots,
            seed=self.config.seed,
        )

        solver = QAOAPortfolio(dummy_qubo, config, self.backend)
        # Overwrite the internal Q matrix with our augmented sub-QUBO
        solver._Q = sub_Q

        result = solver.run()
        return result.best_bitstring

    # ------------------------------------------------------------------
    # Main ADMM loop
    # ------------------------------------------------------------------

    def run(self) -> ADMMResult:
        """Execute the ADMM decomposition and return the best portfolio.

        Returns
        -------
        ADMMResult
            Contains the best solution, convergence diagnostics, and
            per-iteration sub-problem objective traces.
        """
        start = time.perf_counter()
        n = self._n
        Q = self._Q
        partitions = self._partition_variables()

        # Initialise primal, consensus, and dual variables
        x = np.zeros(n, dtype=np.float64)
        z = np.zeros(n, dtype=np.float64)
        u = np.zeros(n, dtype=np.float64)
        rho = self.config.rho

        primal_residuals: list[float] = []
        dual_residuals: list[float] = []
        sub_objectives: list[list[float]] = []

        best_bitstring = "0" * n
        best_objective = float("inf")
        best_weights = np.zeros(n, dtype=np.float64)
        converged = False

        for _iteration in range(self.config.max_iterations):
            z_old = z.copy()
            iter_sub_objs: list[float] = []

            # ----- x-update: solve each block independently -----
            for idx_list in partitions:
                idx = np.array(idx_list)
                m = len(idx)

                sub_Q = self._build_sub_qubo(idx_list, z, u, rho)
                bs = self._solve_sub_problem(sub_Q, m)

                # Write solution back into x
                for k, bit_char in enumerate(bs):
                    x[idx[k]] = float(bit_char)

                # Record sub-problem objective (on the original sub-block)
                x_block = x[idx]
                sub_obj = float(x_block @ self._Q[np.ix_(idx, idx)] @ x_block)
                iter_sub_objs.append(sub_obj)

            sub_objectives.append(iter_sub_objs)

            # ----- z-update: consensus (non-overlapping blocks) -----
            z = x + u

            # Project z to {0, 1} via rounding (maintain binary feasibility)
            z = np.clip(np.round(z), 0.0, 1.0)

            # ----- u-update: dual ascent -----
            u = u + x - z

            # ----- Residuals -----
            primal_res = float(np.linalg.norm(x - z))
            dual_res = float(rho * np.linalg.norm(z - z_old))
            primal_residuals.append(primal_res)
            dual_residuals.append(dual_res)

            # ----- Track best feasible solution -----
            bitstring = "".join(str(int(b)) for b in x)
            obj = float(x @ Q @ x)
            if obj < best_objective:
                best_objective = obj
                best_bitstring = bitstring
                best_weights = self.qubo.decode_weights(bitstring)

            # ----- Adaptive rho -----
            if self.config.rho_update:
                if primal_res > 10 * dual_res and dual_res > 0:
                    rho *= 2.0
                elif dual_res > 10 * primal_res and primal_res > 0:
                    rho /= 2.0

            # ----- Convergence check -----
            if (
                primal_res < self.config.tol_primal
                and dual_res < self.config.tol_dual
            ):
                converged = True
                break

        wall_time = time.perf_counter() - start
        n_iters = len(primal_residuals)

        return ADMMResult(
            value=best_objective,
            n_shots=self.config.shots * len(partitions) * n_iters,
            wall_time_s=wall_time,
            backend_id=self.backend.backend_id,
            seed=self.config.seed,
            best_bitstring=best_bitstring,
            weights=best_weights,
            objective=best_objective,
            n_iterations=n_iters,
            primal_residuals=primal_residuals,
            dual_residuals=dual_residuals,
            sub_problem_objectives=sub_objectives,
            converged=converged,
        )
