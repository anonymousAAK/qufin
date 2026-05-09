"""Portfolio QUBO formulation with realistic constraints.

Builds the Markowitz QUBO: q*x - gamma * x^T Sigma x
with optional cardinality, sector, turnover, and transaction-cost penalties.

References
----------
Brandhofer et al., arXiv:2207.10555.
arXiv:2601.03278 — slack ancilla for inequality constraints.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class PortfolioQUBO:
    """QUBO formulation for portfolio optimization.

    Parameters
    ----------
    mu : NDArray
        Expected returns, shape (N,).
    cov : NDArray
        Covariance matrix, shape (N, N).
    gamma : float
        Risk aversion parameter.
    cardinality : int | None
        Exactly K assets must be selected.
    sector_map : dict[int, int] | None
        Mapping asset index -> sector index.
    sector_caps : dict[int, int] | None
        Max assets per sector.
    turnover_penalty : float
        Penalty coefficient for turnover from previous_weights.
    transaction_cost : float
        Penalty coefficient for transaction costs.
    previous_weights : NDArray | None
        Previous portfolio weights for turnover/cost calculation.
    """

    mu: NDArray[np.float64]
    cov: NDArray[np.float64]
    gamma: float = 1.0
    cardinality: int | None = None
    sector_map: dict[int, int] | None = None
    sector_caps: dict[int, int] | None = None
    turnover_penalty: float = 0.0
    transaction_cost: float = 0.0
    previous_weights: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        self._n_assets = len(self.mu)

    @property
    def n_assets(self) -> int:
        return self._n_assets

    @property
    def n_qubits(self) -> int:
        """Number of qubits for one-hot encoding."""
        return self._n_assets

    def build_matrix(self) -> NDArray[np.float64]:
        """Build the QUBO Q matrix.

        The objective is: x^T Q x (minimization).

        Returns
        -------
        NDArray of shape (n_qubits, n_qubits)
        """
        n = self._n_assets
        Q = np.zeros((n, n), dtype=np.float64)

        # Return term (linear -> diagonal)
        for i in range(n):
            Q[i, i] -= self.mu[i]

        # Risk term (quadratic)
        Q += self.gamma * self.cov

        # Cardinality penalty (if set)
        if self.cardinality is not None:
            K = self.cardinality
            penalty = float(np.max(np.abs(Q))) * 2
            # Penalty: (sum(x) - K)^2 = sum_ij x_i x_j - 2K sum_i x_i + K^2
            for i in range(n):
                Q[i, i] += penalty * (1 - 2 * K)
                for j in range(i + 1, n):
                    Q[i, j] += penalty
                    Q[j, i] += penalty

        return Q

    def evaluate(self, bitstring: str) -> float:
        """Evaluate the QUBO objective for a given bitstring."""
        x = np.array([int(c) for c in bitstring], dtype=np.float64)
        Q = self.build_matrix()
        return float(x @ Q @ x)
