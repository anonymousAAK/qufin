"""Portfolio QUBO formulation with realistic constraints.

Builds the Markowitz QUBO with optional cardinality, sector, turnover,
and transaction-cost penalty terms. Supports one-hot and binary encodings.

References
----------
Brandhofer et al., arXiv:2207.10555 — portfolio QAOA benchmarking.
arXiv:2601.03278 — slack ancilla for inequality constraints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

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
        Exactly K assets must be selected (one-hot encoding).
    sector_map : dict[int, int] | None
        Mapping asset index -> sector index.
    sector_caps : dict[int, int] | None
        Max assets per sector.
    turnover_penalty : float
        Penalty coefficient for turnover from previous_weights.
    transaction_cost : float
        Per-asset transaction cost coefficient.
    previous_weights : NDArray | None
        Previous portfolio (binary selection vector) for turnover/cost.
    budget_penalty : float | None
        Penalty for budget constraint (sum x_i = 1 or K).
        If None, auto-scaled from Q matrix magnitude.
    encoding : str
        "one_hot" (1 qubit per asset) or "binary" (log-bits per asset).
    bits_per_asset : int
        Bits per asset in binary encoding (ignored for one_hot).
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
    budget_penalty: float | None = None
    encoding: Literal["one_hot", "binary"] = "one_hot"
    bits_per_asset: int = 3

    def __post_init__(self) -> None:
        self._n_assets = len(self.mu)
        if self.cov.shape != (self._n_assets, self._n_assets):
            raise ValueError(
                f"Covariance shape {self.cov.shape} doesn't match "
                f"mu length {self._n_assets}"
            )

    @property
    def n_assets(self) -> int:
        return self._n_assets

    @property
    def n_qubits(self) -> int:
        """Total qubits needed for the chosen encoding."""
        if self.encoding == "one_hot":
            return self._n_assets
        return self._n_assets * self.bits_per_asset

    def _auto_penalty(self, Q: NDArray[np.float64]) -> float:
        """Auto-scale penalty from Q matrix magnitude."""
        max_val = float(np.max(np.abs(Q)))
        return max_val * 2 if max_val > 0 else 1.0

    def build_matrix(self) -> NDArray[np.float64]:
        """Build the QUBO Q matrix for minimization: min x^T Q x.

        Returns
        -------
        NDArray of shape (n_qubits, n_qubits)
        """
        if self.encoding == "one_hot":
            return self._build_onehot()
        return self._build_binary()

    def _build_onehot(self) -> NDArray[np.float64]:
        """Build QUBO for one-hot (binary inclusion) encoding."""
        n = self._n_assets
        Q = np.zeros((n, n), dtype=np.float64)

        # Return term (linear -> diagonal): maximize return = minimize -return
        for i in range(n):
            Q[i, i] -= self.mu[i]

        # Risk term (quadratic)
        Q += self.gamma * self.cov

        # Penalty auto-scaling
        penalty = self.budget_penalty if self.budget_penalty is not None else self._auto_penalty(Q)

        # Cardinality constraint: (sum(x) - K)^2
        if self.cardinality is not None:
            K = self.cardinality
            for i in range(n):
                Q[i, i] += penalty * (1 - 2 * K)
                for j in range(i + 1, n):
                    Q[i, j] += penalty
                    Q[j, i] += penalty

        # Sector constraints: for each sector s, (sum_{i in s} x_i - cap_s)^2
        # penalized only when exceeded
        if self.sector_map is not None and self.sector_caps is not None:
            sectors: dict[int, list[int]] = {}
            for asset_idx, sector_idx in self.sector_map.items():
                sectors.setdefault(sector_idx, []).append(asset_idx)

            for sector_idx, assets in sectors.items():
                cap = self.sector_caps.get(sector_idx)
                if cap is None:
                    continue
                # Penalty: (sum_{i in sector} x_i - cap)^2
                # Only penalize if exceeding cap; approximate with squared penalty
                sector_penalty = penalty * 0.5
                for i in assets:
                    Q[i, i] += sector_penalty * (1 - 2 * cap)
                    for j in assets:
                        if j > i:
                            Q[i, j] += sector_penalty
                            Q[j, i] += sector_penalty

        # Turnover penalty: sum_i |x_i - x_i^prev|
        # For binary variables: |x_i - x_i^prev| = x_i(1-x_i^prev) + x_i^prev(1-x_i)
        # = x_i + x_i^prev - 2*x_i*x_i^prev (since x_i^prev is constant)
        # Linear in x_i: coefficient = turnover_penalty * (1 - 2*x_i^prev)
        if self.turnover_penalty > 0 and self.previous_weights is not None:
            for i in range(n):
                Q[i, i] += self.turnover_penalty * (1 - 2 * self.previous_weights[i])

        # Transaction cost: cost * sum_i |x_i - x_i^prev|
        # Same linearization as turnover
        if self.transaction_cost > 0 and self.previous_weights is not None:
            for i in range(n):
                Q[i, i] += self.transaction_cost * (1 - 2 * self.previous_weights[i])

        return Q

    def _build_binary(self) -> NDArray[np.float64]:
        """Build QUBO for binary (integer weight) encoding.

        Each asset i uses `bits_per_asset` qubits to encode weight level.
        Weight_i = sum_b 2^b * x_{i,b} / (2^bits_per_asset - 1).
        """
        n = self._n_assets
        B = self.bits_per_asset
        nq = n * B
        max_level = 2**B - 1
        Q = np.zeros((nq, nq), dtype=np.float64)

        # Map qubit indices: asset i, bit b -> qubit index i*B + b
        # Weight factor for bit b of asset i: 2^b / max_level
        def _wf(b: int) -> float:
            return (2**b) / max_level

        # Return term: -mu_i * w_i = -mu_i * sum_b wf(b) * x_{i,b}
        for i in range(n):
            for b in range(B):
                idx = i * B + b
                Q[idx, idx] -= self.mu[i] * _wf(b)

        # Risk term: gamma * sum_{ij} cov_{ij} * w_i * w_j
        # = gamma * sum_{ij} cov_{ij} * (sum_b wf(b)*x_{ib}) * (sum_c wf(c)*x_{jc})
        for i in range(n):
            for j in range(n):
                for bi in range(B):
                    for bj in range(B):
                        qi = i * B + bi
                        qj = j * B + bj
                        val = self.gamma * self.cov[i, j] * _wf(bi) * _wf(bj)
                        if qi == qj:
                            # x^2 = x for binary, so diagonal
                            Q[qi, qi] += val
                        elif qi < qj:
                            Q[qi, qj] += val
                            Q[qj, qi] += val

        # Budget constraint: (sum_i w_i - 1)^2
        penalty = self.budget_penalty if self.budget_penalty is not None else self._auto_penalty(Q)
        # (sum_i sum_b wf(b)*x_{ib} - 1)^2
        for i in range(n):
            for bi in range(B):
                qi = i * B + bi
                # Linear term: -2 * wf(bi) * penalty  (from -2*1*wf*x)
                # Plus wf^2 from x^2=x
                Q[qi, qi] += penalty * (_wf(bi) ** 2 - 2 * _wf(bi))
                for j in range(n):
                    for bj in range(B):
                        qj = j * B + bj
                        if qj > qi:
                            Q[qi, qj] += penalty * _wf(bi) * _wf(bj)
                            Q[qj, qi] += penalty * _wf(bi) * _wf(bj)

        return Q

    def evaluate(self, bitstring: str) -> float:
        """Evaluate the QUBO objective for a given bitstring."""
        x = np.array([int(c) for c in bitstring], dtype=np.float64)
        Q = self.build_matrix()
        return float(x @ Q @ x)

    def decode_weights(self, bitstring: str) -> NDArray[np.float64]:
        """Decode a bitstring into portfolio weights.

        For one-hot: weights are the binary selection (equal weight among selected).
        For binary: weights are decoded integer levels, normalized to sum to 1.
        """
        if self.encoding == "one_hot":
            selection = np.array([int(c) for c in bitstring], dtype=np.float64)
            total = selection.sum()
            if total > 0:
                return selection / total
            return selection
        else:
            B = self.bits_per_asset
            max_level = 2**B - 1
            weights = np.zeros(self._n_assets, dtype=np.float64)
            for i in range(self._n_assets):
                bits = bitstring[i * B : (i + 1) * B]
                # Bit b of asset i sits at offset b with significance 2**b, to
                # match _build_binary's weight factor 2**b / max_level. Parsing
                # the slice as MSB-first (int(bits, 2)) reverses bit significance
                # and decodes the wrong weight level.
                level = sum(int(bits[b]) << b for b in range(len(bits)))
                weights[i] = level / max_level
            total = weights.sum()
            if total > 0:
                weights = weights / total
            return weights

    def feasibility_check(self, bitstring: str) -> dict[str, bool]:
        """Check which constraints a bitstring satisfies."""
        x = np.array([int(c) for c in bitstring], dtype=np.float64)
        result: dict[str, bool] = {}

        if self.encoding == "one_hot":
            selected = int(x.sum())
            if self.cardinality is not None:
                result["cardinality"] = selected == self.cardinality

            if self.sector_map is not None and self.sector_caps is not None:
                sectors: dict[int, int] = {}
                for asset_idx, sector_idx in self.sector_map.items():
                    if x[asset_idx] > 0.5:
                        sectors[sector_idx] = sectors.get(sector_idx, 0) + 1
                result["sector"] = all(
                    sectors.get(s, 0) <= cap
                    for s, cap in self.sector_caps.items()
                )

        return result
