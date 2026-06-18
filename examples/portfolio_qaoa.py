"""Cardinality-constrained portfolio optimization with QAOA.

Run:
    python examples/portfolio_qaoa.py

Builds a small Markowitz QUBO (pick exactly K of N assets to maximize
risk-adjusted return) and solves it with QAOA on the Aer simulator, using an
XY-ring mixer that preserves the cardinality (Hamming-weight) constraint.
"""

from __future__ import annotations

import numpy as np

from qufin.backends.qiskit_backend import QiskitAerBackend
from qufin.portfolio.optimizers.qaoa import QAOAConfig, QAOAPortfolio
from qufin.portfolio.qubo import PortfolioQUBO


def main() -> None:
    rng = np.random.default_rng(0)
    n_assets, k = 6, 3

    mu = rng.normal(0.0010, 0.0005, n_assets)
    factor = rng.normal(0, 1, (n_assets, n_assets)) * 0.1
    cov = factor @ factor.T + np.eye(n_assets) * 0.02

    qubo = PortfolioQUBO(mu=mu, cov=cov, gamma=1.0, cardinality=k)
    print(f"{n_assets} assets, choose K={k}  ->  {qubo.n_qubits} qubits")

    config = QAOAConfig(p=2, mixer="xy_ring", cardinality=k, shots=2048, maxiter=60, seed=42)
    result = QAOAPortfolio(qubo, config, QiskitAerBackend(seed=42)).run()

    selected = [i for i, b in enumerate(result.best_bitstring) if b == "1"]
    print(f"QAOA selection : {result.best_bitstring}  -> assets {selected}")
    print(f"feasible (==K) : {result.feasible}")
    print(f"QUBO objective : {result.best_objective:.6f}")


if __name__ == "__main__":
    main()
