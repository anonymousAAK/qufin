"""Runnable quickstart for qufin.

Every block below uses the real public API and runs end-to-end on the
default install (qiskit + Aer). The README Quickstart mirrors these
snippets verbatim, so this script is the executable contract for the
documentation: if it runs clean, the README examples run clean.

Run with::

    python examples/quickstart.py
"""

from __future__ import annotations

import numpy as np


def option_pricing() -> None:
    """Option pricing: classical Black-Scholes vs. quantum amplitude estimation.

    The quantum block estimates an amplitude ``a = sin^2(theta)`` with
    Iterative Quantum Amplitude Estimation (IQAE) and compares it to the
    analytically known value. The same IQAE engine drives the option-pricing
    estimation problems; here we use a gate-based Bernoulli oracle so the
    example is fully self-contained and fast.
    """
    from qiskit.circuit import QuantumCircuit

    from qufin.backends.qiskit_backend import QiskitAerBackend
    from qufin.options.amplitude_estimation.estimation_problem import EstimationProblem
    from qufin.options.amplitude_estimation.iqae import (
        IQAEConfig,
        IterativeAmplitudeEstimation,
    )
    from qufin.options.classical.black_scholes import call_price, price_and_greeks

    # Classical: Black-Scholes closed form (price + full Greeks).
    price = call_price(s=100, k=105, r=0.05, sigma=0.2, T=1.0)
    greeks = price_and_greeks(s=100, k=105, r=0.05, sigma=0.2, T=1.0, option_type="call")

    # Quantum: Iterative Quantum Amplitude Estimation of a = sin^2(theta).
    theta = np.pi / 5
    oracle = QuantumCircuit(1)
    oracle.ry(2 * theta, 0)  # A|0> = cos(theta)|0> + sin(theta)|1>
    problem = EstimationProblem(state_preparation=oracle, objective_qubits=[0], n_qubits=1)

    backend = QiskitAerBackend(method="automatic", seed=42)
    config = IQAEConfig(epsilon_target=0.01, shots_per_round=2048, seed=42)
    estimate = IterativeAmplitudeEstimation(problem, config, backend).estimate()

    print(f"Black-Scholes call: {price:.4f}  (delta={greeks.delta:.4f})")
    print(f"IQAE estimate:      {estimate.estimate:.4f}  (true {np.sin(theta) ** 2:.4f})")


def portfolio_qaoa() -> None:
    """Cardinality-constrained portfolio optimization with QAOA."""
    from qufin.backends.qiskit_backend import QiskitAerBackend
    from qufin.portfolio.optimizers.qaoa import QAOAConfig, QAOAPortfolio
    from qufin.portfolio.qubo import PortfolioQUBO

    rng = np.random.default_rng(42)
    n_assets = 6
    mu = rng.uniform(0.05, 0.15, n_assets)
    factor = rng.standard_normal((n_assets, n_assets))
    cov = (factor @ factor.T) / n_assets

    qubo = PortfolioQUBO(
        mu=mu, cov=cov, gamma=0.5, cardinality=3, encoding="one_hot",
    )
    config = QAOAConfig(
        p=2, mixer="xy_ring", cardinality=3, maxiter=50, shots=2048, seed=42,
    )
    backend = QiskitAerBackend(method="automatic", seed=42)
    result = QAOAPortfolio(qubo, config, backend).run()

    print(f"Selected (bitstring): {result.best_bitstring}")
    print(f"Objective:            {result.best_objective:.6f}  feasible={result.feasible}")


def synthetic_market_data() -> None:
    """Synthetic market data: GBM and Heston paths."""
    from qufin.data.synthetic import gbm_paths, heston_paths

    gbm = gbm_paths(
        s0=100, mu=0.08, sigma=0.2, T=1.0, n_steps=252, n_paths=10_000, seed=42,
    )

    # heston_paths returns (prices, variances), each shape (n_paths, n_steps + 1).
    prices, _variances = heston_paths(
        s0=100, v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7,
        mu=0.08, T=1.0, n_steps=252, n_paths=10_000, seed=42,
    )

    print(f"GBM terminal mean:    {gbm[:, -1].mean():.2f}")
    print(f"Heston terminal mean: {prices[:, -1].mean():.2f}")


def backtesting() -> None:
    """Walk-forward backtesting with a performance summary."""
    from qufin.backtesting.engine import BacktestEngine
    from qufin.backtesting.metrics import performance_summary

    rng = np.random.default_rng(0)
    returns = rng.normal(0.0004, 0.01, size=(800, 5))  # T x N return matrix

    def equal_weight(mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
        return np.ones(len(mu)) / len(mu)

    engine = BacktestEngine(returns, train_window=252, test_window=21)
    result = engine.run(equal_weight, strategy_name="equal_weight")

    # Or compute a summary directly from a return series.
    summary = performance_summary(result.portfolio_returns)

    print(f"Backtest Sharpe: {result.summary.sharpe_ratio:.2f}")
    print(f"Max drawdown:    {summary.max_drawdown:.2%}")


def automatic_backend() -> None:
    """Automatic backend selection for a given circuit."""
    from qiskit.circuit import QuantumCircuit

    from qufin.backends.auto_select import auto_select_backend

    circuit = QuantumCircuit(4)
    circuit.h(range(4))

    backend = auto_select_backend(circuit)  # GPU -> Aer -> Mock
    print(f"Auto-selected backend: {backend.backend_id}")


def benchmarks() -> None:
    """Standardized quantum-vs-classical benchmark harness."""
    from qufin.benchmarks.problems import portfolio_small
    from qufin.benchmarks.runner import BenchmarkRunner, SolverEntry

    def mean_variance(problem) -> dict:
        mu, cov = problem.mu, problem.cov
        weights = np.linalg.solve(cov + 1e-6 * np.eye(len(mu)), mu)
        weights = np.clip(weights, 0, None)
        weights /= weights.sum()
        objective = float(weights @ mu - 0.5 * weights @ cov @ weights)
        return {"objective": objective, "backend": "cpu", "seed": 42}

    runner = BenchmarkRunner()
    runner.register(
        SolverEntry(name="mean_variance", family="classical", solve_fn=mean_variance)
    )
    rows = runner.run_problem(portfolio_small())  # 15-asset benchmark
    for row in rows:
        print(f"{row.solver_name} ({row.solver_family}): objective={row.objective:.4f}")


def main() -> None:
    sections = [
        ("Option pricing: classical vs. quantum", option_pricing),
        ("Portfolio optimization with QAOA", portfolio_qaoa),
        ("Synthetic market data", synthetic_market_data),
        ("Backtesting", backtesting),
        ("Automatic backend selection", automatic_backend),
        ("Benchmarks", benchmarks),
    ]
    for title, fn in sections:
        print(f"\n=== {title} ===")
        fn()
    print("\nAll quickstart examples ran successfully.")


if __name__ == "__main__":
    main()
