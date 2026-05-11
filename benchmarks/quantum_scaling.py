"""Phase 4.2 — Quantum scaling analysis.

Measures how quantum algorithm cost scales with problem size:
- QAOA: wall-clock, circuit depth, n_qubits vs n_assets
- QAE: shots-to-accuracy vs n_eval_qubits
- QUBO: build time vs n_assets
- Circuit depth vs mixer type

Usage:
    python benchmarks/quantum_scaling.py [--output results/quantum_scaling.json]
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class ScalingEntry:
    category: str
    method: str
    n: int
    wall_time_s: float
    circuit_depth: int | None = None
    n_qubits: int | None = None
    n_params: int | None = None
    result: float | None = None
    extra: dict = field(default_factory=dict)


def _generate_problem(n_assets: int, seed: int = 42):
    """Generate realistic portfolio problem data."""
    rng = np.random.default_rng(seed)
    factors = rng.normal(0, 0.01, (504, min(3, n_assets)))
    loadings = rng.normal(0, 1, (min(3, n_assets), n_assets))
    idio = rng.normal(0, 0.005, (504, n_assets))
    returns = factors @ loadings + idio
    mu = np.mean(returns, axis=0)
    cov = np.cov(returns, rowvar=False)
    return mu, cov


# ---------------------------------------------------------------------------
# 1. QUBO build time scaling
# ---------------------------------------------------------------------------

def bench_qubo_build() -> list[ScalingEntry]:
    from qufin.portfolio.qubo import PortfolioQUBO

    entries = []

    for n_assets in [5, 10, 15, 20, 30, 50, 75, 100, 150, 200]:
        mu, cov = _generate_problem(n_assets)
        k = max(3, n_assets // 3)

        t0 = time.perf_counter()
        qubo = PortfolioQUBO(
            mu=mu, cov=cov, gamma=1.0,
            cardinality=k,
            budget_penalty=1e4,
            encoding="one_hot",
        )
        Q = qubo.build_matrix()
        wall = time.perf_counter() - t0

        entries.append(ScalingEntry(
            category="QUBO Build", method="one_hot",
            n=n_assets, wall_time_s=wall,
            n_qubits=Q.shape[0],
            extra={"cardinality": k, "nnz": int(np.count_nonzero(Q))},
        ))

    # Binary encoding scaling
    for n_assets in [5, 10, 15, 20, 30, 50]:
        mu, cov = _generate_problem(n_assets)

        t0 = time.perf_counter()
        qubo = PortfolioQUBO(
            mu=mu, cov=cov, gamma=1.0,
            budget_penalty=1e4,
            encoding="binary", bits_per_asset=3,
        )
        Q = qubo.build_matrix()
        wall = time.perf_counter() - t0

        entries.append(ScalingEntry(
            category="QUBO Build", method="binary_3bit",
            n=n_assets, wall_time_s=wall,
            n_qubits=Q.shape[0],
        ))

    return entries


# ---------------------------------------------------------------------------
# 2. Exhaustive solver scaling
# ---------------------------------------------------------------------------

def bench_exhaustive() -> list[ScalingEntry]:
    from qufin.portfolio.qubo import PortfolioQUBO
    from qufin.portfolio.optimizers.exhaustive import exhaustive_solve

    entries = []

    for n_assets in [4, 6, 8, 10, 12, 14, 16, 18, 20]:
        mu, cov = _generate_problem(n_assets)
        k = max(2, n_assets // 3)

        qubo = PortfolioQUBO(
            mu=mu, cov=cov, gamma=1.0,
            cardinality=k, budget_penalty=1e4,
            encoding="one_hot",
        )

        t0 = time.perf_counter()
        res = exhaustive_solve(qubo)
        wall = time.perf_counter() - t0

        entries.append(ScalingEntry(
            category="Exhaustive Solver", method="exhaustive",
            n=n_assets, wall_time_s=wall,
            n_qubits=n_assets,
            result=res.best_objective,
            extra={"n_evaluated": res.n_evaluated, "feasible": res.feasible},
        ))

        # Stop if taking too long
        if wall > 30:
            break

    return entries


# ---------------------------------------------------------------------------
# 3. QAOA scaling (wall-clock, depth, qubits)
# ---------------------------------------------------------------------------

def bench_qaoa_scaling() -> list[ScalingEntry]:
    from qufin.portfolio.qubo import PortfolioQUBO
    from qufin.portfolio.optimizers.qaoa import QAOAPortfolio, QAOAConfig
    from qufin.backends.qiskit_backend import QiskitAerBackend

    entries = []
    backend = QiskitAerBackend(seed=42)

    for n_assets in [4, 6, 8, 10, 12]:
        mu, cov = _generate_problem(n_assets)
        k = max(2, n_assets // 3)

        qubo = PortfolioQUBO(
            mu=mu, cov=cov, gamma=1.0,
            cardinality=k, budget_penalty=1e4,
            encoding="one_hot",
        )

        for p in [1, 2, 3]:
            for mixer in ["x", "xy_ring"]:
                config = QAOAConfig(
                    p=p, mixer=mixer,
                    cardinality=k if mixer != "x" else None,
                    maxiter=50,  # Reduced for benchmarking
                    shots=2048,
                    seed=42,
                )

                t0 = time.perf_counter()
                try:
                    solver = QAOAPortfolio(qubo, config, backend)
                    result = solver.run()
                    wall = time.perf_counter() - t0

                    entries.append(ScalingEntry(
                        category="QAOA Scaling", method=f"qaoa_p{p}_{mixer}",
                        n=n_assets, wall_time_s=wall,
                        n_qubits=n_assets,
                        n_params=2 * p,
                        result=result.best_objective,
                        extra={
                            "feasible": result.feasible,
                            "n_iterations": len(result.history),
                        },
                    ))
                except Exception as e:
                    wall = time.perf_counter() - t0
                    entries.append(ScalingEntry(
                        category="QAOA Scaling", method=f"qaoa_p{p}_{mixer}",
                        n=n_assets, wall_time_s=wall,
                        n_qubits=n_assets,
                        extra={"error": str(e)},
                    ))

    return entries


# ---------------------------------------------------------------------------
# 4. VQE scaling
# ---------------------------------------------------------------------------

def bench_vqe_scaling() -> list[ScalingEntry]:
    from qufin.portfolio.qubo import PortfolioQUBO
    from qufin.portfolio.optimizers.vqe import VQEPortfolio, VQEConfig
    from qufin.backends.qiskit_backend import QiskitAerBackend

    entries = []
    backend = QiskitAerBackend(seed=42)

    for n_assets in [4, 6, 8]:
        mu, cov = _generate_problem(n_assets)

        qubo = PortfolioQUBO(
            mu=mu, cov=cov, gamma=1.0, budget_penalty=1e4,
            encoding="one_hot",
        )

        for reps in [1, 2, 3]:
            config = VQEConfig(
                reps=reps,
                maxiter=50,
                shots=2048,
                seed=42,
            )

            t0 = time.perf_counter()
            try:
                solver = VQEPortfolio(qubo, config, backend)
                result = solver.run()
                wall = time.perf_counter() - t0

                entries.append(ScalingEntry(
                    category="VQE Scaling", method=f"vqe_reps{reps}",
                    n=n_assets, wall_time_s=wall,
                    n_qubits=n_assets,
                    n_params=len(result.optimal_params) if result.optimal_params is not None else 0,
                    result=result.best_objective,
                ))
            except Exception as e:
                wall = time.perf_counter() - t0
                entries.append(ScalingEntry(
                    category="VQE Scaling", method=f"vqe_reps{reps}",
                    n=n_assets, wall_time_s=wall,
                    n_qubits=n_assets,
                    extra={"error": str(e)},
                ))

    return entries


# ---------------------------------------------------------------------------
# 5. QAE accuracy vs eval qubits
# ---------------------------------------------------------------------------

def bench_qae_scaling() -> list[ScalingEntry]:
    from qufin.options.amplitude_estimation.estimation_problem import EstimationProblem
    from qufin.options.amplitude_estimation.canonical import (
        CanonicalAmplitudeEstimation, CanonicalQAEConfig,
    )
    from qufin.options.amplitude_estimation.iqae import (
        IterativeAmplitudeEstimation, IQAEConfig,
    )
    from qufin.backends.qiskit_backend import QiskitAerBackend
    from qiskit.circuit import QuantumCircuit

    entries = []
    backend = QiskitAerBackend(seed=42)
    true_amplitude = 0.25  # sin^2(pi/6)

    # Build A operator: RY(2*arcsin(sqrt(0.25))) = RY(pi/3)
    a_circuit = QuantumCircuit(1)
    a_circuit.ry(2 * np.arcsin(np.sqrt(true_amplitude)), 0)

    problem = EstimationProblem(
        state_preparation=a_circuit,
        objective_qubits=[0],
    )

    # Canonical QAE: accuracy vs n_eval_qubits
    for n_eval in [3, 4, 5, 6, 7, 8]:
        t0 = time.perf_counter()
        try:
            config = CanonicalQAEConfig(n_eval_qubits=n_eval, shots=4096, seed=42)
            solver = CanonicalAmplitudeEstimation(problem, config, backend)
            result = solver.estimate()
            wall = time.perf_counter() - t0
            est = result.estimation
            error = abs(est - true_amplitude)

            entries.append(ScalingEntry(
                category="QAE Accuracy", method="canonical_qae",
                n=n_eval, wall_time_s=wall,
                n_qubits=n_eval + 1,
                result=est,
                extra={"abs_error": error, "true_amplitude": true_amplitude},
            ))
        except Exception as e:
            wall = time.perf_counter() - t0
            entries.append(ScalingEntry(
                category="QAE Accuracy", method="canonical_qae",
                n=n_eval, wall_time_s=wall,
                extra={"error": str(e)},
            ))

    # IQAE: accuracy vs epsilon
    for eps in [0.1, 0.05, 0.01, 0.005, 0.001]:
        t0 = time.perf_counter()
        try:
            config = IQAEConfig(epsilon=eps, alpha=0.05, shots=4096, seed=42)
            solver = IterativeAmplitudeEstimation(problem, config, backend)
            result = solver.estimate()
            wall = time.perf_counter() - t0
            est = result.estimation
            error = abs(est - true_amplitude)

            entries.append(ScalingEntry(
                category="QAE Accuracy", method="iqae",
                n=int(1 / eps),
                wall_time_s=wall,
                result=est,
                extra={"epsilon": eps, "abs_error": error, "true_amplitude": true_amplitude},
            ))
        except Exception as e:
            wall = time.perf_counter() - t0
            entries.append(ScalingEntry(
                category="QAE Accuracy", method="iqae",
                n=int(1 / eps),
                wall_time_s=wall,
                extra={"epsilon": eps, "error": str(e)},
            ))

    return entries


# ---------------------------------------------------------------------------
# 6. Circuit depth analysis
# ---------------------------------------------------------------------------

def bench_circuit_depth() -> list[ScalingEntry]:
    from qufin.portfolio.qubo import PortfolioQUBO
    from qufin.portfolio.mixers import get_mixer, DickeInitialState
    from qiskit.circuit import QuantumCircuit

    entries = []

    for n_qubits in [4, 6, 8, 10, 12, 14, 16]:
        mu, cov = _generate_problem(n_qubits)
        k = max(2, n_qubits // 3)

        for mixer_name in ["x", "xy_ring", "xy_full"]:
            try:
                mixer = get_mixer(
                    mixer_name, n_qubits,
                    cardinality=k if mixer_name != "x" else None,
                )
                qc = QuantumCircuit(n_qubits)

                # Initial state
                if mixer_name in ("xy_ring", "xy_full"):
                    init = DickeInitialState(n_qubits, k)
                    qc.compose(init.circuit(), inplace=True)

                # One layer of mixer
                qc.compose(mixer.circuit(beta=0.5), inplace=True)

                from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
                pm = generate_preset_pass_manager(optimization_level=2, basis_gates=["cx", "rz", "sx", "x"])
                transpiled = pm.run(qc)
                depth = transpiled.depth()
                cx_count = transpiled.count_ops().get("cx", 0)

                entries.append(ScalingEntry(
                    category="Circuit Depth", method=f"mixer_{mixer_name}",
                    n=n_qubits, wall_time_s=0,
                    circuit_depth=depth,
                    n_qubits=n_qubits,
                    extra={"cx_count": cx_count, "cardinality": k},
                ))
            except Exception as e:
                entries.append(ScalingEntry(
                    category="Circuit Depth", method=f"mixer_{mixer_name}",
                    n=n_qubits, wall_time_s=0,
                    n_qubits=n_qubits,
                    extra={"error": str(e)},
                ))

    return entries


# ---------------------------------------------------------------------------
# 7. Quantum vs Classical comparison
# ---------------------------------------------------------------------------

def bench_quantum_vs_classical() -> list[ScalingEntry]:
    """Compare QAOA vs classical on same problem instances."""
    from qufin.portfolio.qubo import PortfolioQUBO
    from qufin.portfolio.optimizers.exhaustive import exhaustive_solve
    from qufin.portfolio.optimizers.qaoa import QAOAPortfolio, QAOAConfig
    from qufin.portfolio.classical.mean_variance import mean_variance, Objective
    from qufin.backends.qiskit_backend import QiskitAerBackend

    entries = []
    backend = QiskitAerBackend(seed=42)

    for n_assets in [4, 6, 8, 10]:
        mu, cov = _generate_problem(n_assets)
        k = max(2, n_assets // 3)

        # Exhaustive (optimal solution)
        qubo = PortfolioQUBO(
            mu=mu, cov=cov, gamma=1.0,
            cardinality=k, budget_penalty=1e4,
            encoding="one_hot",
        )
        t0 = time.perf_counter()
        exact = exhaustive_solve(qubo)
        t_exact = time.perf_counter() - t0

        # QAOA p=2, xy_ring
        config = QAOAConfig(
            p=2, mixer="xy_ring", cardinality=k,
            maxiter=100, shots=4096, seed=42,
        )
        t0 = time.perf_counter()
        try:
            qaoa_solver = QAOAPortfolio(qubo, config, backend)
            qaoa_res = qaoa_solver.run()
            t_qaoa = time.perf_counter() - t0
            qaoa_obj = qaoa_res.best_objective
        except Exception:
            t_qaoa = time.perf_counter() - t0
            qaoa_obj = None

        # Classical mean-variance with cardinality
        t0 = time.perf_counter()
        mv_res = mean_variance(mu, cov, objective=Objective.MIN_VARIANCE, cardinality=k)
        t_classical = time.perf_counter() - t0

        entries.append(ScalingEntry(
            category="Quantum vs Classical", method="exhaustive_optimal",
            n=n_assets, wall_time_s=t_exact, result=exact.best_objective,
            extra={"cardinality": k},
        ))
        if qaoa_obj is not None:
            approx_ratio = qaoa_obj / exact.best_objective if exact.best_objective != 0 else 0
            entries.append(ScalingEntry(
                category="Quantum vs Classical", method="qaoa_p2_xy_ring",
                n=n_assets, wall_time_s=t_qaoa, result=qaoa_obj,
                extra={"approximation_ratio": approx_ratio, "feasible": qaoa_res.feasible},
            ))
        entries.append(ScalingEntry(
            category="Quantum vs Classical", method="classical_mean_var",
            n=n_assets, wall_time_s=t_classical, result=mv_res.volatility,
            extra={"cardinality": k},
        ))

    return entries


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all() -> list[ScalingEntry]:
    all_entries = []
    benchmarks = [
        ("QUBO Build", bench_qubo_build),
        ("Exhaustive Solver", bench_exhaustive),
        ("QAOA Scaling", bench_qaoa_scaling),
        ("VQE Scaling", bench_vqe_scaling),
        ("QAE Accuracy", bench_qae_scaling),
        ("Circuit Depth", bench_circuit_depth),
        ("Quantum vs Classical", bench_quantum_vs_classical),
    ]

    for name, fn in benchmarks:
        print(f"  Running {name}...")
        try:
            entries = fn()
            all_entries.extend(entries)
            print(f"    {len(entries)} entries collected")
        except Exception as e:
            print(f"    FAILED: {e}")

    return all_entries


def to_markdown(entries: list[ScalingEntry]) -> str:
    lines = ["# Quantum Scaling Benchmark Results\n"]

    categories = sorted(set(e.category for e in entries))
    for cat in categories:
        lines.append(f"\n## {cat}\n")
        lines.append("| Method | N | Time (s) | Qubits | Depth | Result |")
        lines.append("|--------|---|----------|--------|-------|--------|")
        for e in entries:
            if e.category != cat:
                continue
            q = str(e.n_qubits) if e.n_qubits else "—"
            d = str(e.circuit_depth) if e.circuit_depth else "—"
            r = f"{e.result:.6f}" if e.result is not None else "—"
            lines.append(
                f"| {e.method} | {e.n} | {e.wall_time_s:.4f} | {q} | {d} | {r} |"
            )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Run quantum scaling benchmarks")
    parser.add_argument("--output", type=str, default="benchmarks/results/quantum_scaling.json")
    args = parser.parse_args()

    print("=" * 60)
    print("qufin Quantum Scaling Benchmark Suite")
    print("=" * 60)

    entries = run_all()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump([asdict(e) for e in entries], f, indent=2)
    print(f"\nJSON saved to {out_path}")

    md_path = out_path.with_suffix(".md")
    md_path.write_text(to_markdown(entries))
    print(f"Markdown saved to {md_path}")

    print(f"\nTotal entries: {len(entries)}")


if __name__ == "__main__":
    main()
