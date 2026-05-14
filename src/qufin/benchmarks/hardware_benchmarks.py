"""Hardware benchmark framework for quantum finance validation campaigns.

Provides configurable benchmark runners for QAOA and QAE circuits on
real hardware (IBM, IonQ via Braket) with statistical analysis,
error mitigation comparison, and reproducibility manifests.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class HardwareBenchmarkConfig:
    """Configuration for a hardware benchmark campaign.

    Parameters
    ----------
    target_devices : list[str]
        Backend identifiers to benchmark on.
    qubit_counts : list[int]
        Number of qubits to test at each scale.
    qaoa_depths : list[int]
        QAOA circuit depths (p values) to benchmark.
    qae_precisions : list[int]
        QAE precision levels (number of evaluation qubits).
    shots : int
        Number of measurement shots per circuit.
    n_runs : int
        Number of independent runs for statistical analysis.
    mitigation_methods : list[str]
        Error mitigation methods to apply: "none", "zne", "trex", "readout".
    seed : int | None
        Base random seed for reproducibility.
    """

    target_devices: list[str] = field(default_factory=lambda: ["mock"])
    qubit_counts: list[int] = field(default_factory=lambda: [4, 6, 8])
    qaoa_depths: list[int] = field(default_factory=lambda: [1, 2, 3])
    qae_precisions: list[int] = field(default_factory=lambda: [3, 4, 5])
    shots: int = 4096
    n_runs: int = 5
    mitigation_methods: list[str] = field(
        default_factory=lambda: ["none", "zne", "trex"]
    )
    seed: int | None = 42


@dataclass
class HardwareBenchmarkResult:
    """Result from a single hardware benchmark run.

    Parameters
    ----------
    device_id : str
        Backend identifier used.
    circuit_type : str
        Type of circuit: "qaoa", "qae", "mitigation".
    n_qubits : int
        Number of qubits in the circuit.
    depth : int
        Circuit depth.
    approximation_ratio : float
        Ratio of achieved vs optimal objective (for QAOA).
    success_probability : float
        Probability of measuring the correct/target state.
    wall_clock_time : float
        Wall-clock time in seconds.
    raw_results : dict[str, Any]
        Raw measurement counts and metadata.
    mitigated_results : dict[str, Any]
        Results after error mitigation (if applied).
    confidence_interval : tuple[float, float]
        95% confidence interval for the primary metric.
    metadata : dict[str, Any]
        Additional benchmark metadata.
    """

    device_id: str = ""
    circuit_type: str = ""
    n_qubits: int = 0
    depth: int = 0
    approximation_ratio: float = 0.0
    success_probability: float = 0.0
    wall_clock_time: float = 0.0
    raw_results: dict[str, Any] = field(default_factory=dict)
    mitigated_results: dict[str, Any] = field(default_factory=dict)
    confidence_interval: tuple[float, float] = (0.0, 0.0)
    metadata: dict[str, Any] = field(default_factory=dict)


def _compute_success_probability(
    counts: dict[str, int], target_state: str, shots: int
) -> float:
    """Compute probability of measuring the target state."""
    return counts.get(target_state, 0) / max(shots, 1)


def _compute_confidence_interval(
    values: list[float], confidence: float = 0.95
) -> tuple[float, float]:
    """Compute confidence interval for a list of values.

    Uses t-distribution for small samples, normal for large.
    """
    if len(values) < 2:
        val = values[0] if values else 0.0
        return (val, val)

    arr = np.array(values, dtype=np.float64)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1))
    n = len(values)

    # Use 1.96 for 95% CI (normal approx for simplicity)
    z = 1.96 if confidence == 0.95 else 2.576
    margin = z * std / np.sqrt(n)
    return (mean - margin, mean + margin)


class HardwareBenchmarkRunner:
    """Runner for hardware validation benchmark campaigns.

    Executes QAOA and QAE circuits on specified backends, collects
    statistics over multiple runs, and generates reproducibility
    manifests.
    """

    def __init__(self, config: HardwareBenchmarkConfig | None = None) -> None:
        self._config = config or HardwareBenchmarkConfig()
        self._results: list[HardwareBenchmarkResult] = []

    @property
    def config(self) -> HardwareBenchmarkConfig:
        """Return the benchmark configuration."""
        return self._config

    @property
    def results(self) -> list[HardwareBenchmarkResult]:
        """Return collected results."""
        return list(self._results)

    def run_qaoa_benchmark(
        self,
        problem: Any,
        backend: Any,
        config: HardwareBenchmarkConfig | None = None,
    ) -> list[HardwareBenchmarkResult]:
        """Run QAOA benchmarks at various qubit counts and depths.

        Parameters
        ----------
        problem : Problem or dict
            Problem specification with QUBO parameters.
        backend : Backend
            Quantum backend to execute on.
        config : HardwareBenchmarkConfig | None
            Override config; uses self._config if None.

        Returns
        -------
        List of HardwareBenchmarkResult, one per (qubit_count, depth, run).
        """
        cfg = config or self._config
        results: list[HardwareBenchmarkResult] = []

        for n_qubits in cfg.qubit_counts:
            for depth in cfg.qaoa_depths:
                run_values: list[float] = []
                for run_idx in range(cfg.n_runs):
                    seed = (
                        cfg.seed + run_idx if cfg.seed is not None
                        else None
                    )
                    start = time.perf_counter()
                    circuit = _build_qaoa_circuit(n_qubits, depth, seed)

                    try:
                        cr = backend.run(circuit, shots=cfg.shots)
                        counts = cr.counts
                        device_id = backend.backend_id
                    except Exception as exc:
                        counts = {}
                        device_id = getattr(backend, "backend_id", "unknown")
                        counts["__error__"] = str(exc)

                    wall = time.perf_counter() - start

                    # Best bitstring approximation ratio proxy
                    if counts and "__error__" not in counts:
                        best_bs = max(counts, key=counts.__getitem__)
                        success_prob = counts[best_bs] / cfg.shots
                    else:
                        best_bs = ""
                        success_prob = 0.0

                    run_values.append(success_prob)

                    result = HardwareBenchmarkResult(
                        device_id=device_id,
                        circuit_type="qaoa",
                        n_qubits=n_qubits,
                        depth=depth,
                        approximation_ratio=success_prob,
                        success_probability=success_prob,
                        wall_clock_time=wall,
                        raw_results={"counts": counts, "best": best_bs},
                        confidence_interval=(0.0, 0.0),
                        metadata={
                            "run_index": run_idx,
                            "seed": seed,
                            "shots": cfg.shots,
                        },
                    )
                    results.append(result)

                # Attach CI to last result of this (n_qubits, depth) combo
                if run_values:
                    ci = _compute_confidence_interval(run_values)
                    results[-1].confidence_interval = ci

        self._results.extend(results)
        return results

    def run_qae_benchmark(
        self,
        backend: Any,
        config: HardwareBenchmarkConfig | None = None,
    ) -> list[HardwareBenchmarkResult]:
        """Run QAE benchmarks at various precision levels.

        Parameters
        ----------
        backend : Backend
            Quantum backend to execute on.
        config : HardwareBenchmarkConfig | None
            Override config; uses self._config if None.

        Returns
        -------
        List of HardwareBenchmarkResult, one per (precision, run).
        """
        cfg = config or self._config
        results: list[HardwareBenchmarkResult] = []

        for precision in cfg.qae_precisions:
            run_values: list[float] = []
            for run_idx in range(cfg.n_runs):
                seed = (
                    cfg.seed + run_idx if cfg.seed is not None
                    else None
                )
                start = time.perf_counter()
                n_qubits = precision + 1  # eval qubits + 1 target
                circuit = _build_qae_circuit(precision, seed)

                try:
                    cr = backend.run(circuit, shots=cfg.shots)
                    counts = cr.counts
                    device_id = backend.backend_id
                except Exception as exc:
                    counts = {}
                    device_id = getattr(backend, "backend_id", "unknown")
                    counts["__error__"] = str(exc)

                wall = time.perf_counter() - start

                # Estimate amplitude from phase measurement
                if counts and "__error__" not in counts:
                    best_bs = max(counts, key=counts.__getitem__)
                    # Phase estimate from binary fraction
                    phase_bits = best_bs[:precision]
                    phase = int(phase_bits, 2) / (2**precision) if phase_bits else 0.0
                    estimated_amp = np.sin(np.pi * phase) ** 2
                    success_prob = counts[best_bs] / cfg.shots
                else:
                    estimated_amp = 0.0
                    success_prob = 0.0

                run_values.append(estimated_amp)

                result = HardwareBenchmarkResult(
                    device_id=device_id,
                    circuit_type="qae",
                    n_qubits=n_qubits,
                    depth=circuit.depth(),
                    approximation_ratio=float(estimated_amp),
                    success_probability=success_prob,
                    wall_clock_time=wall,
                    raw_results={
                        "counts": counts,
                        "precision": precision,
                    },
                    confidence_interval=(0.0, 0.0),
                    metadata={
                        "run_index": run_idx,
                        "seed": seed,
                        "shots": cfg.shots,
                    },
                )
                results.append(result)

            if run_values:
                ci = _compute_confidence_interval(run_values)
                results[-1].confidence_interval = ci

        self._results.extend(results)
        return results

    def run_mitigation_comparison(
        self,
        circuit: Any,
        backend: Any,
    ) -> list[HardwareBenchmarkResult]:
        """Compare raw and mitigated results for a circuit.

        Tests each mitigation method configured in self._config and
        collects comparative results.

        Parameters
        ----------
        circuit : QuantumCircuit
            Circuit to benchmark (without measurements).
        backend : Backend
            Noisy backend to execute on.

        Returns
        -------
        List of HardwareBenchmarkResult, one per mitigation method.
        """
        cfg = self._config
        results: list[HardwareBenchmarkResult] = []
        n_qubits = circuit.num_qubits

        for method in cfg.mitigation_methods:
            start = time.perf_counter()

            if method == "none":
                # Run raw circuit with measurements
                meas_circ = circuit.copy()
                meas_circ.measure_all()
                cr = backend.run(meas_circ, shots=cfg.shots)
                counts = cr.counts
                mitigated = {}

            elif method == "zne":
                try:
                    from qufin.backends.error_mitigation import zne_extrapolate
                    zne_result = zne_extrapolate(
                        circuit, backend, shots=cfg.shots
                    )
                    # Also get raw counts for comparison
                    meas_circ = circuit.copy()
                    meas_circ.measure_all()
                    cr = backend.run(meas_circ, shots=cfg.shots)
                    counts = cr.counts
                    mitigated = {
                        "mitigated_value": zne_result["mitigated_value"],
                        "raw_values": zne_result["raw_values"],
                    }
                except Exception as exc:
                    counts = {"__error__": str(exc)}
                    mitigated = {}

            elif method == "trex":
                try:
                    from qufin.backends.error_mitigation import trex_mitigate
                    trex_result = trex_mitigate(
                        circuit, backend, shots_per_twirl=cfg.shots // 10
                    )
                    counts = trex_result.raw_counts
                    mitigated = {
                        "mitigated_probs": trex_result.mitigated_probs,
                    }
                except Exception as exc:
                    counts = {"__error__": str(exc)}
                    mitigated = {}

            elif method == "readout":
                try:
                    from qufin.backends.error_mitigation import (
                        calibrate_readout,
                        mitigate_readout,
                    )
                    cal_matrix = calibrate_readout(n_qubits, backend)
                    meas_circ = circuit.copy()
                    meas_circ.measure_all()
                    cr = backend.run(meas_circ, shots=cfg.shots)
                    counts = cr.counts
                    mit_result = mitigate_readout(
                        counts, cal_matrix, cfg.shots
                    )
                    mitigated = {
                        "mitigated_counts": mit_result.mitigated_counts,
                        "mitigated_probs": mit_result.mitigated_probs,
                    }
                except Exception as exc:
                    counts = {"__error__": str(exc)}
                    mitigated = {}

            else:
                counts = {}
                mitigated = {}

            wall = time.perf_counter() - start

            # Compute success probability
            if counts and "__error__" not in counts:
                target = "0" * n_qubits
                success_prob = _compute_success_probability(
                    counts, target, cfg.shots
                )
            else:
                success_prob = 0.0

            result = HardwareBenchmarkResult(
                device_id=getattr(backend, "backend_id", "unknown"),
                circuit_type="mitigation",
                n_qubits=n_qubits,
                depth=circuit.depth(),
                success_probability=success_prob,
                wall_clock_time=wall,
                raw_results={"counts": counts, "method": method},
                mitigated_results=mitigated,
                metadata={"mitigation_method": method, "shots": cfg.shots},
            )
            results.append(result)

        self._results.extend(results)
        return results

    def generate_manifest(
        self, results: list[HardwareBenchmarkResult] | None = None
    ) -> dict[str, Any]:
        """Generate a reproducibility manifest for benchmark results.

        Parameters
        ----------
        results : list[HardwareBenchmarkResult] | None
            Results to include. Uses self._results if None.

        Returns
        -------
        Dict with hardware info, config, and summary statistics.
        """
        import platform
        import sys

        res = results or self._results

        devices = list({r.device_id for r in res})
        circuit_types = list({r.circuit_type for r in res})

        # Package versions
        versions: dict[str, str] = {}
        for pkg in ["qufin", "qiskit", "qiskit-aer", "numpy"]:
            try:
                from importlib.metadata import version
                versions[pkg] = version(pkg)
            except Exception:
                pass

        return {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "python_version": sys.version,
            "platform": platform.platform(),
            "package_versions": versions,
            "config": asdict(self._config),
            "n_results": len(res),
            "devices": devices,
            "circuit_types": circuit_types,
            "summary": self.statistical_analysis(res),
        }

    def statistical_analysis(
        self,
        results: list[HardwareBenchmarkResult] | None = None,
    ) -> dict[str, Any]:
        """Compute summary statistics over benchmark results.

        Parameters
        ----------
        results : list[HardwareBenchmarkResult] | None
            Results to analyze. Uses self._results if None.

        Returns
        -------
        Dict with mean, std, 95% CI for key metrics, grouped by
        circuit type.
        """
        res = results or self._results
        if not res:
            return {}

        # Group by (circuit_type, n_qubits)
        groups: dict[str, list[HardwareBenchmarkResult]] = {}
        for r in res:
            key = f"{r.circuit_type}_q{r.n_qubits}"
            groups.setdefault(key, []).append(r)

        summary: dict[str, Any] = {}
        for key, group_results in groups.items():
            success_vals = [r.success_probability for r in group_results]
            wall_times = [r.wall_clock_time for r in group_results]

            success_arr = np.array(success_vals)
            wall_arr = np.array(wall_times)

            ci = _compute_confidence_interval(success_vals)

            summary[key] = {
                "n_runs": len(group_results),
                "success_probability_mean": float(np.mean(success_arr)),
                "success_probability_std": float(np.std(success_arr, ddof=1))
                if len(success_arr) > 1
                else 0.0,
                "success_probability_ci95": ci,
                "wall_time_mean": float(np.mean(wall_arr)),
                "wall_time_std": float(np.std(wall_arr, ddof=1))
                if len(wall_arr) > 1
                else 0.0,
            }

        return summary

    def save_results(
        self,
        results: list[HardwareBenchmarkResult] | None = None,
        path: str | Path = "benchmarks/hardware",
    ) -> Path:
        """Save benchmark results as JSON.

        Parameters
        ----------
        results : list[HardwareBenchmarkResult] | None
            Results to save. Uses self._results if None.
        path : str | Path
            Directory to save results in.

        Returns
        -------
        Path to the saved JSON file.
        """
        res = results or self._results
        out_dir = Path(path)
        out_dir.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        filename = out_dir / f"benchmark_{timestamp}.json"

        data = {
            "manifest": self.generate_manifest(res),
            "results": [_result_to_dict(r) for r in res],
        }

        filename.write_text(json.dumps(data, indent=2, default=str))
        return filename


class IonQBenchmarkRunner(HardwareBenchmarkRunner):
    """Benchmark runner specialized for IonQ devices via Amazon Braket.

    Extends HardwareBenchmarkRunner with IonQ-specific metrics:
    circuit depth comparison (native vs transpiled), 2Q gate count
    analysis, and cost estimation.
    """

    def __init__(
        self,
        config: HardwareBenchmarkConfig | None = None,
        device_arn: str | None = None,
    ) -> None:
        super().__init__(config)
        self._device_arn = device_arn

    def run_qaoa_benchmark(
        self,
        problem: Any,
        backend: Any,
        config: HardwareBenchmarkConfig | None = None,
    ) -> list[HardwareBenchmarkResult]:
        """Run QAOA benchmark with IonQ cost analysis.

        Extends the base runner with 2Q gate counting and
        cost estimates for IonQ Harmony/Aria.
        """
        results = super().run_qaoa_benchmark(problem, backend, config)

        # Add IonQ-specific metadata
        for result in results:
            n_qubits = result.n_qubits
            depth = result.depth
            # IonQ cost model: per-shot + per-gate pricing
            estimated_2q_gates = max(1, depth * (n_qubits - 1) // 2)
            cost_per_shot = 0.01  # approximate IonQ Harmony pricing
            estimated_cost = cost_per_shot * self._config.shots
            result.metadata["ionq_estimated_2q_gates"] = estimated_2q_gates
            result.metadata["ionq_estimated_cost_usd"] = estimated_cost
            result.metadata["device_arn"] = self._device_arn

        return results

    def cost_analysis(
        self,
        results: list[HardwareBenchmarkResult] | None = None,
    ) -> dict[str, float]:
        """Compute total estimated cost for all benchmark runs.

        Returns
        -------
        Dict with total_cost_usd, cost_per_run, total_shots.
        """
        res = results or self._results
        total_cost = sum(
            r.metadata.get("ionq_estimated_cost_usd", 0.0) for r in res
        )
        total_shots = sum(r.metadata.get("shots", 0) for r in res)
        n_runs = len(res) if res else 1

        return {
            "total_cost_usd": total_cost,
            "cost_per_run": total_cost / n_runs,
            "total_shots": total_shots,
            "n_runs": n_runs,
        }


# ---------------------------------------------------------------------------
# Helper functions (private)
# ---------------------------------------------------------------------------


def _build_qaoa_circuit(n_qubits: int, depth: int, seed: int | None = None) -> Any:
    """Build a simple QAOA-style circuit for benchmarking.

    Creates a parameterized circuit with alternating cost/mixer layers.
    Uses fixed angles for reproducibility.
    """
    from qiskit.circuit import QuantumCircuit

    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n_qubits, n_qubits)

    # Initial superposition
    for q in range(n_qubits):
        qc.h(q)

    # QAOA layers
    for _layer in range(depth):
        gamma = float(rng.uniform(0, 2 * np.pi))
        beta = float(rng.uniform(0, np.pi))

        # Cost layer: ZZ interactions on nearest-neighbor pairs
        for q in range(n_qubits - 1):
            qc.cx(q, q + 1)
            qc.rz(gamma, q + 1)
            qc.cx(q, q + 1)

        # Mixer layer: X rotations
        for q in range(n_qubits):
            qc.rx(2 * beta, q)

    # Measurements
    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def _build_qae_circuit(precision: int, seed: int | None = None) -> Any:
    """Build a simple QAE-style circuit for benchmarking.

    Creates a phase estimation circuit with a Grover-style oracle.
    """
    from qiskit.circuit import QuantumCircuit

    n_qubits = precision + 1  # evaluation + target
    qc = QuantumCircuit(n_qubits, precision)

    # Prepare target qubit in superposition
    qc.ry(np.pi / 3, n_qubits - 1)

    # Phase estimation registers
    for q in range(precision):
        qc.h(q)

    # Controlled rotations (simplified QPE structure)
    for q in range(precision):
        angle = np.pi / (2**q)
        qc.cp(angle, q, n_qubits - 1)

    # Inverse QFT on evaluation register
    for q in range(precision // 2):
        qc.swap(q, precision - 1 - q)
    for q in range(precision):
        for j in range(q):
            qc.cp(-np.pi / (2 ** (q - j)), j, q)
        qc.h(q)

    # Measure evaluation register
    qc.measure(range(precision), range(precision))
    return qc


def _result_to_dict(result: HardwareBenchmarkResult) -> dict[str, Any]:
    """Convert a HardwareBenchmarkResult to a JSON-serializable dict."""
    d = asdict(result)
    # Convert tuple to list for JSON
    d["confidence_interval"] = list(d["confidence_interval"])
    return d
