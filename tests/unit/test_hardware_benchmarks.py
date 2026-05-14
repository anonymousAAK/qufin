"""Tests for the hardware benchmark framework."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from qufin.benchmarks.hardware_benchmarks import (
    HardwareBenchmarkConfig,
    HardwareBenchmarkResult,
    HardwareBenchmarkRunner,
    IonQBenchmarkRunner,
    _compute_confidence_interval,
    _compute_success_probability,
)

# ---------------------------------------------------------------------------
# Stub backend for testing (avoids Qiskit Aer dependency in unit tests)
# ---------------------------------------------------------------------------


class StubBackend:
    """Minimal backend stub for testing."""

    @property
    def backend_id(self) -> str:
        return "stub-test"

    def run(self, circuit: Any, shots: int = 1024) -> Any:
        n = circuit.num_qubits
        # Return a deterministic counts dict
        target = "0" * n
        alt = "1" * n
        return _StubResult(
            counts={target: shots // 2, alt: shots - shots // 2},
            shots=shots,
        )


class _StubResult:
    def __init__(self, counts: dict, shots: int) -> None:
        self.counts = counts
        self.shots = shots
        self.backend_id = "stub-test"


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestHardwareBenchmarkConfig:
    def test_defaults(self) -> None:
        cfg = HardwareBenchmarkConfig()
        assert cfg.shots == 4096
        assert cfg.n_runs == 5
        assert cfg.seed == 42
        assert "none" in cfg.mitigation_methods

    def test_custom_config(self) -> None:
        cfg = HardwareBenchmarkConfig(
            target_devices=["ibm_brisbane"],
            qubit_counts=[4],
            qaoa_depths=[1],
            shots=1000,
            n_runs=2,
        )
        assert cfg.shots == 1000
        assert cfg.n_runs == 2
        assert cfg.target_devices == ["ibm_brisbane"]


# ---------------------------------------------------------------------------
# Result tests
# ---------------------------------------------------------------------------


class TestHardwareBenchmarkResult:
    def test_defaults(self) -> None:
        r = HardwareBenchmarkResult()
        assert r.device_id == ""
        assert r.n_qubits == 0
        assert r.confidence_interval == (0.0, 0.0)
        assert r.raw_results == {}

    def test_custom_result(self) -> None:
        r = HardwareBenchmarkResult(
            device_id="test-device",
            circuit_type="qaoa",
            n_qubits=4,
            depth=10,
            success_probability=0.85,
            confidence_interval=(0.80, 0.90),
        )
        assert r.success_probability == 0.85
        assert r.confidence_interval[0] == 0.80


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_success_probability(self) -> None:
        counts = {"000": 800, "111": 200}
        assert _compute_success_probability(counts, "000", 1000) == 0.8

    def test_success_probability_missing(self) -> None:
        counts = {"111": 1000}
        assert _compute_success_probability(counts, "000", 1000) == 0.0

    def test_confidence_interval_single(self) -> None:
        ci = _compute_confidence_interval([0.5])
        assert ci == (0.5, 0.5)

    def test_confidence_interval_multiple(self) -> None:
        values = [0.8, 0.82, 0.78, 0.81, 0.79]
        ci = _compute_confidence_interval(values)
        mean = np.mean(values)
        assert ci[0] < mean < ci[1]
        assert ci[0] > 0.0
        assert ci[1] < 1.0


# ---------------------------------------------------------------------------
# Runner tests
# ---------------------------------------------------------------------------


class TestHardwareBenchmarkRunner:
    def test_init_default_config(self) -> None:
        runner = HardwareBenchmarkRunner()
        assert runner.config.shots == 4096
        assert runner.results == []

    def test_run_qaoa_benchmark(self) -> None:
        cfg = HardwareBenchmarkConfig(
            qubit_counts=[4],
            qaoa_depths=[1],
            shots=100,
            n_runs=2,
            seed=42,
        )
        runner = HardwareBenchmarkRunner(cfg)
        backend = StubBackend()
        results = runner.run_qaoa_benchmark({}, backend)
        assert len(results) == 2  # 1 qubit_count * 1 depth * 2 runs
        assert all(r.circuit_type == "qaoa" for r in results)
        assert all(r.device_id == "stub-test" for r in results)
        assert all(r.n_qubits == 4 for r in results)

    def test_run_qae_benchmark(self) -> None:
        cfg = HardwareBenchmarkConfig(
            qae_precisions=[3],
            shots=100,
            n_runs=2,
            seed=42,
        )
        runner = HardwareBenchmarkRunner(cfg)
        backend = StubBackend()
        results = runner.run_qae_benchmark(backend)
        assert len(results) == 2  # 1 precision * 2 runs
        assert all(r.circuit_type == "qae" for r in results)

    def test_results_accumulate(self) -> None:
        cfg = HardwareBenchmarkConfig(
            qubit_counts=[4],
            qaoa_depths=[1],
            qae_precisions=[3],
            shots=100,
            n_runs=1,
        )
        runner = HardwareBenchmarkRunner(cfg)
        backend = StubBackend()
        runner.run_qaoa_benchmark({}, backend)
        runner.run_qae_benchmark(backend)
        assert len(runner.results) == 2

    def test_statistical_analysis(self) -> None:
        cfg = HardwareBenchmarkConfig(
            qubit_counts=[4],
            qaoa_depths=[1],
            shots=100,
            n_runs=3,
        )
        runner = HardwareBenchmarkRunner(cfg)
        backend = StubBackend()
        results = runner.run_qaoa_benchmark({}, backend)
        analysis = runner.statistical_analysis(results)
        assert "qaoa_q4" in analysis
        assert "success_probability_mean" in analysis["qaoa_q4"]
        assert analysis["qaoa_q4"]["n_runs"] == 3

    def test_statistical_analysis_empty(self) -> None:
        runner = HardwareBenchmarkRunner()
        assert runner.statistical_analysis([]) == {}

    def test_generate_manifest(self) -> None:
        cfg = HardwareBenchmarkConfig(
            qubit_counts=[4],
            qaoa_depths=[1],
            shots=100,
            n_runs=1,
        )
        runner = HardwareBenchmarkRunner(cfg)
        backend = StubBackend()
        results = runner.run_qaoa_benchmark({}, backend)
        manifest = runner.generate_manifest(results)
        assert "timestamp" in manifest
        assert "python_version" in manifest
        assert "config" in manifest
        assert manifest["n_results"] == 1

    def test_save_results(self, tmp_path: Path) -> None:
        cfg = HardwareBenchmarkConfig(
            qubit_counts=[4],
            qaoa_depths=[1],
            shots=100,
            n_runs=1,
        )
        runner = HardwareBenchmarkRunner(cfg)
        backend = StubBackend()
        results = runner.run_qaoa_benchmark({}, backend)
        out_path = runner.save_results(results, path=tmp_path / "hw")
        assert out_path.exists()
        data = json.loads(out_path.read_text())
        assert "manifest" in data
        assert "results" in data
        assert len(data["results"]) == 1


# ---------------------------------------------------------------------------
# IonQ runner tests
# ---------------------------------------------------------------------------


class TestIonQBenchmarkRunner:
    def test_inherits_runner(self) -> None:
        runner = IonQBenchmarkRunner()
        assert isinstance(runner, HardwareBenchmarkRunner)

    def test_qaoa_adds_ionq_metadata(self) -> None:
        cfg = HardwareBenchmarkConfig(
            qubit_counts=[4],
            qaoa_depths=[1],
            shots=100,
            n_runs=1,
        )
        runner = IonQBenchmarkRunner(
            config=cfg, device_arn="arn:aws:braket:::device/qpu/ionq/Harmony"
        )
        backend = StubBackend()
        results = runner.run_qaoa_benchmark({}, backend)
        assert len(results) == 1
        assert "ionq_estimated_2q_gates" in results[0].metadata
        assert "ionq_estimated_cost_usd" in results[0].metadata

    def test_cost_analysis(self) -> None:
        cfg = HardwareBenchmarkConfig(
            qubit_counts=[4],
            qaoa_depths=[1],
            shots=100,
            n_runs=2,
        )
        runner = IonQBenchmarkRunner(config=cfg)
        backend = StubBackend()
        runner.run_qaoa_benchmark({}, backend)
        cost = runner.cost_analysis()
        assert cost["total_cost_usd"] > 0
        assert cost["n_runs"] == 2
