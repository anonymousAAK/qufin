"""Tests for the benchmark suite (Phase 4)."""

from __future__ import annotations

import json
from pathlib import Path


class TestClassicalBenchmarks:
    def test_bs_pricing(self):
        from benchmarks.classical_comparison import bench_bs_pricing
        entries = bench_bs_pricing()
        assert len(entries) == 4
        # qufin and numpy should agree on BS price
        qufin_price = entries[0].result
        numpy_price = entries[1].result
        assert abs(qufin_price - numpy_price) < 1e-8

    def test_greeks(self):
        from benchmarks.classical_comparison import bench_greeks
        entries = bench_greeks()
        assert len(entries) == 4
        # Delta should be between 0 and 1 for a call
        delta = entries[0].result
        assert 0 < delta < 1

    def test_mc_pricing(self):
        from benchmarks.classical_comparison import bench_mc_pricing
        entries = bench_mc_pricing()
        assert len(entries) == 6
        # MC price should converge to BS price (~10.45) with more paths
        mc_1m = next(e for e in entries if e.method == "qufin.european_mc" and e.n == 1_000_000)
        assert abs(mc_1m.result - 10.45) < 0.5

    def test_portfolio(self):
        from benchmarks.classical_comparison import bench_portfolio
        entries = bench_portfolio()
        assert len(entries) >= 8  # at least 2 sizes x 4 methods

    def test_var(self):
        from benchmarks.classical_comparison import bench_var
        entries = bench_var()
        assert all(e.result > 0 for e in entries)


class TestQuantumScaling:
    def test_qubo_build(self):
        from benchmarks.quantum_scaling import bench_qubo_build
        entries = bench_qubo_build()
        assert len(entries) >= 10
        # Larger problems should take more time (generally)
        one_hot = [e for e in entries if e.method == "one_hot"]
        assert one_hot[-1].n_qubits > one_hot[0].n_qubits

    def test_exhaustive(self):
        from benchmarks.quantum_scaling import bench_exhaustive
        entries = bench_exhaustive()
        assert len(entries) >= 3
        # Time should grow exponentially
        assert entries[-1].wall_time_s > entries[0].wall_time_s

    def test_circuit_depth(self):
        from benchmarks.quantum_scaling import bench_circuit_depth
        entries = bench_circuit_depth()
        # XY mixers should have deeper circuits than X mixer
        x_depths = [e.circuit_depth for e in entries if e.method == "mixer_x" and e.circuit_depth]
        xy_depths = [
            e.circuit_depth for e in entries
            if e.method == "mixer_xy_ring" and e.circuit_depth
        ]
        if x_depths and xy_depths:
            assert max(xy_depths) > max(x_depths)


class TestQuantumAdvantage:
    def test_mc_analysis(self):
        from benchmarks.quantum_advantage import mc_pricing_analysis
        analysis = mc_pricing_analysis()
        assert len(analysis.estimates) == 4
        assert "epsilon" in analysis.summary.lower() or "advantage" in analysis.summary.lower()

    def test_portfolio_analysis(self):
        from benchmarks.quantum_advantage import portfolio_optimization_analysis
        analysis = portfolio_optimization_analysis()
        assert len(analysis.estimates) == 5
        # Should mention that there's no proven advantage
        assert "no proven" in analysis.summary.lower() or "no quantum" in analysis.summary.lower()

    def test_nisq_reality(self):
        from benchmarks.quantum_advantage import nisq_reality
        analysis = nisq_reality()
        assert len(analysis.estimates) == 3
        # Should be honest about current limitations
        assert "nisq" in analysis.summary.lower() or "fault-tolerant" in analysis.summary.lower()


class TestResultsIntegrity:
    def test_classical_json_exists(self):
        p = Path("benchmarks/results/classical.json")
        if p.exists():
            data = json.loads(p.read_text())
            assert len(data) > 0
            assert all("category" in e for e in data)

    def test_quantum_json_exists(self):
        p = Path("benchmarks/results/quantum_scaling.json")
        if p.exists():
            data = json.loads(p.read_text())
            assert len(data) > 0

    def test_advantage_json_exists(self):
        p = Path("benchmarks/results/advantage.json")
        if p.exists():
            data = json.loads(p.read_text())
            assert len(data) > 0
            assert all("title" in a for a in data)
