"""Tests for the resource estimation suite."""

from __future__ import annotations

from qufin.benchmarks.resource_estimation import (
    HARDWARE_ROADMAP,
    AlgorithmResource,
    BreakEvenPoint,
    SurfaceCodeOverhead,
    break_even_summary,
    compute_break_even_timeline,
    compute_surface_code_overhead,
    estimate_qae_pricing,
    estimate_qaoa_optimization,
    estimate_quantum_credit,
    estimate_quantum_var,
    generate_resource_table,
    plot_break_even_timeline,
    plot_resource_table,
    resource_table_to_dicts,
)

# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------


class TestAlgorithmResource:
    def test_defaults(self) -> None:
        r = AlgorithmResource()
        assert r.algorithm == ""
        assert r.problem_size == 0
        assert r.n_logical_qubits == 0
        assert r.t_gate_count == 0
        assert r.metadata == {}

    def test_custom(self) -> None:
        r = AlgorithmResource(
            algorithm="QAE pricing",
            problem_size=8,
            n_logical_qubits=50,
            t_gate_count=100000,
        )
        assert r.algorithm == "QAE pricing"
        assert r.problem_size == 8


class TestSurfaceCodeOverhead:
    def test_defaults(self) -> None:
        s = SurfaceCodeOverhead()
        assert s.code_distance == 0
        assert s.n_physical_qubits == 0
        assert s.distillation_qubits == 0

    def test_custom(self) -> None:
        s = SurfaceCodeOverhead(
            n_logical_qubits=20,
            code_distance=17,
            n_physical_qubits=12000,
        )
        assert s.n_logical_qubits == 20


class TestBreakEvenPoint:
    def test_defaults(self) -> None:
        b = BreakEvenPoint()
        assert b.algorithm == ""
        assert b.break_even_year == 0
        assert b.is_practical is False

    def test_custom(self) -> None:
        b = BreakEvenPoint(
            algorithm="QAOA",
            break_even_year=2028,
            is_practical=True,
            hardware_target="IBM",
        )
        assert b.break_even_year == 2028


# ---------------------------------------------------------------------------
# Per-algorithm estimation tests
# ---------------------------------------------------------------------------


class TestEstimateQAEPricing:
    def test_basic(self) -> None:
        res = estimate_qae_pricing()
        assert res.algorithm == "QAE pricing"
        assert res.n_logical_qubits > 0
        assert res.t_gate_count > 0
        assert res.n_physical_qubits > 0
        assert res.runtime_seconds > 0

    def test_scaling_with_assets(self) -> None:
        r1 = estimate_qae_pricing(n_assets=1)
        r3 = estimate_qae_pricing(n_assets=3)
        assert r3.n_logical_qubits > r1.n_logical_qubits
        assert r3.t_gate_count > r1.t_gate_count

    def test_scaling_with_precision(self) -> None:
        r4 = estimate_qae_pricing(n_precision_qubits=4)
        r8 = estimate_qae_pricing(n_precision_qubits=8)
        assert r8.t_gate_count > r4.t_gate_count

    def test_metadata(self) -> None:
        res = estimate_qae_pricing(n_price_qubits=6, n_precision_qubits=10, n_assets=2)
        assert res.metadata["n_price_qubits"] == 6
        assert res.metadata["n_precision_qubits"] == 10
        assert res.metadata["n_assets"] == 2
        assert res.metadata["n_iterations"] == 2**10


class TestEstimateQAOAOptimization:
    def test_basic(self) -> None:
        res = estimate_qaoa_optimization()
        assert res.algorithm == "QAOA optimisation"
        assert res.n_logical_qubits == 4  # n_assets
        assert res.t_gate_count > 0

    def test_scaling_with_assets(self) -> None:
        r4 = estimate_qaoa_optimization(n_assets=4)
        r10 = estimate_qaoa_optimization(n_assets=10)
        assert r10.n_logical_qubits > r4.n_logical_qubits
        # More assets -> more ZZ interactions -> more T-gates
        assert r10.t_gate_count > r4.t_gate_count

    def test_scaling_with_layers(self) -> None:
        r1 = estimate_qaoa_optimization(p_layers=1)
        r5 = estimate_qaoa_optimization(p_layers=5)
        assert r5.t_gate_count > r1.t_gate_count

    def test_metadata(self) -> None:
        res = estimate_qaoa_optimization(n_assets=6, p_layers=3)
        assert res.metadata["n_assets"] == 6
        assert res.metadata["p_layers"] == 3
        assert res.metadata["n_zz_interactions"] == 15  # 6*5/2


class TestEstimateQuantumVar:
    def test_basic(self) -> None:
        res = estimate_quantum_var()
        assert res.algorithm == "Quantum VaR"
        assert res.n_logical_qubits > 0
        assert res.t_gate_count > 0

    def test_scaling_with_assets(self) -> None:
        r2 = estimate_quantum_var(n_assets=2)
        r8 = estimate_quantum_var(n_assets=8)
        assert r8.n_logical_qubits > r2.n_logical_qubits

    def test_metadata(self) -> None:
        res = estimate_quantum_var(n_assets=5, n_price_qubits=6)
        assert res.metadata["n_assets"] == 5
        assert res.metadata["n_price_qubits"] == 6


class TestEstimateQuantumCredit:
    def test_basic(self) -> None:
        res = estimate_quantum_credit()
        assert res.algorithm == "Quantum credit risk"
        assert res.n_logical_qubits > 0

    def test_scaling_with_obligors(self) -> None:
        r5 = estimate_quantum_credit(n_obligors=5)
        r50 = estimate_quantum_credit(n_obligors=50)
        assert r50.n_logical_qubits > r5.n_logical_qubits
        assert r50.t_gate_count > r5.t_gate_count

    def test_metadata(self) -> None:
        res = estimate_quantum_credit(n_obligors=20)
        assert res.metadata["n_obligors"] == 20


# ---------------------------------------------------------------------------
# Resource table tests
# ---------------------------------------------------------------------------


class TestGenerateResourceTable:
    def test_default(self) -> None:
        table = generate_resource_table()
        # 4 problem sizes * 4 algorithms = 16
        assert len(table) == 16
        algorithms = {r.algorithm for r in table}
        assert "QAE pricing" in algorithms
        assert "QAOA optimisation" in algorithms
        assert "Quantum VaR" in algorithms
        assert "Quantum credit risk" in algorithms

    def test_custom_sizes(self) -> None:
        table = generate_resource_table(problem_sizes=[4, 8])
        assert len(table) == 8  # 2 sizes * 4 algorithms

    def test_all_positive_values(self) -> None:
        table = generate_resource_table(problem_sizes=[4])
        for r in table:
            assert r.n_logical_qubits > 0
            assert r.t_gate_count > 0
            assert r.n_physical_qubits > 0


class TestResourceTableToDicts:
    def test_conversion(self) -> None:
        table = generate_resource_table(problem_sizes=[4])
        dicts = resource_table_to_dicts(table)
        assert len(dicts) == 4
        for d in dicts:
            assert "algorithm" in d
            assert "problem_size" in d
            assert "logical_qubits" in d
            assert "t_gates" in d
            assert "physical_qubits_d17" in d
            assert "circuit_depth" in d
            assert "runtime_s" in d

    def test_empty(self) -> None:
        dicts = resource_table_to_dicts([])
        assert dicts == []


# ---------------------------------------------------------------------------
# Surface code overhead tests
# ---------------------------------------------------------------------------


class TestComputeSurfaceCodeOverhead:
    def test_basic(self) -> None:
        overhead = compute_surface_code_overhead(
            n_logical_qubits=20,
            t_gate_count=10000,
        )
        assert overhead.code_distance >= 3
        assert overhead.code_distance % 2 == 1  # odd distance
        assert overhead.n_physical_qubits > 0
        assert overhead.physical_per_logical > 0
        assert overhead.distillation_qubits > 0
        assert overhead.n_rounds > 0

    def test_higher_error_rate_needs_larger_distance(self) -> None:
        low_err = compute_surface_code_overhead(
            n_logical_qubits=10,
            t_gate_count=1000,
            physical_error_rate=1e-4,
        )
        high_err = compute_surface_code_overhead(
            n_logical_qubits=10,
            t_gate_count=1000,
            physical_error_rate=5e-3,
        )
        assert high_err.code_distance >= low_err.code_distance

    def test_below_threshold(self) -> None:
        """When error rate is above threshold, uses minimum distance."""
        overhead = compute_surface_code_overhead(
            n_logical_qubits=10,
            t_gate_count=1000,
            physical_error_rate=0.05,  # 5% -> above threshold
        )
        assert overhead.code_distance == 3

    def test_more_qubits_more_physical(self) -> None:
        small = compute_surface_code_overhead(n_logical_qubits=5, t_gate_count=1000)
        large = compute_surface_code_overhead(n_logical_qubits=50, t_gate_count=1000)
        assert large.n_physical_qubits > small.n_physical_qubits

    def test_runtime_scaling(self) -> None:
        short = compute_surface_code_overhead(
            n_logical_qubits=10, t_gate_count=100
        )
        long = compute_surface_code_overhead(
            n_logical_qubits=10, t_gate_count=100000
        )
        assert long.runtime_seconds > short.runtime_seconds


# ---------------------------------------------------------------------------
# Break-even timeline tests
# ---------------------------------------------------------------------------


class TestHardwareRoadmap:
    def test_roadmap_has_vendors(self) -> None:
        assert "IBM" in HARDWARE_ROADMAP
        assert "IonQ" in HARDWARE_ROADMAP
        assert "QuEra" in HARDWARE_ROADMAP

    def test_roadmap_entries_have_fields(self) -> None:
        for _vendor, entries in HARDWARE_ROADMAP.items():
            assert len(entries) > 0
            for entry in entries:
                assert "year" in entry
                assert "qubits" in entry
                assert "error_rate" in entry


class TestComputeBreakEvenTimeline:
    def test_returns_results(self) -> None:
        timeline = compute_break_even_timeline()
        assert len(timeline) > 0
        for bp in timeline:
            assert isinstance(bp, BreakEvenPoint)
            assert bp.algorithm != ""
            assert bp.hardware_target != ""

    def test_custom_algorithms(self) -> None:
        algos = [estimate_qaoa_optimization(n_assets=4, p_layers=1)]
        timeline = compute_break_even_timeline(algorithms=algos)
        # 1 algorithm * 3 vendors
        assert len(timeline) == 3
        assert all(bp.algorithm == "QAOA optimisation" for bp in timeline)

    def test_infeasible_algorithm(self) -> None:
        """Very large algorithm should be infeasible on all hardware."""
        huge = AlgorithmResource(
            algorithm="Huge",
            n_logical_qubits=1000000,
            t_gate_count=10**15,
            n_physical_qubits=10**9,
        )
        timeline = compute_break_even_timeline(algorithms=[huge])
        for bp in timeline:
            assert bp.is_practical is False
            assert "exceeds" in bp.notes


class TestBreakEvenSummary:
    def test_returns_summary(self) -> None:
        summary = break_even_summary()
        assert isinstance(summary, dict)
        assert len(summary) > 0
        for _algo, info in summary.items():
            assert "earliest_break_even" in info
            assert "earliest_vendor" in info
            assert "n_vendors_feasible" in info
            assert "vendors" in info

    def test_custom_timeline(self) -> None:
        timeline = [
            BreakEvenPoint(
                algorithm="TestAlgo",
                break_even_year=2028,
                is_practical=True,
                hardware_target="IBM",
            ),
            BreakEvenPoint(
                algorithm="TestAlgo",
                break_even_year=0,
                is_practical=False,
                hardware_target="IonQ",
                notes="too large",
            ),
        ]
        summary = break_even_summary(timeline)
        assert "TestAlgo" in summary
        assert summary["TestAlgo"]["earliest_break_even"] == 2028
        assert summary["TestAlgo"]["earliest_vendor"] == "IBM"
        assert summary["TestAlgo"]["n_vendors_feasible"] == 1


# ---------------------------------------------------------------------------
# Visualisation tests (guarded)
# ---------------------------------------------------------------------------


class TestVisualization:
    def test_plot_resource_table_returns(self) -> None:
        """plot_resource_table should return a figure or None."""
        table = generate_resource_table(problem_sizes=[4])
        result = plot_resource_table(table)
        # Either a Plotly figure or None if plotly not installed
        assert result is None or hasattr(result, "update_layout")

    def test_plot_break_even_timeline_returns(self) -> None:
        result = plot_break_even_timeline()
        assert result is None or hasattr(result, "update_layout")

    def test_plot_resource_empty(self) -> None:
        result = plot_resource_table([])
        assert result is None or hasattr(result, "update_layout")
