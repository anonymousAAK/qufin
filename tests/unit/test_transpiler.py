"""Tests for the finance-optimized circuit transpiler."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit.circuit import QuantumCircuit

from qufin.backends.transpiler import (
    FinanceTranspiler,
    TranspilationResult,
    find_commuting_groups,
    initial_layout_from_qubo,
    qubo_interaction_graph,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_qubo() -> np.ndarray:
    """A 4x4 QUBO matrix with known structure."""
    Q = np.array([
        [1.0, 0.5, 0.0, 0.0],
        [0.5, 2.0, 0.3, 0.0],
        [0.0, 0.3, 1.5, 0.7],
        [0.0, 0.0, 0.7, 1.0],
    ])
    return Q


@pytest.fixture
def linear_coupling_map() -> list[tuple[int, int]]:
    """Linear coupling map: 0-1-2-3-4-5."""
    return [(i, i + 1) for i in range(5)]


@pytest.fixture
def qaoa_circuit() -> QuantumCircuit:
    """Simple 4-qubit QAOA-style circuit."""
    qc = QuantumCircuit(4)
    for q in range(4):
        qc.h(q)
    # Cost layer
    for q in range(3):
        qc.cx(q, q + 1)
        qc.rz(0.5, q + 1)
        qc.cx(q, q + 1)
    # Mixer layer
    for q in range(4):
        qc.rx(0.3, q)
    return qc


@pytest.fixture
def transpiler() -> FinanceTranspiler:
    return FinanceTranspiler(optimization_level=2, seed=42)


# ---------------------------------------------------------------------------
# TranspilationResult tests
# ---------------------------------------------------------------------------


class TestTranspilationResult:
    def test_depth_reduction_nonzero(self) -> None:
        r = TranspilationResult(original_depth=100, optimized_depth=60)
        assert r.depth_reduction == pytest.approx(0.4)

    def test_depth_reduction_zero_original(self) -> None:
        r = TranspilationResult(original_depth=0, optimized_depth=0)
        assert r.depth_reduction == 0.0

    def test_cx_reduction(self) -> None:
        r = TranspilationResult(original_cx_count=50, optimized_cx_count=30)
        assert r.cx_reduction == pytest.approx(0.4)

    def test_cx_reduction_zero_original(self) -> None:
        r = TranspilationResult(original_cx_count=0, optimized_cx_count=0)
        assert r.cx_reduction == 0.0

    def test_dataclass_defaults(self) -> None:
        r = TranspilationResult()
        assert r.original_depth == 0
        assert r.optimized_depth == 0
        assert r.method_used == ""
        assert r.metadata == {}


# ---------------------------------------------------------------------------
# qubo_interaction_graph tests
# ---------------------------------------------------------------------------


class TestQuboInteractionGraph:
    def test_symmetric_qubo(self, simple_qubo: np.ndarray) -> None:
        graph = qubo_interaction_graph(simple_qubo)
        # Check expected edges exist
        assert (0, 1) in graph
        assert (1, 2) in graph
        assert (2, 3) in graph
        # No interaction between 0,2 or 0,3 or 1,3
        assert (0, 2) not in graph
        assert (0, 3) not in graph
        assert (1, 3) not in graph

    def test_weights_correct(self, simple_qubo: np.ndarray) -> None:
        graph = qubo_interaction_graph(simple_qubo)
        # Q[0,1] + Q[1,0] = 0.5 + 0.5 = 1.0
        assert graph[(0, 1)] == pytest.approx(1.0)
        # Q[1,2] + Q[2,1] = 0.3 + 0.3 = 0.6
        assert graph[(1, 2)] == pytest.approx(0.6)

    def test_empty_qubo(self) -> None:
        Q = np.diag([1.0, 2.0, 3.0])
        graph = qubo_interaction_graph(Q)
        assert len(graph) == 0

    def test_dense_qubo(self) -> None:
        Q = np.ones((3, 3))
        graph = qubo_interaction_graph(Q)
        # All off-diagonal pairs should be present
        assert len(graph) == 3  # (0,1), (0,2), (1,2)


# ---------------------------------------------------------------------------
# find_commuting_groups tests
# ---------------------------------------------------------------------------


class TestFindCommutingGroups:
    def test_disjoint_gates(self) -> None:
        qc = QuantumCircuit(4)
        qc.h(0)
        qc.h(2)
        qc.x(1)
        qc.x(3)
        groups = find_commuting_groups(qc)
        # All gates on disjoint qubits could form one group
        assert len(groups) >= 1
        total_gates = sum(len(g) for g in groups)
        assert total_gates == 4

    def test_overlapping_gates(self) -> None:
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.h(1)
        groups = find_commuting_groups(qc)
        # H(0) and CX(0,1) share qubit 0, cannot be in same group
        assert len(groups) >= 2

    def test_empty_circuit(self) -> None:
        qc = QuantumCircuit(2)
        groups = find_commuting_groups(qc)
        assert len(groups) == 0


# ---------------------------------------------------------------------------
# initial_layout_from_qubo tests
# ---------------------------------------------------------------------------


class TestInitialLayoutFromQubo:
    def test_layout_covers_all_qubits(
        self, simple_qubo: np.ndarray, linear_coupling_map: list
    ) -> None:
        layout = initial_layout_from_qubo(simple_qubo, linear_coupling_map)
        n = simple_qubo.shape[0]
        assert len(layout) == n
        for i in range(n):
            assert i in layout

    def test_layout_unique_physical(
        self, simple_qubo: np.ndarray, linear_coupling_map: list
    ) -> None:
        layout = initial_layout_from_qubo(simple_qubo, linear_coupling_map)
        physical = list(layout.values())
        assert len(physical) == len(set(physical))

    def test_high_weight_adjacent(self) -> None:
        # Q with strong (0,1) interaction
        Q = np.zeros((3, 3))
        Q[0, 1] = 10.0
        Q[1, 0] = 10.0
        coupling = [(0, 1), (1, 2)]
        layout = initial_layout_from_qubo(Q, coupling)
        # Qubits 0 and 1 should map to adjacent physical qubits
        p0, p1 = layout[0], layout[1]
        assert (min(p0, p1), max(p0, p1)) in coupling


# ---------------------------------------------------------------------------
# FinanceTranspiler tests
# ---------------------------------------------------------------------------


class TestFinanceTranspiler:
    def test_optimize_qaoa_circuit(
        self,
        transpiler: FinanceTranspiler,
        qaoa_circuit: QuantumCircuit,
        simple_qubo: np.ndarray,
    ) -> None:
        optimized, result = transpiler.optimize_qaoa_circuit(
            qaoa_circuit, simple_qubo
        )
        assert isinstance(result, TranspilationResult)
        assert result.method_used == "qaoa_structure_aware"
        assert result.original_depth > 0
        assert optimized.num_qubits == 4

    def test_parallelize_commuting_zz(
        self, transpiler: FinanceTranspiler, qaoa_circuit: QuantumCircuit
    ) -> None:
        optimized, result = transpiler.parallelize_commuting_zz(qaoa_circuit)
        assert result.method_used == "commuting_zz_parallel"
        assert "n_commuting_groups" in result.metadata
        assert optimized.num_qubits == qaoa_circuit.num_qubits

    def test_reduce_cnot_count(
        self, transpiler: FinanceTranspiler, qaoa_circuit: QuantumCircuit
    ) -> None:
        optimized, result = transpiler.reduce_cnot_count(qaoa_circuit)
        assert result.method_used == "cnot_reduction_level3"
        assert result.original_cx_count >= 0
        assert optimized.num_qubits == qaoa_circuit.num_qubits

    def test_connectivity_aware_routing(
        self,
        transpiler: FinanceTranspiler,
        qaoa_circuit: QuantumCircuit,
        linear_coupling_map: list,
        simple_qubo: np.ndarray,
    ) -> None:
        _optimized, result = transpiler.connectivity_aware_routing(
            qaoa_circuit, linear_coupling_map, simple_qubo
        )
        assert result.method_used == "qubo_aware_routing"
        assert "initial_layout" in result.metadata

    def test_benchmark_transpilation_default(
        self, transpiler: FinanceTranspiler, qaoa_circuit: QuantumCircuit
    ) -> None:
        results = transpiler.benchmark_transpilation(qaoa_circuit)
        assert "level0" in results
        assert "level3" in results
        for method, res in results.items():
            assert isinstance(res, TranspilationResult)
            assert res.method_used == method

    def test_benchmark_transpilation_custom_methods(
        self, transpiler: FinanceTranspiler, qaoa_circuit: QuantumCircuit
    ) -> None:
        results = transpiler.benchmark_transpilation(
            qaoa_circuit, methods=["level1", "level2"]
        )
        assert len(results) == 2
        assert "level1" in results
        assert "level2" in results

    def test_estimated_error_rate_nonnegative(
        self,
        transpiler: FinanceTranspiler,
        qaoa_circuit: QuantumCircuit,
        simple_qubo: np.ndarray,
    ) -> None:
        _, result = transpiler.optimize_qaoa_circuit(qaoa_circuit, simple_qubo)
        assert result.estimated_error_rate >= 0.0


# ---------------------------------------------------------------------------
# initial_layout_from_qubo: edge cases for lines 189-228
# ---------------------------------------------------------------------------


class TestInitialLayoutFallbacks:
    """Cover branches where one logical qubit is placed and the other
    is not, plus fallback paths when no adjacent physical qubit is free.
    """

    def test_li_placed_lj_not_adjacent_available(self) -> None:
        """Lines 197-206: li already placed, lj placed on neighbor."""
        # Q with strong (0,1) and (0,2) interactions
        # After placing (0,1) on first edge, qubit 2 needs to be
        # adjacent to qubit 0's physical position.
        Q = np.zeros((3, 3))
        Q[0, 1] = 10.0  # highest weight, placed first
        Q[1, 0] = 10.0
        Q[0, 2] = 5.0   # second highest, li=0 already placed
        Q[2, 0] = 5.0
        coupling = [(0, 1), (1, 2), (0, 2)]
        layout = initial_layout_from_qubo(Q, coupling)
        assert len(layout) == 3
        # All physical qubits should be unique
        assert len(set(layout.values())) == 3

    def test_lj_placed_li_not(self) -> None:
        """Lines 214-228: lj already placed, li needs placement."""
        # Create a QUBO where edge (1,2) is strongest, then (0,2)
        # After placing (1,2), for edge (0,2): lj=2 is placed, li=0 not
        Q = np.zeros((3, 3))
        Q[1, 2] = 10.0  # placed first
        Q[2, 1] = 10.0
        Q[0, 2] = 5.0   # lj=2 already placed, li=0 not
        Q[2, 0] = 5.0
        coupling = [(0, 1), (1, 2), (2, 3)]
        layout = initial_layout_from_qubo(Q, coupling)
        assert len(layout) == 3
        assert len(set(layout.values())) == 3

    def test_fallback_no_adjacent_free_li_placed(self) -> None:
        """Lines 207-212: li placed, no adjacent free -> fallback."""
        # Physical qubit 2 is isolated (not adjacent to 0 or 1).
        # After placing (0,1) on edge (0,1), qubit 2's only neighbor
        # for li=0 is physical 1, which is taken -> fallback to pq=2.
        Q = np.zeros((3, 3))
        Q[0, 1] = 10.0
        Q[1, 0] = 10.0
        Q[0, 2] = 5.0  # li=0 placed, neighbors of phys 0 all taken
        Q[2, 0] = 5.0
        # phys 2 exists but is not connected to anything
        coupling = [(0, 1), (2, 3)]
        layout = initial_layout_from_qubo(Q, coupling)
        assert len(layout) == 3
        assert len(set(layout.values())) == 3

    def test_fallback_no_adjacent_free_lj_placed(self) -> None:
        """Lines 223-228: lj placed, no adjacent free -> fallback."""
        Q = np.zeros((3, 3))
        Q[1, 2] = 10.0
        Q[2, 1] = 10.0
        Q[0, 2] = 5.0  # lj=2 placed, neighbors all taken
        Q[2, 0] = 5.0
        # phys 2 exists but isolated from 0,1
        coupling = [(0, 1), (2, 3)]
        layout = initial_layout_from_qubo(Q, coupling)
        assert len(layout) == 3
        assert len(set(layout.values())) == 3

    def test_both_unplaced_no_edge_available(self) -> None:
        """Lines 187-196: both unplaced, no free physical edge."""
        # All coupling edges used by prior placements, forcing
        # the fallback that assigns to any unused physical qubits.
        Q = np.zeros((4, 4))
        Q[0, 1] = 10.0  # highest, takes edge (0,1)
        Q[1, 0] = 10.0
        Q[2, 3] = 5.0   # both unplaced, edges (0,1) used
        Q[3, 2] = 5.0
        # Two edges but first is taken; second taken by (2,3)
        coupling = [(0, 1), (2, 3)]
        layout = initial_layout_from_qubo(Q, coupling)
        assert len(layout) == 4
        assert len(set(layout.values())) == 4

    def test_remaining_unplaced_qubits(self) -> None:
        """Lines 231-237: qubits with no interactions get assigned."""
        Q = np.zeros((4, 4))
        Q[0, 1] = 5.0
        Q[1, 0] = 5.0
        # qubits 2 and 3 have no interactions
        coupling = [(0, 1), (1, 2), (2, 3)]
        layout = initial_layout_from_qubo(Q, coupling)
        assert len(layout) == 4
        for i in range(4):
            assert i in layout
