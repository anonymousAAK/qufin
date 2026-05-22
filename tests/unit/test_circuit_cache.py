"""Unit tests for circuit compilation cache."""

from __future__ import annotations

import threading

import pytest

from qufin.utils.circuit_cache import CachedTranspiler, CacheStats, _circuit_hash


def _make_circuit(n_qubits: int = 2, label: str = "") -> object:
    """Create a simple test circuit."""
    from qiskit.circuit import QuantumCircuit

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(0)
    if n_qubits > 1:
        qc.cx(0, 1)
    if label:
        qc.name = label
    qc.measure_all(add_bits=False) if qc.num_clbits >= n_qubits else None
    return qc


class TestCacheStats:
    def test_initial_values(self) -> None:
        stats = CacheStats()
        assert stats.hits == 0
        assert stats.misses == 0

    def test_total(self) -> None:
        stats = CacheStats(hits=3, misses=7)
        assert stats.total == 10

    def test_hit_rate(self) -> None:
        stats = CacheStats(hits=3, misses=7)
        assert abs(stats.hit_rate - 0.3) < 1e-10

    def test_hit_rate_zero_total(self) -> None:
        stats = CacheStats()
        assert stats.hit_rate == 0.0


class TestCircuitHash:
    def test_deterministic(self) -> None:
        c1 = _make_circuit(2)
        h1 = _circuit_hash(c1)
        h2 = _circuit_hash(c1)
        assert h1 == h2

    def test_different_circuits_differ(self) -> None:
        c1 = _make_circuit(2)
        c2 = _make_circuit(3)
        assert _circuit_hash(c1) != _circuit_hash(c2)

    def test_returns_string(self) -> None:
        c = _make_circuit(2)
        assert isinstance(_circuit_hash(c), str)
        assert len(_circuit_hash(c)) == 64  # SHA-256 hex digest


class TestCachedTranspiler:
    def test_basic_transpile(self) -> None:
        ct = CachedTranspiler(max_size=4)
        circ = _make_circuit(2)
        result = ct.transpile(circ, backend_id="test", optimization_level=0)
        assert result is not None
        assert ct.stats.misses == 1
        assert ct.stats.hits == 0

    def test_cache_hit(self) -> None:
        ct = CachedTranspiler(max_size=4)
        circ = _make_circuit(2)
        r1 = ct.transpile(circ, backend_id="test", optimization_level=0)
        r2 = ct.transpile(circ, backend_id="test", optimization_level=0)
        assert ct.stats.hits == 1
        assert ct.stats.misses == 1
        assert r1 is r2  # Same cached object

    def test_different_backend_id_misses(self) -> None:
        ct = CachedTranspiler(max_size=4)
        circ = _make_circuit(2)
        ct.transpile(circ, backend_id="aer", optimization_level=0)
        ct.transpile(circ, backend_id="ibm", optimization_level=0)
        assert ct.stats.misses == 2

    def test_different_opt_level_misses(self) -> None:
        ct = CachedTranspiler(max_size=4)
        circ = _make_circuit(2)
        ct.transpile(circ, backend_id="test", optimization_level=0)
        ct.transpile(circ, backend_id="test", optimization_level=2)
        assert ct.stats.misses == 2

    def test_eviction(self) -> None:
        ct = CachedTranspiler(max_size=2)
        c1 = _make_circuit(2)
        c2 = _make_circuit(3)
        from qiskit.circuit import QuantumCircuit

        c3 = QuantumCircuit(1)
        c3.x(0)

        ct.transpile(c1, backend_id="t", optimization_level=0)
        ct.transpile(c2, backend_id="t", optimization_level=0)
        assert ct.size == 2
        ct.transpile(c3, backend_id="t", optimization_level=0)
        assert ct.size == 2  # oldest evicted

    def test_size_property(self) -> None:
        ct = CachedTranspiler(max_size=10)
        assert ct.size == 0
        ct.transpile(_make_circuit(2), backend_id="t", optimization_level=0)
        assert ct.size == 1

    def test_max_size_property(self) -> None:
        ct = CachedTranspiler(max_size=64)
        assert ct.max_size == 64

    def test_invalid_max_size(self) -> None:
        with pytest.raises(ValueError, match="max_size"):
            CachedTranspiler(max_size=0)

    def test_clear(self) -> None:
        ct = CachedTranspiler(max_size=4)
        ct.transpile(_make_circuit(2), backend_id="t", optimization_level=0)
        assert ct.size == 1
        ct.clear()
        assert ct.size == 0
        assert ct.stats.hits == 0
        assert ct.stats.misses == 0

    def test_thread_safety(self) -> None:
        ct = CachedTranspiler(max_size=50)
        errors: list[Exception] = []

        def worker(i: int) -> None:
            try:
                circ = _make_circuit(2)
                ct.transpile(circ, backend_id=f"b{i % 3}", optimization_level=0)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert ct.stats.total == 10

    def test_lru_ordering(self) -> None:
        """Most recently used item survives eviction."""
        ct = CachedTranspiler(max_size=2)
        c1 = _make_circuit(2)
        from qiskit.circuit import QuantumCircuit

        c2 = QuantumCircuit(1)
        c2.x(0)
        c3 = QuantumCircuit(1)
        c3.h(0)

        ct.transpile(c1, backend_id="t", optimization_level=0)
        ct.transpile(c2, backend_id="t", optimization_level=0)
        # Access c1 again to make it most recent
        ct.transpile(c1, backend_id="t", optimization_level=0)
        # Insert c3 — should evict c2 (oldest), not c1
        ct.transpile(c3, backend_id="t", optimization_level=0)
        # c1 should still be cached
        ct.transpile(c1, backend_id="t", optimization_level=0)
        assert ct.stats.hits == 2  # c1 hit twice (access + after eviction)
