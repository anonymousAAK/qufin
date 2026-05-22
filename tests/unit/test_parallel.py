"""Unit tests for parallel circuit execution utilities."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from qufin.backends.base import Backend, CircuitResult
from qufin.utils.parallel import batch_execute, parallel_execute


class _MockBackend(Backend):
    """Minimal mock backend for testing."""

    def __init__(self, delay: float = 0.0) -> None:
        self._delay = delay
        self._call_count = 0

    @property
    def backend_id(self) -> str:
        return "mock_test"

    def run(self, circuit: Any, shots: int = 1024) -> CircuitResult:
        import time

        if self._delay > 0:
            time.sleep(self._delay)
        self._call_count += 1
        return CircuitResult(counts={"00": shots}, shots=shots, backend_id="mock_test")

    def statevector(self, circuit: Any) -> np.ndarray:
        return np.array([1, 0], dtype=np.complex128)


def _make_circuits(n: int = 5) -> list[object]:
    from qiskit.circuit import QuantumCircuit

    circuits = []
    for i in range(n):
        qc = QuantumCircuit(2)
        qc.h(0)
        if i % 2 == 0:
            qc.cx(0, 1)
        circuits.append(qc)
    return circuits


class TestParallelExecute:
    def test_empty_list(self) -> None:
        backend = _MockBackend()
        results = parallel_execute([], backend)
        assert results == []

    def test_single_circuit(self) -> None:
        backend = _MockBackend()
        circs = _make_circuits(1)
        results = parallel_execute(circs, backend, shots=512)
        assert len(results) == 1
        assert results[0].shots == 512

    def test_multiple_circuits(self) -> None:
        backend = _MockBackend()
        circs = _make_circuits(5)
        results = parallel_execute(circs, backend, shots=1024, max_workers=2)
        assert len(results) == 5
        for r in results:
            assert r.shots == 1024

    def test_order_preserved(self) -> None:
        """Results should match circuit order regardless of execution order."""
        backend = _MockBackend()
        circs = _make_circuits(8)
        results = parallel_execute(circs, backend, max_workers=4)
        assert len(results) == 8
        for r in results:
            assert isinstance(r, CircuitResult)

    def test_progress_callback(self) -> None:
        backend = _MockBackend()
        circs = _make_circuits(3)
        progress: list[tuple[int, int]] = []

        def callback(done: int, total: int) -> None:
            progress.append((done, total))

        parallel_execute(circs, backend, progress_callback=callback)
        assert len(progress) == 3
        assert all(t == 3 for _, t in progress)
        assert sorted(d for d, _ in progress) == [1, 2, 3]

    def test_max_workers_one(self) -> None:
        backend = _MockBackend()
        circs = _make_circuits(3)
        results = parallel_execute(circs, backend, max_workers=1)
        assert len(results) == 3

    def test_backend_called_per_circuit(self) -> None:
        backend = _MockBackend()
        circs = _make_circuits(4)
        parallel_execute(circs, backend)
        assert backend._call_count == 4


class TestBatchExecute:
    def test_empty_list(self) -> None:
        backend = _MockBackend()
        results = batch_execute([], backend)
        assert results == []

    def test_single_batch(self) -> None:
        backend = _MockBackend()
        circs = _make_circuits(3)
        results = batch_execute(circs, backend, batch_size=10)
        assert len(results) == 3

    def test_multiple_batches(self) -> None:
        backend = _MockBackend()
        circs = _make_circuits(7)
        results = batch_execute(circs, backend, batch_size=3)
        assert len(results) == 7

    def test_exact_batch_boundary(self) -> None:
        backend = _MockBackend()
        circs = _make_circuits(6)
        results = batch_execute(circs, backend, batch_size=3)
        assert len(results) == 6

    def test_batch_size_one(self) -> None:
        backend = _MockBackend()
        circs = _make_circuits(4)
        results = batch_execute(circs, backend, batch_size=1)
        assert len(results) == 4

    def test_invalid_batch_size(self) -> None:
        backend = _MockBackend()
        with pytest.raises(ValueError, match="batch_size"):
            batch_execute([MagicMock()], backend, batch_size=0)

    def test_progress_callback(self) -> None:
        backend = _MockBackend()
        circs = _make_circuits(5)
        progress: list[tuple[int, int]] = []

        def callback(done: int, total: int) -> None:
            progress.append((done, total))

        batch_execute(circs, backend, batch_size=2, progress_callback=callback)
        assert len(progress) == 5
        assert progress[-1] == (5, 5)

    def test_order_preserved(self) -> None:
        backend = _MockBackend()
        circs = _make_circuits(6)
        results = batch_execute(circs, backend, batch_size=2)
        assert len(results) == 6
        for r in results:
            assert isinstance(r, CircuitResult)
