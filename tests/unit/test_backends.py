"""Tests for backend abstraction layer."""

from __future__ import annotations

import numpy as np
import pytest

from qufin.backends.mock import MockBackend


class TestMockBackend:
    def test_default_counts(self) -> None:
        backend = MockBackend()
        result = backend.run(None, shots=1024)
        assert sum(result.counts.values()) == 1024

    def test_custom_counts(self) -> None:
        backend = MockBackend(default_counts={"00": 3, "11": 1})
        result = backend.run(None, shots=100)
        assert sum(result.counts.values()) == 100
        assert result.counts["00"] > result.counts["11"]

    def test_most_frequent(self) -> None:
        backend = MockBackend(default_counts={"00": 3, "01": 1})
        result = backend.run(None, shots=1000)
        assert result.most_frequent == "00"

    def test_probabilities(self) -> None:
        backend = MockBackend(default_counts={"0": 1, "1": 1})
        result = backend.run(None, shots=1000)
        probs = result.probabilities
        assert abs(probs["0"] - 0.5) < 0.01

    def test_statevector(self) -> None:
        backend = MockBackend()

        class FakeCircuit:
            num_qubits = 3

        sv = backend.statevector(FakeCircuit())
        assert sv.shape == (8,)
        np.testing.assert_allclose(np.abs(sv) ** 2, 1 / 8, atol=1e-10)

    def test_is_simulator(self) -> None:
        backend = MockBackend()
        assert backend.is_simulator()

    def test_backend_id(self) -> None:
        backend = MockBackend(seed=123)
        assert backend.backend_id == "mock-seed123"
