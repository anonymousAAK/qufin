"""Deterministic mock backend for testing."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from qufin.backends.base import Backend, CircuitResult


class MockBackend(Backend):
    """Deterministic backend that returns configurable results.

    Useful for unit testing algorithms without quantum simulation overhead.
    """

    def __init__(self, default_counts: dict[str, int] | None = None, seed: int = 42) -> None:
        self._default_counts = default_counts or {"0": 512, "1": 512}
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    @property
    def backend_id(self) -> str:
        return f"mock-seed{self._seed}"

    def run(self, circuit: Any, shots: int = 1024) -> CircuitResult:
        """Return default counts scaled to requested shots."""
        total = sum(self._default_counts.values())
        counts = {k: int(v / total * shots) for k, v in self._default_counts.items()}
        # Distribute rounding remainder to first key
        remainder = shots - sum(counts.values())
        if remainder and counts:
            first_key = next(iter(counts))
            counts[first_key] += remainder
        return CircuitResult(
            counts=counts,
            shots=shots,
            backend_id=self.backend_id,
        )

    def statevector(self, circuit: Any) -> NDArray[np.complex128]:
        """Return a uniform statevector for the mock backend."""
        n_qubits = getattr(circuit, "num_qubits", 2)
        dim = 2**n_qubits
        sv = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
        return sv
