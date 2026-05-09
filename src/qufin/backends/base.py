"""Abstract base class for quantum backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class CircuitResult:
    """Result from executing a quantum circuit."""

    counts: dict[str, int] = field(default_factory=dict)
    statevector: NDArray[np.complex128] | None = None
    shots: int = 0
    backend_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def probabilities(self) -> dict[str, float]:
        """Convert counts to probability distribution."""
        if self.shots == 0:
            return {}
        return {k: v / self.shots for k, v in self.counts.items()}

    @property
    def most_frequent(self) -> str:
        """Return the most frequently measured bitstring."""
        if not self.counts:
            return ""
        return max(self.counts, key=self.counts.__getitem__)


class Backend(ABC):
    """Abstract backend for circuit execution.

    All quantum backends (Qiskit Aer, IBM Runtime, PennyLane, Cirq, Braket)
    implement this interface so algorithms are backend-agnostic.
    """

    @property
    @abstractmethod
    def backend_id(self) -> str:
        """Unique identifier for this backend."""

    @abstractmethod
    def run(self, circuit: Any, shots: int = 1024) -> CircuitResult:
        """Execute a circuit and return measurement results."""

    @abstractmethod
    def statevector(self, circuit: Any) -> NDArray[np.complex128]:
        """Return the statevector for a circuit (simulator only)."""

    def is_simulator(self) -> bool:
        """Whether this backend is a simulator (default True)."""
        return True
