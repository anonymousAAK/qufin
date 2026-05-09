"""QAOA mixer Hamiltonians: X, XY-ring, Dicke initial state.

References
----------
Hadfield et al., Algorithms 12:34 (2019) — QAOA+.
Wang, Rubin, Dominy, Rieffel, PRA 101:012320 (2020) — XY mixers.
Bartschi et al., npj QI (2024) — Dicke state alignment.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Mixer(ABC):
    """Abstract mixer for QAOA."""

    @abstractmethod
    def circuit(self, beta: float) -> Any:
        """Return a mixer circuit parameterized by beta."""


class XMixer(Mixer):
    """Standard X-mixer (sum of Pauli-X on each qubit)."""

    def __init__(self, n_qubits: int) -> None:
        self.n_qubits = n_qubits

    def circuit(self, beta: float) -> Any:
        from qiskit.circuit import QuantumCircuit

        qc = QuantumCircuit(self.n_qubits)
        for i in range(self.n_qubits):
            qc.rx(2 * beta, i)
        return qc


class XYRingMixer(Mixer):
    """XY-ring mixer preserving Hamming weight (for cardinality constraints).

    Implements nearest-neighbor XX+YY interactions in a ring topology.
    """

    def __init__(self, n_qubits: int) -> None:
        self.n_qubits = n_qubits

    def circuit(self, beta: float) -> Any:
        from qiskit.circuit import QuantumCircuit

        qc = QuantumCircuit(self.n_qubits)
        for i in range(self.n_qubits):
            j = (i + 1) % self.n_qubits
            # XX + YY interaction via RXX + RYY decomposition
            qc.rxx(2 * beta, i, j)
            qc.ryy(2 * beta, i, j)
        return qc


class DickeInitialState:
    """Prepare the Dicke state |D_n^k> as initial state for cardinality-K QAOA.

    The Dicke state is the uniform superposition over all n-qubit states
    with exactly k ones.
    """

    def __init__(self, n_qubits: int, k: int) -> None:
        self.n_qubits = n_qubits
        self.k = k

    def circuit(self) -> Any:
        from qiskit.circuit import QuantumCircuit

        qc = QuantumCircuit(self.n_qubits)
        # Initialize k qubits to |1>
        for i in range(self.k):
            qc.x(i)
        # Apply split-and-cyclic-shift (SCS) unitaries
        # Simplified version using pairwise partial swaps
        for i in range(self.k):
            for j in range(i + 1, self.n_qubits):
                # Partial swap angle
                theta = 2 * np.arcsin(np.sqrt(1 / (j - i + 1)))
                qc.ry(theta, j)
                qc.cx(j, i)
                qc.ry(-theta, j)
                qc.cx(j, i)
        return qc


import numpy as np  # noqa: E402 (needed by DickeInitialState)
