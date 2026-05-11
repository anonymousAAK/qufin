"""QAOA mixer Hamiltonians: X, XY-ring, Grover, Dicke initial state.

References
----------
Hadfield et al., Algorithms 12:34 (2019) — QAOA+.
Wang, Rubin, Dominy, Rieffel, PRA 101:012320 (2020) — XY mixers.
Bartschi et al., npj QI (2024) — Dicke state alignment.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Mixer(ABC):
    """Abstract mixer for QAOA."""

    @abstractmethod
    def circuit(self, beta: float) -> Any:
        """Return a mixer circuit parameterized by beta."""

    @property
    @abstractmethod
    def preserves_hamming_weight(self) -> bool:
        """Whether this mixer preserves Hamming weight (for cardinality)."""


class XMixer(Mixer):
    """Standard X-mixer (sum of Pauli-X on each qubit).

    Does NOT preserve Hamming weight — use for unconstrained problems.
    """

    def __init__(self, n_qubits: int) -> None:
        self.n_qubits = n_qubits

    @property
    def preserves_hamming_weight(self) -> bool:
        return False

    def circuit(self, beta: float) -> Any:
        from qiskit.circuit import QuantumCircuit

        qc = QuantumCircuit(self.n_qubits)
        for i in range(self.n_qubits):
            qc.rx(2 * beta, i)
        return qc


class XYRingMixer(Mixer):
    """XY-ring mixer preserving Hamming weight (for cardinality constraints).

    Implements nearest-neighbor XX+YY interactions in a ring topology.
    This keeps the total number of 1s constant, naturally enforcing
    cardinality K when initialized from a Dicke state |D_n^k>.
    """

    def __init__(self, n_qubits: int) -> None:
        self.n_qubits = n_qubits

    @property
    def preserves_hamming_weight(self) -> bool:
        return True

    def circuit(self, beta: float) -> Any:
        from qiskit.circuit import QuantumCircuit

        qc = QuantumCircuit(self.n_qubits)
        for i in range(self.n_qubits):
            j = (i + 1) % self.n_qubits
            # XX + YY interaction via RXX + RYY decomposition
            qc.rxx(2 * beta, i, j)
            qc.ryy(2 * beta, i, j)
        return qc


class XYFullMixer(Mixer):
    """Fully-connected XY mixer (all-to-all XX+YY).

    More expressive than ring but deeper circuits.
    Preserves Hamming weight.
    """

    def __init__(self, n_qubits: int) -> None:
        self.n_qubits = n_qubits

    @property
    def preserves_hamming_weight(self) -> bool:
        return True

    def circuit(self, beta: float) -> Any:
        from qiskit.circuit import QuantumCircuit

        qc = QuantumCircuit(self.n_qubits)
        for i in range(self.n_qubits):
            for j in range(i + 1, self.n_qubits):
                qc.rxx(2 * beta, i, j)
                qc.ryy(2 * beta, i, j)
        return qc


class GroverMixer(Mixer):
    """Grover-style mixer: reflects about the uniform superposition.

    Used in the original Grover-QAOA variant. Does not preserve Hamming weight.
    """

    def __init__(self, n_qubits: int) -> None:
        self.n_qubits = n_qubits

    @property
    def preserves_hamming_weight(self) -> bool:
        return False

    def circuit(self, beta: float) -> Any:
        """exp(-i * beta * D) where D = 2|s><s| - I and |s> = |+>^n.

        Implemented as: R_s(beta) = I - (1 - e^{-2i*beta}) |s><s|
        which decomposes to H^n . R_0(beta) . H^n
        where R_0(beta) = I - (1 - e^{-2i*beta}) |0><0|.
        """
        from qiskit.circuit import QuantumCircuit

        qc = QuantumCircuit(self.n_qubits)
        qc.h(range(self.n_qubits))
        qc.x(range(self.n_qubits))
        # Phase gate on |11...1> by angle 2*beta
        if self.n_qubits > 1:
            # Multi-controlled phase: apply P(2*beta) on last qubit
            # controlled by all others = phase on |11...1> state
            from qiskit.circuit.library import PhaseGate
            qc.append(
                PhaseGate(2 * beta).control(self.n_qubits - 1),
                list(range(self.n_qubits)),
            )
        else:
            qc.p(2 * beta, 0)
        qc.x(range(self.n_qubits))
        qc.h(range(self.n_qubits))
        return qc


class DickeInitialState:
    """Prepare the Dicke state |D_n^k> as initial state for cardinality-K QAOA.

    The Dicke state is the uniform superposition over all n-qubit states
    with exactly k ones. Used with XY-ring or XY-full mixers to maintain
    cardinality throughout the QAOA evolution.

    References
    ----------
    Bartschi & Eidenbenz, arXiv:1904.07358 — deterministic Dicke state preparation.
    """

    def __init__(self, n_qubits: int, k: int) -> None:
        if k > n_qubits:
            raise ValueError(f"k={k} > n_qubits={n_qubits}")
        self.n_qubits = n_qubits
        self.k = k

    def circuit(self) -> Any:
        """Prepare |D_n^k> using the Bärtschi & Eidenbenz SCS algorithm.

        The algorithm works by iterating through positions n-1 down to 1.
        At each position j, a split operation distributes Hamming weight
        from position j to position j using a controlled rotation.
        """
        from qiskit.circuit import QuantumCircuit

        n = self.n_qubits
        k = self.k
        qc = QuantumCircuit(n)

        # Initialize last k qubits to |1> (place all excitations at the end)
        for i in range(n - k, n):
            qc.x(i)

        # SCS: iterate from position n-1 down to 1
        # At step j, we split excitations between qubit j and qubits 0..j-1
        for j in range(n - 1, 0, -1):
            # Number of excitations that should be in qubits 0..j
            # is min(k, j+1). Number in qubits 0..j-1 is min(k, j).
            # The "remaining" at position j needs partial transfer.
            upper = min(k, j + 1)
            lower = min(k, j)
            if upper == lower:
                continue
            # Split: controlled-Ry on qubit j-1, controlled by qubit j
            for l in range(lower, upper):
                # Probability that qubit j-1 gets a |1>
                # given that qubit j has a |1> and we need to
                # distribute weight across positions 0..j
                denom = j - l + 1
                if denom <= 1:
                    continue
                theta = 2 * np.arcsin(np.sqrt(1.0 / denom))
                # Controlled rotation: if qubit j is |1>, rotate qubit j-1
                qc.cry(theta, j, j - 1)
                # Swap the excitation: CNOT to move |1> from j to j-1
                qc.cx(j - 1, j)
        return qc


def get_mixer(
    name: str, n_qubits: int, **kwargs: Any
) -> Mixer:
    """Factory function to create a mixer by name.

    Parameters
    ----------
    name : str
        One of "x", "xy_ring", "xy_full", "grover".
    n_qubits : int
        Number of qubits.
    """
    mixers = {
        "x": XMixer,
        "xy_ring": XYRingMixer,
        "xy_full": XYFullMixer,
        "grover": GroverMixer,
    }
    if name not in mixers:
        raise ValueError(f"Unknown mixer '{name}'. Available: {list(mixers.keys())}")
    return mixers[name](n_qubits)
