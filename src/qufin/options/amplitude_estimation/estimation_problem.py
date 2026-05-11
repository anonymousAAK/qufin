"""Estimation problem abstraction for Quantum Amplitude Estimation.

Defines the oracle interface: a state preparation operator A and
an objective qubit index such that measuring the objective qubit
in |1> has probability a = sin^2(theta_a).

QAE estimates this probability a (or equivalently the angle theta_a).

References
----------
Brassard et al., "Quantum Amplitude Amplification and Estimation" (2002).
Stamatopoulos et al., Quantum 4:291 (2020), arXiv:1905.02666.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class EstimationProblem:
    """Amplitude estimation problem specification.

    Parameters
    ----------
    state_preparation : Any
        Quantum circuit A that prepares the state.
        A|0> = sqrt(1-a)|psi_0>|0> + sqrt(a)|psi_1>|1>
    objective_qubits : list[int]
        Qubit indices that define the "good" subspace.
    grover_operator : Any | None
        Custom Grover operator Q = A S_0 A^dag S_chi.
        If None, constructed automatically from state_preparation.
    n_qubits : int
        Total number of qubits in the circuit.
    is_good_state : callable | None
        Function that takes a bitstring and returns True if it is
        in the "good" subspace. Defaults to checking objective qubits.
    """

    state_preparation: Any
    objective_qubits: list[int]
    grover_operator: Any | None = None
    n_qubits: int = 0
    is_good_state: Any = None

    def __post_init__(self) -> None:
        if self.n_qubits == 0:
            # Try to infer from circuit
            with contextlib.suppress(AttributeError):
                self.n_qubits = self.state_preparation.num_qubits

    def build_grover_operator(self) -> Any:
        """Build the Grover operator Q = -A S_0 A^dag S_chi.

        Per Brassard et al. (2002) Eq. (4), the operator is:

            Q = -A (I - 2|0><0|) A^dag (I - 2 P_good)

        The minus sign ensures eigenvalues are e^{±2i*theta_a},
        which is critical for QPE-based amplitude estimation to
        recover a = sin^2(theta_a) via the formula a = sin^2(pi * phi).

        For circuit implementation, the global phase -1 is applied
        via `global_phase += pi`, which becomes a relative phase on
        the control qubit in controlled-Q operations (QPE).
        """
        if self.grover_operator is not None:
            return self.grover_operator

        from qiskit.circuit import QuantumCircuit

        n = self.n_qubits
        qc = QuantumCircuit(n)

        # S_chi: flip phase of good states (objective qubits = |1>)
        if len(self.objective_qubits) == 1:
            qc.z(self.objective_qubits[0])
        else:
            # Multi-controlled Z: flip phase when ALL objective qubits are |1>
            obj = self.objective_qubits
            qc.h(obj[-1])
            qc.mcx(obj[:-1], obj[-1])
            qc.h(obj[-1])

        # A^dag
        qc.compose(self.state_preparation.inverse(), inplace=True)

        # S_0: I - 2|0><0| (flip phase of |0> state)
        # Implementation: X on all, multi-controlled Z, X on all
        qc.x(range(n))
        if n == 1:
            qc.z(0)
        else:
            qc.h(n - 1)
            qc.mcx(list(range(n - 1)), n - 1)
            qc.h(n - 1)
        qc.x(range(n))

        # A
        qc.compose(self.state_preparation, inplace=True)

        # Global phase -1 (Brassard's minus sign)
        # This is essential: without it, eigenvalues are e^{±4i*theta_a}
        # instead of e^{±2i*theta_a}, causing QPE to return doubled phases.
        qc.global_phase += np.pi

        return qc
