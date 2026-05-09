"""Qiskit Aer backend adapter."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from qufin.backends.base import Backend, CircuitResult


class QiskitAerBackend(Backend):
    """Backend using Qiskit Aer for local simulation.

    Parameters
    ----------
    method : str
        Simulation method: "automatic", "statevector", "matrix_product_state".
    seed : int | None
        Random seed for reproducibility.
    """

    def __init__(self, method: str = "automatic", seed: int | None = 42) -> None:
        from qiskit_aer import AerSimulator

        self._sim = AerSimulator(method=method, seed_simulator=seed)
        self._seed = seed
        self._method = method

    @property
    def backend_id(self) -> str:
        return f"qiskit-aer-{self._method}"

    def run(self, circuit: Any, shots: int = 1024) -> CircuitResult:
        from qiskit import transpile

        transpiled = transpile(circuit, self._sim)
        job = self._sim.run(transpiled, shots=shots)
        result = job.result()
        counts = result.get_counts()
        # Qiskit returns counts with spaces for multi-register; flatten
        flat_counts = {k.replace(" ", ""): v for k, v in counts.items()}
        return CircuitResult(
            counts=flat_counts,
            shots=shots,
            backend_id=self.backend_id,
        )

    def statevector(self, circuit: Any) -> NDArray[np.complex128]:
        from qiskit.quantum_info import Statevector

        sv = Statevector.from_instruction(circuit)
        return np.array(sv.data, dtype=np.complex128)
