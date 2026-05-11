"""PennyLane Lightning backend adapter.

Provides a qufin Backend interface wrapping PennyLane's
default.qubit or lightning.qubit simulators.

Requires: pip install qufin[pennylane]
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from qufin.backends.base import Backend, CircuitResult


class PennyLaneBackend(Backend):
    """PennyLane backend adapter.

    Converts Qiskit circuits to PennyLane via qiskit-to-pennylane
    conversion, or runs native PennyLane circuits.

    Parameters
    ----------
    device_name : str
        PennyLane device name (e.g., "default.qubit", "lightning.qubit").
    n_qubits : int
        Number of qubits (required for device initialization).
    """

    def __init__(
        self,
        device_name: str = "default.qubit",
        n_qubits: int = 10,
    ) -> None:
        try:
            import pennylane as qml
        except ImportError as e:
            raise ImportError(
                "PennyLane is required. Install with: pip install qufin[pennylane]"
            ) from e

        self._device_name = device_name
        self._n_qubits = n_qubits
        self._qml = qml
        self._device = qml.device(device_name, wires=n_qubits)

    @property
    def backend_id(self) -> str:
        return f"pennylane:{self._device_name}"

    def run(self, circuit: Any, shots: int = 1024) -> CircuitResult:
        """Execute a circuit and return measurement results.

        Accepts either a PennyLane QNode or a Qiskit QuantumCircuit.
        For Qiskit circuits, converts via from_qiskit().
        """

        if hasattr(circuit, "num_qubits"):
            # Qiskit circuit → convert
            return self._run_qiskit_circuit(circuit, shots)

        # Native PennyLane QNode
        if callable(circuit):
            results = circuit()
            if isinstance(results, np.ndarray):
                probs = results
                counts = {}
                for i, p in enumerate(probs):
                    if p > 1e-10:
                        bs = format(i, f"0{self._n_qubits}b")
                        counts[bs] = round(p * shots)
                return CircuitResult(
                    counts=counts, shots=shots, backend_id=self.backend_id
                )

        return CircuitResult(counts={}, shots=shots, backend_id=self.backend_id)

    def _run_qiskit_circuit(self, circuit: Any, shots: int) -> CircuitResult:
        """Run a Qiskit circuit by converting to PennyLane."""
        qml = self._qml

        n_qubits = circuit.num_qubits
        dev = qml.device(self._device_name, wires=n_qubits, shots=shots)

        @qml.qnode(dev)
        def converted_circuit():
            qml.from_qiskit(circuit)()
            return qml.counts()

        try:
            result = converted_circuit()
            counts = {k: int(v) for k, v in result.items()}
        except Exception:
            # Fallback: use statevector sampling
            counts = self._sample_from_statevector(circuit, shots, n_qubits)

        return CircuitResult(
            counts=counts, shots=shots, backend_id=self.backend_id
        )

    def _sample_from_statevector(
        self, circuit: Any, shots: int, n_qubits: int
    ) -> dict[str, int]:
        """Fallback: compute statevector and sample."""
        qml = self._qml
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev)
        def sv_circuit():
            qml.from_qiskit(circuit)()
            return qml.probs(wires=range(n_qubits))

        try:
            probs = sv_circuit()
            rng = np.random.default_rng()
            samples = rng.choice(len(probs), size=shots, p=np.array(probs))
            counts: dict[str, int] = {}
            for s in samples:
                bs = format(s, f"0{n_qubits}b")
                counts[bs] = counts.get(bs, 0) + 1
            return counts
        except Exception:
            return {}

    def statevector(self, circuit: Any) -> NDArray[np.complex128]:
        """Return the statevector for a circuit."""
        qml = self._qml

        if hasattr(circuit, "num_qubits"):
            n_qubits = circuit.num_qubits
            dev = qml.device("default.qubit", wires=n_qubits)

            @qml.qnode(dev)
            def sv_circuit():
                qml.from_qiskit(circuit)()
                return qml.state()

            return np.array(sv_circuit(), dtype=np.complex128)

        raise ValueError("Cannot compute statevector for non-Qiskit circuit")

    def is_simulator(self) -> bool:
        return True
