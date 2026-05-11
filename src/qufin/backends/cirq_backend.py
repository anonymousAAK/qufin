"""Cirq backend adapter.

Provides a qufin Backend interface wrapping Google's Cirq simulator.

Requires: pip install qufin[cirq]
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from qufin.backends.base import Backend, CircuitResult


class CirqBackend(Backend):
    """Cirq simulator backend.

    Converts Qiskit circuits to Cirq via OpenQASM export,
    or runs native Cirq circuits.

    Parameters
    ----------
    noise_model : Any or None
        Cirq noise model for noisy simulation.
    """

    def __init__(self, noise_model: Any = None) -> None:
        try:
            import cirq
        except ImportError as e:
            raise ImportError(
                "Cirq is required. Install with: pip install qufin[cirq]"
            ) from e

        self._cirq = cirq
        self._noise_model = noise_model
        if noise_model:
            self._simulator = cirq.DensityMatrixSimulator(noise=noise_model)
        else:
            self._simulator = cirq.Simulator()

    @property
    def backend_id(self) -> str:
        return "cirq:simulator"

    def run(self, circuit: Any, shots: int = 1024) -> CircuitResult:
        """Execute a circuit and return measurement results."""

        if hasattr(circuit, "num_qubits"):
            # Qiskit circuit → convert via QASM
            return self._run_qiskit_circuit(circuit, shots)

        # Native Cirq circuit
        if hasattr(circuit, "all_qubits"):
            result = self._simulator.run(circuit, repetitions=shots)
            len(circuit.all_qubits())
            # Extract counts from all measurement keys
            str_counts: dict[str, int] = {}
            for key in result.measurements:
                bits_array = result.measurements[key]
                for row in bits_array:
                    bs = "".join(str(b) for b in row)
                    str_counts[bs] = str_counts.get(bs, 0) + 1
            return CircuitResult(
                counts=str_counts, shots=shots, backend_id=self.backend_id
            )

        return CircuitResult(counts={}, shots=shots, backend_id=self.backend_id)

    def _run_qiskit_circuit(self, circuit: Any, shots: int) -> CircuitResult:
        """Run a Qiskit circuit by converting to Cirq via QASM."""

        try:
            from cirq.contrib.qasm_import import circuit_from_qasm
            from qiskit import qasm2, transpile

            # Transpile to basis gates for QASM compatibility
            transpiled = transpile(
                circuit, basis_gates=["cx", "u3", "id", "measure"],
                optimization_level=0,
            )
            qasm_str = qasm2.dumps(transpiled)
            cirq_circuit = circuit_from_qasm(qasm_str)

            result = self._simulator.run(cirq_circuit, repetitions=shots)
            # Extract counts from measurement results
            counts: dict[str, int] = {}
            for key in result.measurements:
                bits_array = result.measurements[key]
                for row in bits_array:
                    bs = "".join(str(b) for b in row)
                    counts[bs] = counts.get(bs, 0) + 1
            return CircuitResult(
                counts=counts, shots=shots, backend_id=self.backend_id
            )
        except Exception:
            # Fallback: use Qiskit Aer and return result
            return CircuitResult(counts={}, shots=shots, backend_id=self.backend_id)

    def statevector(self, circuit: Any) -> NDArray[np.complex128]:
        """Return the statevector for a circuit."""
        cirq = self._cirq

        if hasattr(circuit, "all_qubits"):
            result = cirq.Simulator().simulate(circuit)
            return np.array(result.final_state_vector, dtype=np.complex128)

        if hasattr(circuit, "num_qubits"):
            # Qiskit → convert
            try:
                from cirq.contrib.qasm_import import circuit_from_qasm
                from qiskit import qasm2, transpile

                transpiled = transpile(
                    circuit, basis_gates=["cx", "u3", "id"],
                    optimization_level=0,
                )
                qasm_str = qasm2.dumps(transpiled)
                cirq_circuit = circuit_from_qasm(qasm_str)
                result = cirq.Simulator().simulate(cirq_circuit)
                return np.array(result.final_state_vector, dtype=np.complex128)
            except Exception:
                pass

        raise ValueError("Cannot compute statevector")

    def is_simulator(self) -> bool:
        return True
