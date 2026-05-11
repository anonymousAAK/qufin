"""AWS Braket backend adapter (IonQ, Rigetti, IQM, QuEra).

Provides a qufin Backend interface wrapping Amazon Braket devices
and simulators.

Requires: pip install qufin[braket]
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from qufin.backends.base import Backend, CircuitResult


class BraketBackend(Backend):
    """Amazon Braket backend.

    Supports both local simulator and managed QPU devices.

    Parameters
    ----------
    device_arn : str or None
        AWS device ARN. None uses local simulator.
    s3_bucket : str or None
        S3 bucket for results (required for QPU).
    s3_prefix : str
        S3 key prefix for results.
    """

    def __init__(
        self,
        device_arn: str | None = None,
        s3_bucket: str | None = None,
        s3_prefix: str = "qufin-results",
    ) -> None:
        try:
            from braket.devices import LocalSimulator
        except ImportError as e:
            raise ImportError(
                "Amazon Braket SDK is required. Install with: pip install qufin[braket]"
            ) from e

        self._device_arn = device_arn
        self._s3_bucket = s3_bucket
        self._s3_prefix = s3_prefix

        if device_arn is None:
            self._device = LocalSimulator()
            self._is_local = True
        else:
            from braket.aws import AwsDevice
            self._device = AwsDevice(device_arn)
            self._is_local = False

    @property
    def backend_id(self) -> str:
        if self._device_arn:
            return f"braket:{self._device_arn.split('/')[-1]}"
        return "braket:local"

    def run(self, circuit: Any, shots: int = 1024) -> CircuitResult:
        """Execute a circuit and return measurement results."""
        if hasattr(circuit, "num_qubits"):
            return self._run_qiskit_circuit(circuit, shots)

        # Native Braket circuit
        from braket.circuits import Circuit as BraketCircuit
        if isinstance(circuit, BraketCircuit):
            return self._run_braket_circuit(circuit, shots)

        return CircuitResult(counts={}, shots=shots, backend_id=self.backend_id)

    def _run_braket_circuit(self, circuit: Any, shots: int) -> CircuitResult:
        """Run a native Braket circuit."""
        if self._is_local:
            task = self._device.run(circuit, shots=shots)
        else:
            s3_loc = (self._s3_bucket, self._s3_prefix)
            task = self._device.run(circuit, s3_loc, shots=shots)

        result = task.result()
        counts = {k: int(v) for k, v in result.measurement_counts.items()}
        return CircuitResult(
            counts=counts, shots=shots, backend_id=self.backend_id
        )

    def _run_qiskit_circuit(self, circuit: Any, shots: int) -> CircuitResult:
        """Run a Qiskit circuit by converting to Braket via OpenQASM."""
        try:
            from braket.circuits import Circuit as BraketCircuit
            from qiskit import qasm2, transpile

            transpiled = transpile(
                circuit, basis_gates=["cx", "u3", "id", "measure"],
                optimization_level=0,
            )
            qasm_str = qasm2.dumps(transpiled)
            braket_circuit = BraketCircuit.from_ir(qasm_str, ir_type="OPENQASM")
            return self._run_braket_circuit(braket_circuit, shots)
        except Exception:
            return CircuitResult(counts={}, shots=shots, backend_id=self.backend_id)

    def statevector(self, circuit: Any) -> NDArray[np.complex128]:
        """Return the statevector (local simulator only)."""
        if not self._is_local:
            raise RuntimeError("Statevector only available on local simulator")

        from braket.devices import LocalSimulator
        sv_device = LocalSimulator("default")

        if hasattr(circuit, "num_qubits"):
            try:
                from braket.circuits import Circuit as BraketCircuit
                from qiskit import qasm2, transpile

                transpiled = transpile(
                    circuit, basis_gates=["cx", "u3", "id"],
                    optimization_level=0,
                )
                qasm_str = qasm2.dumps(transpiled)
                braket_circuit = BraketCircuit.from_ir(qasm_str, ir_type="OPENQASM")
                braket_circuit.state_vector()
                task = sv_device.run(braket_circuit, shots=0)
                result = task.result()
                return np.array(result.result_types[0].value, dtype=np.complex128)
            except Exception:
                pass

        raise ValueError("Cannot compute statevector")

    def is_simulator(self) -> bool:
        return self._is_local
