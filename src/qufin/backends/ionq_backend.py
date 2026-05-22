"""IonQ backend adapter via Amazon Braket or direct API.

Connects to IonQ trapped-ion hardware (Aria, Forte) through
Amazon Braket managed devices.

Requires the optional ``[ionq]`` extra: ``pip install qufin[ionq]``.

References
----------
IonQ Hardware: https://ionq.com/
Amazon Braket IonQ: https://docs.aws.amazon.com/braket/latest/developerguide/braket-devices.html
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from qufin.backends.base import Backend, CircuitResult


@dataclass
class IonQConfig:
    """Configuration for IonQ backend.

    Attributes
    ----------
    device_name : str
        IonQ device identifier, e.g. "Aria-1", "Forte-1".
    shots : int
        Default number of shots per execution.
    api_key : str | None
        Optional IonQ API key (unused when routing through Braket).
    s3_bucket : str | None
        S3 bucket for Braket task results.
    s3_prefix : str
        S3 key prefix for results.
    """

    device_name: str = "Aria-1"
    shots: int = 1024
    api_key: str | None = None
    s3_bucket: str | None = None
    s3_prefix: str = "qufin-ionq-results"


# Device ARN mapping
IONQ_DEVICE_ARNS: dict[str, str] = {
    "Aria-1": "arn:aws:braket:us-east-1::device/qpu/ionq/Aria-1",
    "Aria-2": "arn:aws:braket:us-east-1::device/qpu/ionq/Aria-2",
    "Forte-1": "arn:aws:braket:us-east-1::device/qpu/ionq/Forte-1",
}


class IonQBackend(Backend):
    """Backend targeting IonQ trapped-ion hardware via Amazon Braket.

    Parameters
    ----------
    config : IonQConfig | None
        Configuration dataclass. Uses defaults if None.
    """

    def __init__(self, config: IonQConfig | None = None) -> None:
        try:
            from braket.aws import AwsDevice
        except ImportError as e:
            raise ImportError(
                "Amazon Braket SDK is required for IonQBackend. "
                "Install it with: pip install qufin[ionq]"
            ) from e

        self._config = config or IonQConfig()
        arn = IONQ_DEVICE_ARNS.get(self._config.device_name)
        if arn is None:
            raise ValueError(
                f"Unknown IonQ device '{self._config.device_name}'. "
                f"Available: {list(IONQ_DEVICE_ARNS.keys())}"
            )
        self._arn = arn

        from braket.aws import AwsDevice

        self._device = AwsDevice(self._arn)

    @property
    def backend_id(self) -> str:
        return f"ionq:{self._config.device_name}"

    def is_simulator(self) -> bool:
        return False

    def run(self, circuit: Any, shots: int = 1024) -> CircuitResult:
        """Execute a circuit on IonQ hardware via Braket."""
        if hasattr(circuit, "num_qubits"):
            braket_circuit = self._convert_qiskit(circuit)
        elif hasattr(circuit, "qubits") or hasattr(circuit, "_moments"):
            # Native Braket circuit (duck-typed)
            braket_circuit = circuit
        else:
            raise TypeError(
                f"Unsupported circuit type: {type(circuit).__name__}. "
                "Provide a Braket Circuit or Qiskit QuantumCircuit."
            )

        s3_loc = (
            self._config.s3_bucket or "amazon-braket-default",
            self._config.s3_prefix,
        )
        task = self._device.run(braket_circuit, s3_loc, shots=shots)
        result = task.result()
        counts = {k: int(v) for k, v in result.measurement_counts.items()}

        return CircuitResult(
            counts=counts,
            shots=shots,
            backend_id=self.backend_id,
            metadata={
                "device": self._config.device_name,
                "device_arn": self._arn,
            },
        )

    def _convert_qiskit(self, circuit: Any) -> Any:
        """Convert a Qiskit circuit to Braket via OpenQASM 2.0."""
        from braket.circuits import Circuit as BraketCircuit
        from qiskit import qasm2, transpile

        transpiled = transpile(
            circuit,
            basis_gates=["cx", "u3", "id", "measure"],
            optimization_level=0,
        )
        qasm_str = qasm2.dumps(transpiled)
        return BraketCircuit.from_ir(qasm_str, ir_type="OPENQASM")

    def statevector(self, circuit: Any) -> NDArray[np.complex128]:
        """Not supported on IonQ hardware."""
        raise NotImplementedError(
            "Statevector simulation is not available on IonQ hardware. "
            "Use MockBackend or QiskitAerBackend for statevector access."
        )
