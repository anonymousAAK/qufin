"""Quantinuum (H-Series) backend adapter.

Connects to Quantinuum H-Series trapped-ion hardware via pytket-quantinuum.

Requires the optional ``[quantinuum]`` extra:
``pip install qufin[quantinuum]``.

References
----------
Quantinuum H-Series: https://www.quantinuum.com/
pytket-quantinuum: https://docs.quantinuum.com/tket/extensions/pytket-quantinuum/
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from qufin.backends.base import Backend, CircuitResult


@dataclass
class QuantinuumConfig:
    """Configuration for Quantinuum backend.

    Attributes
    ----------
    device_name : str
        Quantinuum device name, e.g. "H1-1", "H2-1", "H1-1E" (emulator).
    shots : int
        Default number of shots per execution.
    """

    device_name: str = "H1-1"
    shots: int = 1024


# Known devices and their qubit counts
QUANTINUUM_DEVICES: dict[str, int] = {
    "H1-1": 20,
    "H1-1E": 20,  # emulator
    "H2-1": 56,
    "H2-1E": 56,  # emulator
}


class QuantinuumBackend(Backend):
    """Backend targeting Quantinuum H-Series trapped-ion hardware.

    Parameters
    ----------
    config : QuantinuumConfig | None
        Configuration dataclass. Uses defaults if None.
    """

    def __init__(self, config: QuantinuumConfig | None = None) -> None:
        try:
            from pytket.extensions.quantinuum import QuantinuumBackend as _QtBackend
        except ImportError as e:
            raise ImportError(
                "pytket and pytket-quantinuum are required for QuantinuumBackend. "
                "Install them with: pip install qufin[quantinuum]"
            ) from e

        self._config = config or QuantinuumConfig()
        if self._config.device_name not in QUANTINUUM_DEVICES:
            raise ValueError(
                f"Unknown Quantinuum device '{self._config.device_name}'. "
                f"Available: {list(QUANTINUUM_DEVICES.keys())}"
            )

        from pytket.extensions.quantinuum import QuantinuumBackend as _QtBackend

        self._tket_backend = _QtBackend(device_name=self._config.device_name)

    @property
    def backend_id(self) -> str:
        return f"quantinuum:{self._config.device_name}"

    def is_simulator(self) -> bool:
        return self._config.device_name.endswith("E")

    def run(self, circuit: Any, shots: int = 1024) -> CircuitResult:
        """Execute a circuit on Quantinuum hardware."""
        if hasattr(circuit, "num_qubits"):
            # Qiskit circuit: convert via OpenQASM 2.0
            tket_circuit = self._convert_qiskit(circuit)
        else:
            tket_circuit = circuit

        compiled = self._tket_backend.get_compiled_circuit(tket_circuit)
        handle = self._tket_backend.process_circuit(compiled, n_shots=shots)
        result = self._tket_backend.get_result(handle)

        counts: dict[str, int] = {}
        for bitstring, count in result.get_counts().items():
            key = "".join(str(b) for b in bitstring)
            counts[key] = int(count)

        return CircuitResult(
            counts=counts,
            shots=shots,
            backend_id=self.backend_id,
            metadata={
                "device": self._config.device_name,
                "max_qubits": QUANTINUUM_DEVICES[self._config.device_name],
            },
        )

    def _convert_qiskit(self, circuit: Any) -> Any:
        """Convert a Qiskit circuit to a pytket circuit via OpenQASM 2.0."""
        from pytket.qasm import circuit_from_qasm_str
        from qiskit import qasm2, transpile

        transpiled = transpile(
            circuit,
            basis_gates=["cx", "u3", "id", "measure"],
            optimization_level=0,
        )
        qasm_str = qasm2.dumps(transpiled)
        return circuit_from_qasm_str(qasm_str)

    def statevector(self, circuit: Any) -> NDArray[np.complex128]:
        """Not supported on Quantinuum hardware."""
        raise NotImplementedError(
            "Statevector simulation is not available on Quantinuum hardware. "
            "Use MockBackend or QiskitAerBackend for statevector access."
        )
