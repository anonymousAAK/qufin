"""IBM Quantum Runtime backend adapter.

Connects to IBM Quantum hardware (e.g., ibm_kingston, Heron r2)
or cloud simulators via qiskit-ibm-runtime.

Requires the optional ``[ibm]`` extra: ``pip install qufin[ibm]``.

References
----------
IBM Quantum Platform: https://quantum.ibm.com/
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from qufin.backends.base import Backend, CircuitResult


class IBMRuntimeBackend(Backend):
    """Backend using IBM Quantum Runtime (real hardware or cloud sim).

    Parameters
    ----------
    backend_name : str
        IBM backend identifier, e.g. "ibm_kingston", "ibm_brisbane",
        "ibm_sherbrooke". Use "ibmq_qasm_simulator" for cloud simulator.
    channel : str
        Channel: "ibm_quantum" (Open Plan) or "ibm_cloud".
    instance : str | None
        Hub/group/project instance, e.g. "ibm-q/open/main".
    optimization_level : int
        Transpiler optimization level (0-3). Higher = slower but better.
    resilience_level : int
        Error mitigation level (0-2). 0 = none, 1 = M3 readout mitigation,
        2 = ZNE + M3.
    """

    def __init__(
        self,
        backend_name: str = "ibm_brisbane",
        channel: str = "ibm_quantum",
        instance: str | None = None,
        optimization_level: int = 1,
        resilience_level: int = 1,
    ) -> None:
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService
        except ImportError as e:
            raise ImportError(
                "qiskit-ibm-runtime is required for IBMRuntimeBackend. "
                "Install it with: pip install qufin[ibm]"
            ) from e

        service_kwargs: dict[str, Any] = {"channel": channel}
        if instance is not None:
            service_kwargs["instance"] = instance

        self._service = QiskitRuntimeService(**service_kwargs)
        self._backend = self._service.backend(backend_name)
        self._backend_name = backend_name
        self._optimization_level = optimization_level
        self._resilience_level = resilience_level

    @property
    def backend_id(self) -> str:
        return f"ibm-runtime-{self._backend_name}"

    @property
    def num_qubits(self) -> int:
        """Number of qubits available on this backend."""
        return self._backend.num_qubits

    def is_simulator(self) -> bool:
        return self._backend.simulator

    def run(self, circuit: Any, shots: int = 1024) -> CircuitResult:
        """Execute a circuit on IBM hardware via Sampler primitive."""
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        from qiskit_ibm_runtime import SamplerV2

        pm = generate_preset_pass_manager(
            optimization_level=self._optimization_level,
            backend=self._backend,
        )
        isa_circuit = pm.run(circuit)

        sampler = SamplerV2(mode=self._backend)
        job = sampler.run([isa_circuit], shots=shots)
        result = job.result()

        # Extract counts from SamplerV2 result
        pub_result = result[0]
        counts_dict: dict[str, int] = {}
        # SamplerV2 returns BitArray; convert to counts
        try:
            bit_array = pub_result.data.meas
            counts_dict = dict(bit_array.get_counts())
        except AttributeError:
            # Fallback for different result formats
            try:
                counts_dict = dict(pub_result.data.c.get_counts())
            except AttributeError:
                # Try classical register iteration
                for cr_name in dir(pub_result.data):
                    if not cr_name.startswith("_"):
                        cr = getattr(pub_result.data, cr_name)
                        if hasattr(cr, "get_counts"):
                            counts_dict = dict(cr.get_counts())
                            break

        # Flatten any spaces in bitstrings
        flat_counts = {k.replace(" ", ""): v for k, v in counts_dict.items()}

        return CircuitResult(
            counts=flat_counts,
            shots=shots,
            backend_id=self.backend_id,
            metadata={
                "job_id": job.job_id(),
                "backend": self._backend_name,
                "optimization_level": self._optimization_level,
                "resilience_level": self._resilience_level,
                "circuit_depth": isa_circuit.depth(),
                "n_gates": isa_circuit.size(),
            },
        )

    def statevector(self, circuit: Any) -> NDArray[np.complex128]:
        """Not supported on real hardware."""
        raise NotImplementedError(
            "Statevector simulation is not available on IBM hardware. "
            "Use QiskitAerBackend for statevector access."
        )
