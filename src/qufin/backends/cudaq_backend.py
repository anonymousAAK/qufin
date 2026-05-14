"""NVIDIA CUDA-Q backend adapter.

Provides GPU-accelerated quantum circuit simulation via the CUDA-Q SDK.
Supports single-GPU state vector simulation and multi-GPU for >30 qubits.

Requires: pip install cuda-quantum
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from qufin.backends.base import Backend, CircuitResult

try:
    import cudaq  # type: ignore[import-untyped]

    CUDAQ_AVAILABLE = True
except ImportError:
    cudaq = None
    CUDAQ_AVAILABLE = False


def _require_cudaq() -> None:
    """Raise ImportError if CUDA-Q is not installed."""
    if not CUDAQ_AVAILABLE:
        raise ImportError(
            "CUDA-Q is required for GPU simulation. "
            "Install with: pip install cuda-quantum"
        )


class CudaQBackend(Backend):
    """GPU-accelerated backend using NVIDIA CUDA-Q.

    Parameters
    ----------
    target : str
        CUDA-Q target: ``"nvidia"`` (single GPU) or
        ``"nvidia-mgpu"`` (multi-GPU).
    seed : int | None
        Random seed for reproducibility.
    multi_gpu_threshold : int
        Qubit count above which multi-GPU target is used
        automatically when ``target="auto"``.
    """

    def __init__(
        self,
        target: str = "nvidia",
        seed: int | None = 42,
        multi_gpu_threshold: int = 30,
    ) -> None:
        _require_cudaq()
        self._target = target
        self._seed = seed
        self._multi_gpu_threshold = multi_gpu_threshold
        cudaq.set_target(target)
        if seed is not None:
            cudaq.set_random_seed(seed)

    @property
    def backend_id(self) -> str:
        return f"cudaq-{self._target}"

    # ------------------------------------------------------------------
    # Circuit translation
    # ------------------------------------------------------------------

    @staticmethod
    def _translate_circuit(circuit: Any) -> Any:
        """Translate a Qiskit QuantumCircuit to a CUDA-Q kernel.

        Parameters
        ----------
        circuit : qiskit.QuantumCircuit
            Input circuit using standard gates.

        Returns
        -------
        tuple[cudaq.Kernel, int]
            The CUDA-Q kernel and qubit count.
        """
        _require_cudaq()
        n_qubits: int = circuit.num_qubits

        kernel = cudaq.make_kernel()
        qubits = kernel.qalloc(n_qubits)

        _GATE_MAP = {
            "h": "h",
            "x": "x",
            "y": "y",
            "z": "z",
            "s": "s",
            "t": "t",
            "sdg": "sdg",
            "tdg": "tdg",
        }

        _PARAM_GATE_MAP = {
            "rx": "rx",
            "ry": "ry",
            "rz": "rz",
            "r1": "r1",
        }

        for instruction in circuit.data:
            gate_name = instruction.operation.name
            qubit_indices = [
                circuit.find_bit(q).index for q in instruction.qubits
            ]
            params = instruction.operation.params

            if gate_name in _GATE_MAP:
                method = getattr(kernel, _GATE_MAP[gate_name])
                method(qubits[qubit_indices[0]])
            elif gate_name in _PARAM_GATE_MAP:
                method = getattr(kernel, _PARAM_GATE_MAP[gate_name])
                method(float(params[0]), qubits[qubit_indices[0]])
            elif gate_name == "cx":
                kernel.cx(qubits[qubit_indices[0]], qubits[qubit_indices[1]])
            elif gate_name == "cz":
                kernel.cz(qubits[qubit_indices[0]], qubits[qubit_indices[1]])
            elif gate_name == "swap":
                kernel.swap(
                    qubits[qubit_indices[0]], qubits[qubit_indices[1]]
                )
            elif gate_name == "ccx":
                kernel.ccx(
                    qubits[qubit_indices[0]],
                    qubits[qubit_indices[1]],
                    qubits[qubit_indices[2]],
                )
            elif gate_name == "measure":
                kernel.mz(qubits[qubit_indices[0]])
            elif gate_name == "barrier":
                pass  # No-op in CUDA-Q
            else:
                raise ValueError(
                    f"Unsupported gate '{gate_name}' for CUDA-Q translation"
                )

        return kernel, n_qubits

    # ------------------------------------------------------------------
    # Backend interface
    # ------------------------------------------------------------------

    def run(self, circuit: Any, shots: int = 1024) -> CircuitResult:
        """Execute a circuit and return measurement results.

        Parameters
        ----------
        circuit : qiskit.QuantumCircuit
            Circuit to execute.
        shots : int
            Number of measurement shots.

        Returns
        -------
        CircuitResult
        """
        _require_cudaq()

        n_qubits = circuit.num_qubits
        if n_qubits > self._multi_gpu_threshold and self._target != "nvidia-mgpu":
            cudaq.set_target("nvidia-mgpu")

        kernel, _ = self._translate_circuit(circuit)
        result = cudaq.sample(kernel, shots_count=shots)

        counts: dict[str, int] = {}
        for bitstring in result:
            counts[bitstring] = result.count(bitstring)

        return CircuitResult(
            counts=counts,
            shots=shots,
            backend_id=self.backend_id,
        )

    def statevector(self, circuit: Any) -> NDArray[np.complex128]:
        """Return the statevector for a circuit.

        Parameters
        ----------
        circuit : qiskit.QuantumCircuit
            Circuit (no measurements).

        Returns
        -------
        NDArray[np.complex128]
        """
        _require_cudaq()
        kernel, _ = self._translate_circuit(circuit)
        state = cudaq.get_state(kernel)
        return np.array(state, dtype=np.complex128)

    # ------------------------------------------------------------------
    # GPU utilities
    # ------------------------------------------------------------------

    def benchmark_simulation(self, n_qubits: int, depth: int = 10) -> dict:
        """Time a random circuit simulation on GPU.

        Parameters
        ----------
        n_qubits : int
            Number of qubits.
        depth : int
            Circuit depth (layers of random gates).

        Returns
        -------
        dict
            Keys: ``n_qubits``, ``depth``, ``elapsed_seconds``,
            ``target``.
        """
        _require_cudaq()

        kernel = cudaq.make_kernel()
        qubits = kernel.qalloc(n_qubits)
        rng = np.random.default_rng(self._seed)

        for _ in range(depth):
            for q_idx in range(n_qubits):
                angle = float(rng.uniform(0, 2 * np.pi))
                kernel.rx(angle, qubits[q_idx])
                kernel.rz(angle, qubits[q_idx])
            for q_idx in range(0, n_qubits - 1, 2):
                kernel.cx(qubits[q_idx], qubits[q_idx + 1])

        start = time.perf_counter()
        cudaq.get_state(kernel)
        elapsed = time.perf_counter() - start

        return {
            "n_qubits": n_qubits,
            "depth": depth,
            "elapsed_seconds": elapsed,
            "target": self._target,
        }

    @staticmethod
    def get_gpu_info() -> dict:
        """Query available GPU device information.

        Returns
        -------
        dict
            Keys: ``available`` (bool), ``num_gpus`` (int),
            ``targets`` (list[str]).
        """
        if not CUDAQ_AVAILABLE:
            return {"available": False, "num_gpus": 0, "targets": []}

        num_gpus = cudaq.num_available_gpus()
        targets = [t.name for t in cudaq.get_targets()]
        return {
            "available": True,
            "num_gpus": num_gpus,
            "targets": targets,
        }

    def is_simulator(self) -> bool:
        """CUDA-Q targets are simulators (not real hardware)."""
        return True
