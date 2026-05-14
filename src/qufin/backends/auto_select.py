"""Automatic backend selection based on circuit properties.

Analyzes circuit characteristics (qubit count, depth, gate types) and
selects the most suitable available backend, with a configurable
fallback chain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from qufin.backends.base import Backend

# ------------------------------------------------------------------
# Circuit analysis
# ------------------------------------------------------------------


@dataclass
class CircuitAnalysis:
    """Static analysis of a quantum circuit.

    Attributes
    ----------
    qubit_count : int
        Number of qubits.
    depth : int
        Circuit depth (longest path).
    connectivity_graph : dict[int, set[int]]
        Adjacency mapping for two-qubit gates.
    gate_counts : dict[str, int]
        Counts per gate name.
    """

    qubit_count: int = 0
    depth: int = 0
    connectivity_graph: dict[int, set[int]] = field(default_factory=dict)
    gate_counts: dict[str, int] = field(default_factory=dict)

    @property
    def total_gates(self) -> int:
        """Total number of gates in the circuit."""
        return sum(self.gate_counts.values())

    @property
    def two_qubit_count(self) -> int:
        """Number of two-qubit gates (cx, cz, swap, etc.)."""
        two_q_names = {"cx", "cz", "swap", "cy", "rzz", "rxx", "ryy", "ecr"}
        return sum(
            v for k, v in self.gate_counts.items() if k in two_q_names
        )


def analyze_circuit(circuit: Any) -> CircuitAnalysis:
    """Extract structural properties from a quantum circuit.

    Parameters
    ----------
    circuit : qiskit.QuantumCircuit
        The circuit to analyze.

    Returns
    -------
    CircuitAnalysis
    """
    qubit_count: int = getattr(circuit, "num_qubits", 0)
    depth: int = 0
    gate_counts: dict[str, int] = {}
    connectivity: dict[int, set[int]] = {}

    if hasattr(circuit, "depth"):
        depth = circuit.depth()

    if hasattr(circuit, "data"):
        for instruction in circuit.data:
            name = instruction.operation.name
            gate_counts[name] = gate_counts.get(name, 0) + 1

            qubit_indices = [
                circuit.find_bit(q).index for q in instruction.qubits
            ]
            if len(qubit_indices) >= 2:
                for i, qi in enumerate(qubit_indices):
                    for qj in qubit_indices[i + 1 :]:
                        connectivity.setdefault(qi, set()).add(qj)
                        connectivity.setdefault(qj, set()).add(qi)

    return CircuitAnalysis(
        qubit_count=qubit_count,
        depth=depth,
        connectivity_graph=connectivity,
        gate_counts=gate_counts,
    )


# ------------------------------------------------------------------
# Backend registry
# ------------------------------------------------------------------


class BackendRegistry:
    """Registry for discovering and managing available backends.

    Examples
    --------
    >>> registry = BackendRegistry()
    >>> registry.register("mock", MockBackend)
    >>> registry.get("mock")
    <MockBackend instance>
    """

    def __init__(self) -> None:
        self._backends: dict[str, type[Backend]] = {}
        self._instances: dict[str, Backend] = {}

    def register(
        self,
        name: str,
        backend_cls: type[Backend],
    ) -> None:
        """Register a backend class by name."""
        self._backends[name] = backend_cls

    def unregister(self, name: str) -> None:
        """Remove a backend from the registry."""
        self._backends.pop(name, None)
        self._instances.pop(name, None)

    def get(self, name: str, **kwargs: Any) -> Backend:
        """Get or create a backend instance by name.

        Parameters
        ----------
        name : str
            Registered backend name.
        **kwargs
            Passed to the backend constructor on first creation.

        Returns
        -------
        Backend

        Raises
        ------
        KeyError
            If the name is not registered.
        """
        if name not in self._backends:
            raise KeyError(
                f"Backend '{name}' is not registered. "
                f"Available: {list(self._backends.keys())}"
            )
        if name not in self._instances:
            self._instances[name] = self._backends[name](**kwargs)
        return self._instances[name]

    @property
    def registered_names(self) -> list[str]:
        """List of registered backend names."""
        return list(self._backends.keys())

    def clear(self) -> None:
        """Remove all registered backends."""
        self._backends.clear()
        self._instances.clear()


# ------------------------------------------------------------------
# Availability probing
# ------------------------------------------------------------------


def get_available_backends() -> dict[str, bool]:
    """Probe which quantum backends are importable.

    Returns
    -------
    dict[str, bool]
        Mapping of backend name to availability.
    """
    available: dict[str, bool] = {}

    # MockBackend is always available
    available["mock"] = True

    # Qiskit Aer
    try:
        import qiskit_aer  # noqa: F401

        available["qiskit_aer"] = True
    except ImportError:
        available["qiskit_aer"] = False

    # CUDA-Q
    try:
        import cudaq  # type: ignore[import-untyped]  # noqa: F401

        available["cudaq"] = True
    except ImportError:
        available["cudaq"] = False

    # PennyLane
    try:
        import pennylane  # noqa: F401

        available["pennylane"] = True
    except ImportError:
        available["pennylane"] = False

    # Cirq
    try:
        import cirq  # noqa: F401

        available["cirq"] = True
    except ImportError:
        available["cirq"] = False

    # Braket
    try:
        import braket  # noqa: F401

        available["braket"] = True
    except ImportError:
        available["braket"] = False

    return available


# ------------------------------------------------------------------
# Auto-selection logic
# ------------------------------------------------------------------

# Qubit thresholds for backend selection heuristics
_SMALL_CIRCUIT = 15
_MEDIUM_CIRCUIT = 30


def auto_select_backend(
    circuit: Any,
    available_backends: dict[str, bool] | None = None,
    preference: str | None = None,
) -> Backend:
    """Automatically select the best backend for a circuit.

    Selection priorities (unless overridden by *preference*):

    1. If the user specifies a *preference* and it is available, use it.
    2. For large circuits (>30 qubits): prefer CUDA-Q GPU.
    3. For medium circuits (16-30 qubits): prefer Qiskit Aer, fallback
       to CUDA-Q.
    4. For small circuits (<=15 qubits): Qiskit Aer or MockBackend.

    Fallback chain: ``cudaq -> qiskit_aer -> mock``.

    Parameters
    ----------
    circuit : qiskit.QuantumCircuit
        Circuit to select a backend for.
    available_backends : dict[str, bool] | None
        Override availability map; if None, probed automatically.
    preference : str | None
        Preferred backend name (e.g. ``"cudaq"``, ``"qiskit_aer"``).

    Returns
    -------
    Backend
        An instantiated backend.
    """
    if available_backends is None:
        available_backends = get_available_backends()

    analysis = analyze_circuit(circuit)

    # Honour explicit preference
    if preference:
        backend = _try_create_backend(preference, available_backends, analysis)
        if backend is not None:
            return backend

    # Large circuits: GPU first
    if analysis.qubit_count > _MEDIUM_CIRCUIT:
        for name in ("cudaq", "qiskit_aer", "mock"):
            backend = _try_create_backend(name, available_backends, analysis)
            if backend is not None:
                return backend

    # Medium circuits
    if analysis.qubit_count > _SMALL_CIRCUIT:
        for name in ("qiskit_aer", "cudaq", "mock"):
            backend = _try_create_backend(name, available_backends, analysis)
            if backend is not None:
                return backend

    # Small circuits or final fallback
    for name in ("qiskit_aer", "mock"):
        backend = _try_create_backend(name, available_backends, analysis)
        if backend is not None:
            return backend

    # Absolute fallback
    from qufin.backends.mock import MockBackend

    return MockBackend()


def _try_create_backend(
    name: str,
    available: dict[str, bool],
    analysis: CircuitAnalysis,
) -> Backend | None:
    """Attempt to instantiate a backend by name.

    Returns None if unavailable or import fails.
    """
    if not available.get(name, False):
        return None

    try:
        if name == "mock":
            from qufin.backends.mock import MockBackend

            return MockBackend()

        if name == "qiskit_aer":
            from qufin.backends.qiskit_backend import QiskitAerBackend

            return QiskitAerBackend()

        if name == "cudaq":
            from qufin.backends.cudaq_backend import CudaQBackend

            target = "nvidia"
            if analysis.qubit_count > _MEDIUM_CIRCUIT:
                target = "nvidia-mgpu"
            return CudaQBackend(target=target)

        if name == "pennylane":
            from qufin.backends.pennylane_backend import PennyLaneBackend

            return PennyLaneBackend()

        if name == "cirq":
            from qufin.backends.cirq_backend import CirqBackend

            return CirqBackend()

        if name == "braket":
            from qufin.backends.braket_backend import BraketBackend

            return BraketBackend()

    except Exception:
        return None

    return None
