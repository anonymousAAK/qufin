"""PennyLane Lightning backend adapter.

Provides a qufin Backend interface wrapping PennyLane's
default.qubit or lightning.qubit simulators, with parameter-shift
gradient support and cross-framework verification utilities.

Requires: pip install qufin[pennylane]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from qufin.backends.base import Backend, CircuitResult

try:
    import pennylane as _qml

    _HAS_PENNYLANE = True
except ImportError:
    _qml = None  # type: ignore[assignment]
    _HAS_PENNYLANE = False


def _require_pennylane() -> Any:
    """Return the pennylane module or raise ImportError."""
    if not _HAS_PENNYLANE:
        raise ImportError(
            "PennyLane is required. Install with: pip install qufin[pennylane]"
        )
    return _qml


@dataclass
class GradientResult:
    """Container for expectation values and their gradients.

    Attributes
    ----------
    expectation : float
        The expectation value of the cost function.
    gradients : NDArray[np.float64]
        Parameter gradients computed via the parameter-shift rule.
    """

    expectation: float
    gradients: NDArray[np.float64]


class PennyLaneBackend(Backend):
    """PennyLane backend adapter.

    Converts Qiskit circuits to PennyLane via ``qml.from_qiskit()``
    conversion, or runs native PennyLane circuits.  Supports
    parameter-shift gradient computation for variational algorithms.

    Parameters
    ----------
    device_name : str
        PennyLane device name (``"default.qubit"`` or ``"lightning.qubit"``).
    n_qubits : int
        Number of qubits (required for device initialisation).
    diff_method : str
        Differentiation method for QNodes (default ``"parameter-shift"``).
    """

    SUPPORTED_DEVICES = ("default.qubit", "lightning.qubit")

    def __init__(
        self,
        device_name: str = "default.qubit",
        n_qubits: int = 10,
        diff_method: str = "parameter-shift",
    ) -> None:
        self._qml = _require_pennylane()

        if device_name not in self.SUPPORTED_DEVICES:
            raise ValueError(
                f"Unsupported device '{device_name}'. "
                f"Choose from {self.SUPPORTED_DEVICES}."
            )

        self._device_name = device_name
        self._n_qubits = n_qubits
        self._diff_method = diff_method
        self._device = self._qml.device(device_name, wires=n_qubits)

    # ------------------------------------------------------------------
    # Backend ABC implementation
    # ------------------------------------------------------------------

    @property
    def backend_id(self) -> str:
        return f"pennylane:{self._device_name}"

    def run(self, circuit: Any, shots: int = 1024) -> CircuitResult:
        """Execute a circuit and return measurement results.

        Accepts either a PennyLane QNode or a Qiskit QuantumCircuit.
        For Qiskit circuits, converts via ``qml.from_qiskit()``.
        """
        if hasattr(circuit, "num_qubits"):
            return self._run_qiskit_circuit(circuit, shots)

        # Native PennyLane QNode
        if callable(circuit):
            results = circuit()
            if isinstance(results, np.ndarray):
                probs = results
                counts: dict[str, int] = {}
                for i, p in enumerate(probs):
                    if p > 1e-10:
                        bs = format(i, f"0{self._n_qubits}b")
                        counts[bs] = round(p * shots)
                return CircuitResult(
                    counts=counts, shots=shots, backend_id=self.backend_id
                )

        return CircuitResult(counts={}, shots=shots, backend_id=self.backend_id)

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

    # ------------------------------------------------------------------
    # Qiskit circuit execution (with parameterised-circuit support)
    # ------------------------------------------------------------------

    def _run_qiskit_circuit(
        self,
        circuit: Any,
        shots: int,
        parameter_values: dict[Any, float] | None = None,
    ) -> CircuitResult:
        """Run a Qiskit circuit by converting to PennyLane.

        Parameters
        ----------
        circuit : qiskit.QuantumCircuit
            The circuit to execute.
        shots : int
            Number of measurement shots.
        parameter_values : dict, optional
            Mapping of Qiskit ``Parameter`` objects to float values.
            If provided, the circuit is bound before conversion.
        """
        qml = self._qml

        bound = self._bind_parameters(circuit, parameter_values)
        n_qubits = bound.num_qubits
        dev = qml.device(self._device_name, wires=n_qubits, shots=shots)

        @qml.qnode(dev)
        def converted_circuit():
            qml.from_qiskit(bound)()
            return qml.counts()

        try:
            result = converted_circuit()
            counts = {k: int(v) for k, v in result.items()}
        except Exception:
            counts = self._sample_from_statevector(bound, shots, n_qubits)

        return CircuitResult(
            counts=counts, shots=shots, backend_id=self.backend_id
        )

    @staticmethod
    def _bind_parameters(
        circuit: Any,
        parameter_values: dict[Any, float] | None,
    ) -> Any:
        """Bind parameter values to a Qiskit circuit if needed."""
        if parameter_values is None:
            return circuit
        if not hasattr(circuit, "parameters") or len(circuit.parameters) == 0:
            return circuit
        return circuit.assign_parameters(parameter_values)

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

    # ------------------------------------------------------------------
    # Parameter-shift gradient support
    # ------------------------------------------------------------------

    def gradient(
        self,
        cost_fn: Any,
        params: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute gradients of *cost_fn* using the parameter-shift rule.

        Parameters
        ----------
        cost_fn : callable
            A PennyLane-compatible cost function (QNode) that accepts a
            1-D parameter array and returns a scalar expectation value.
        params : NDArray
            Current parameter values.

        Returns
        -------
        NDArray[np.float64]
            Gradient vector with the same shape as *params*.
        """
        qml = self._qml
        grad_fn = qml.grad(cost_fn)
        return np.asarray(grad_fn(params), dtype=np.float64)

    def run_with_gradient(
        self,
        cost_fn: Any,
        params: NDArray[np.float64],
    ) -> GradientResult:
        """Evaluate *cost_fn* and its gradient in one call.

        Parameters
        ----------
        cost_fn : callable
            A PennyLane QNode returning a scalar.
        params : NDArray
            Current parameter values.

        Returns
        -------
        GradientResult
            Container with ``.expectation`` and ``.gradients``.
        """
        expectation = float(cost_fn(params))
        gradients = self.gradient(cost_fn, params)
        return GradientResult(expectation=expectation, gradients=gradients)

    def make_cost_qnode(
        self,
        ansatz_fn: Any,
        hamiltonian: Any,
        n_qubits: int | None = None,
    ) -> Any:
        """Create a differentiable QNode from an ansatz and Hamiltonian.

        Parameters
        ----------
        ansatz_fn : callable
            A function ``ansatz_fn(params, wires)`` that applies gates.
        hamiltonian : pennylane.Hamiltonian
            Observable to measure.
        n_qubits : int, optional
            Qubit count (defaults to ``self._n_qubits``).

        Returns
        -------
        QNode
            A differentiable cost function ``cost(params) -> float``.
        """
        qml = self._qml
        nq = n_qubits or self._n_qubits
        dev = qml.device(self._device_name, wires=nq)

        @qml.qnode(dev, diff_method=self._diff_method)
        def cost(params):
            ansatz_fn(params, wires=range(nq))
            return qml.expval(hamiltonian)

        return cost

    # ------------------------------------------------------------------
    # Cross-framework verification
    # ------------------------------------------------------------------

    @staticmethod
    def verify_against_qiskit(
        circuit: Any,
        shots: int = 4096,
        atol: float = 0.05,
    ) -> dict[str, Any]:
        """Run the same Qiskit circuit on both backends and compare.

        Parameters
        ----------
        circuit : qiskit.QuantumCircuit
            Circuit to verify (must include measurements for the Qiskit
            leg; PennyLane side uses probability-based sampling).
        shots : int
            Number of shots per backend.
        atol : float
            Absolute tolerance on probability differences per bitstring.

        Returns
        -------
        dict
            ``{"match": bool, "qiskit_probs": dict, "pennylane_probs": dict,
              "max_diff": float}``
        """
        qml = _require_pennylane()

        # --- Qiskit execution ---
        try:
            from qiskit_aer import AerSimulator

            sim = AerSimulator()
            job = sim.run(circuit, shots=shots)
            qiskit_counts = job.result().get_counts()
        except ImportError:
            from qiskit.quantum_info import Statevector

            sv = Statevector.from_instruction(circuit.remove_final_measurements(inplace=False))
            qiskit_counts = sv.sample_counts(shots)

        n_qubits = circuit.num_qubits
        qiskit_probs = {k: v / shots for k, v in qiskit_counts.items()}

        # --- PennyLane execution ---
        dev = qml.device("default.qubit", wires=n_qubits)

        bare = circuit.remove_final_measurements(inplace=False)

        @qml.qnode(dev)
        def pl_circuit():
            qml.from_qiskit(bare)()
            return qml.probs(wires=range(n_qubits))

        try:
            probs_arr = np.array(pl_circuit())
        except Exception:
            probs_arr = np.zeros(2**n_qubits)

        rng = np.random.default_rng(42)
        pl_samples = rng.choice(len(probs_arr), size=shots, p=probs_arr)
        pl_counts: dict[str, int] = {}
        for s in pl_samples:
            bs = format(s, f"0{n_qubits}b")
            pl_counts[bs] = pl_counts.get(bs, 0) + 1
        pl_probs = {k: v / shots for k, v in pl_counts.items()}

        # --- Compare ---
        all_keys = set(qiskit_probs) | set(pl_probs)
        max_diff = 0.0
        for k in all_keys:
            diff = abs(qiskit_probs.get(k, 0.0) - pl_probs.get(k, 0.0))
            max_diff = max(max_diff, diff)

        return {
            "match": max_diff <= atol,
            "qiskit_probs": qiskit_probs,
            "pennylane_probs": pl_probs,
            "max_diff": max_diff,
        }

    # ------------------------------------------------------------------
    # Circuit statistics
    # ------------------------------------------------------------------

    def circuit_stats(self, circuit: Any) -> dict[str, int]:
        """Return basic statistics about a Qiskit circuit.

        Parameters
        ----------
        circuit : qiskit.QuantumCircuit
            The circuit to analyse.

        Returns
        -------
        dict
            ``{"n_qubits": int, "depth": int, "gate_count": int}``
        """
        if not hasattr(circuit, "num_qubits"):
            raise ValueError("circuit_stats requires a Qiskit QuantumCircuit")

        return {
            "n_qubits": circuit.num_qubits,
            "depth": circuit.depth(),
            "gate_count": circuit.size(),
        }
