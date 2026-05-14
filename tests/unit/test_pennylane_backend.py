"""Tests for PennyLane backend adapter.

All tests mock PennyLane so they pass WITHOUT pennylane installed.
"""

from __future__ import annotations

import sys
import types
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Build a fake ``pennylane`` module that is injected into sys.modules so
# the import guard in pennylane_backend.py resolves without the real package.
# ---------------------------------------------------------------------------


def _make_mock_qml() -> types.ModuleType:
    """Create a minimal mock of the pennylane module."""
    qml = types.ModuleType("pennylane")

    # Device mock
    def _device(name: str, wires: int, **kwargs: Any) -> MagicMock:
        dev = MagicMock()
        dev.name = name
        dev.num_wires = wires
        dev.shots = kwargs.get("shots")
        return dev

    qml.device = _device  # type: ignore[attr-defined]

    # QNode decorator: return the function as-is (thin passthrough)
    def _qnode(dev: Any, diff_method: str = "parameter-shift"):
        def wrapper(fn: Any) -> Any:
            return fn
        return wrapper

    qml.qnode = _qnode  # type: ignore[attr-defined]

    # Observable helpers
    qml.expval = MagicMock(return_value=0.0)  # type: ignore[attr-defined]
    qml.counts = MagicMock(return_value={})  # type: ignore[attr-defined]
    qml.probs = MagicMock(return_value=np.array([0.5, 0.5]))  # type: ignore[attr-defined]
    qml.state = MagicMock(  # type: ignore[attr-defined]
        return_value=np.array([1, 0], dtype=np.complex128),
    )
    qml.from_qiskit = MagicMock(return_value=MagicMock())  # type: ignore[attr-defined]

    # Gradient helper
    def _grad(fn: Any) -> Any:
        def grad_fn(params: Any) -> Any:
            return np.zeros_like(params)
        return grad_fn

    qml.grad = _grad  # type: ignore[attr-defined]

    # Hamiltonian
    qml.Hamiltonian = MagicMock  # type: ignore[attr-defined]

    return qml


@pytest.fixture(autouse=True)
def _patch_pennylane(monkeypatch: pytest.MonkeyPatch):
    """Inject fake pennylane into sys.modules for every test."""
    mock_qml = _make_mock_qml()
    monkeypatch.setitem(sys.modules, "pennylane", mock_qml)

    # Also patch the module-level sentinel in pennylane_backend
    import qufin.backends.pennylane_backend as plb

    monkeypatch.setattr(plb, "_HAS_PENNYLANE", True)
    monkeypatch.setattr(plb, "_qml", mock_qml)

    yield mock_qml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeQiskitCircuit:
    """Minimal Qiskit QuantumCircuit stand-in."""

    def __init__(self, n_qubits: int = 2, depth: int = 3, size: int = 5):
        self.num_qubits = n_qubits
        self._depth = depth
        self._size = size
        self.parameters: list[Any] = []

    def depth(self) -> int:
        return self._depth

    def size(self) -> int:
        return self._size

    def remove_final_measurements(self, inplace: bool = False):
        return _FakeQiskitCircuit(self.num_qubits, self._depth, self._size)

    def assign_parameters(self, mapping: dict[Any, float]):
        return _FakeQiskitCircuit(self.num_qubits, self._depth, self._size)


class _ParameterisedCircuit(_FakeQiskitCircuit):
    """Fake circuit with unbound parameters."""

    def __init__(self):
        super().__init__(n_qubits=2)
        self.parameters = ["theta", "phi"]


# ---------------------------------------------------------------------------
# Tests: Initialisation
# ---------------------------------------------------------------------------


class TestInit:
    def test_default_init(self) -> None:
        from qufin.backends.pennylane_backend import PennyLaneBackend

        backend = PennyLaneBackend()
        assert backend._device_name == "default.qubit"
        assert backend._n_qubits == 10

    def test_lightning_init(self) -> None:
        from qufin.backends.pennylane_backend import PennyLaneBackend

        backend = PennyLaneBackend(device_name="lightning.qubit", n_qubits=4)
        assert backend._device_name == "lightning.qubit"

    def test_unsupported_device_raises(self) -> None:
        from qufin.backends.pennylane_backend import PennyLaneBackend

        with pytest.raises(ValueError, match="Unsupported device"):
            PennyLaneBackend(device_name="braket.aws")

    def test_backend_id(self) -> None:
        from qufin.backends.pennylane_backend import PennyLaneBackend

        b = PennyLaneBackend(device_name="lightning.qubit")
        assert b.backend_id == "pennylane:lightning.qubit"

    def test_is_simulator(self) -> None:
        from qufin.backends.pennylane_backend import PennyLaneBackend

        assert PennyLaneBackend().is_simulator() is True

    def test_import_error_without_pennylane(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import qufin.backends.pennylane_backend as plb

        monkeypatch.setattr(plb, "_HAS_PENNYLANE", False)
        monkeypatch.setattr(plb, "_qml", None)

        with pytest.raises(ImportError, match="PennyLane is required"):
            plb.PennyLaneBackend()


# ---------------------------------------------------------------------------
# Tests: run()
# ---------------------------------------------------------------------------


class TestRun:
    def test_run_qiskit_circuit(self) -> None:
        from qufin.backends.pennylane_backend import PennyLaneBackend

        backend = PennyLaneBackend(n_qubits=2)
        circ = _FakeQiskitCircuit(n_qubits=2)
        result = backend.run(circ, shots=100)
        assert result.shots == 100
        assert result.backend_id == "pennylane:default.qubit"

    def test_run_callable_probs(self) -> None:
        from qufin.backends.pennylane_backend import PennyLaneBackend

        backend = PennyLaneBackend(n_qubits=1)

        def fake_qnode():
            return np.array([0.25, 0.75])

        result = backend.run(fake_qnode, shots=1000)
        assert result.shots == 1000
        total = sum(result.counts.values())
        assert total == 1000

    def test_run_callable_non_array(self) -> None:
        from qufin.backends.pennylane_backend import PennyLaneBackend

        backend = PennyLaneBackend(n_qubits=1)

        def fake_qnode():
            return 42  # not an ndarray

        result = backend.run(fake_qnode, shots=100)
        assert result.counts == {}

    def test_run_non_callable_non_circuit(self) -> None:
        from qufin.backends.pennylane_backend import PennyLaneBackend

        backend = PennyLaneBackend(n_qubits=2)
        result = backend.run("not a circuit", shots=50)
        assert result.counts == {}


# ---------------------------------------------------------------------------
# Tests: statevector()
# ---------------------------------------------------------------------------


class TestStatevector:
    def test_statevector_qiskit(self) -> None:
        from qufin.backends.pennylane_backend import PennyLaneBackend

        backend = PennyLaneBackend(n_qubits=2)
        circ = _FakeQiskitCircuit(n_qubits=2)
        sv = backend.statevector(circ)
        assert sv.dtype == np.complex128

    def test_statevector_non_qiskit_raises(self) -> None:
        from qufin.backends.pennylane_backend import PennyLaneBackend

        backend = PennyLaneBackend(n_qubits=2)
        with pytest.raises(ValueError, match="Cannot compute statevector"):
            backend.statevector("not a circuit")


# ---------------------------------------------------------------------------
# Tests: gradient / run_with_gradient
# ---------------------------------------------------------------------------


class TestGradient:
    def test_gradient_returns_array(self) -> None:
        from qufin.backends.pennylane_backend import PennyLaneBackend

        backend = PennyLaneBackend(n_qubits=2)
        params = np.array([0.1, 0.2, 0.3])

        def cost(p):
            return 0.5

        grads = backend.gradient(cost, params)
        assert grads.shape == params.shape
        assert grads.dtype == np.float64

    def test_run_with_gradient(self) -> None:
        from qufin.backends.pennylane_backend import GradientResult, PennyLaneBackend

        backend = PennyLaneBackend(n_qubits=2)
        params = np.array([0.5, 1.0])

        def cost(p):
            return 0.42

        result = backend.run_with_gradient(cost, params)
        assert isinstance(result, GradientResult)
        assert result.expectation == pytest.approx(0.42)
        assert result.gradients.shape == params.shape

    def test_make_cost_qnode(self) -> None:
        from qufin.backends.pennylane_backend import PennyLaneBackend

        backend = PennyLaneBackend(n_qubits=2)

        def ansatz(params, wires):
            pass

        ham = MagicMock()
        qnode = backend.make_cost_qnode(ansatz, ham, n_qubits=2)
        assert callable(qnode)

    def test_gradient_result_dataclass(self) -> None:
        from qufin.backends.pennylane_backend import GradientResult

        gr = GradientResult(
            expectation=1.5,
            gradients=np.array([0.1, -0.2]),
        )
        assert gr.expectation == 1.5
        np.testing.assert_array_equal(gr.gradients, [0.1, -0.2])


# ---------------------------------------------------------------------------
# Tests: circuit_stats
# ---------------------------------------------------------------------------


class TestCircuitStats:
    def test_circuit_stats(self) -> None:
        from qufin.backends.pennylane_backend import PennyLaneBackend

        backend = PennyLaneBackend(n_qubits=4)
        circ = _FakeQiskitCircuit(n_qubits=4, depth=7, size=12)
        stats = backend.circuit_stats(circ)
        assert stats == {"n_qubits": 4, "depth": 7, "gate_count": 12}

    def test_circuit_stats_non_qiskit_raises(self) -> None:
        from qufin.backends.pennylane_backend import PennyLaneBackend

        backend = PennyLaneBackend(n_qubits=2)
        with pytest.raises(ValueError, match="circuit_stats requires"):
            backend.circuit_stats("not_a_circuit")


# ---------------------------------------------------------------------------
# Tests: parameter binding
# ---------------------------------------------------------------------------


class TestParameterBinding:
    def test_bind_parameters_none(self) -> None:
        from qufin.backends.pennylane_backend import PennyLaneBackend

        circ = _FakeQiskitCircuit()
        result = PennyLaneBackend._bind_parameters(circ, None)
        assert result is circ

    def test_bind_parameters_empty_params(self) -> None:
        from qufin.backends.pennylane_backend import PennyLaneBackend

        circ = _FakeQiskitCircuit()
        circ.parameters = []
        result = PennyLaneBackend._bind_parameters(circ, {"theta": 1.0})
        assert result is circ

    def test_bind_parameters_with_values(self) -> None:
        from qufin.backends.pennylane_backend import PennyLaneBackend

        circ = _ParameterisedCircuit()
        result = PennyLaneBackend._bind_parameters(
            circ, {"theta": 0.5, "phi": 1.2}
        )
        # assign_parameters returns a new circuit
        assert result is not circ


# ---------------------------------------------------------------------------
# Tests: cross-framework verification
# ---------------------------------------------------------------------------


class TestVerifyAgainstQiskit:
    def test_verify_matching(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Both backends return same probabilities -> match=True."""
        from qufin.backends.pennylane_backend import PennyLaneBackend

        circ = _FakeQiskitCircuit(n_qubits=1)

        # Mock the Qiskit Aer import to use Statevector path
        fake_sv_mod = types.ModuleType("qiskit.quantum_info")

        class FakeSV:
            @staticmethod
            def from_instruction(c):
                return FakeSV()

            def sample_counts(self, shots: int) -> dict[str, int]:
                half = shots // 2
                return {"0": half, "1": shots - half}

        fake_sv_mod.Statevector = FakeSV  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "qiskit.quantum_info", fake_sv_mod)

        # Make pennylane probs return [0.5, 0.5]
        import qufin.backends.pennylane_backend as plb

        mock_qml = plb._qml

        def _fake_qnode(dev, **kwargs):
            def wrapper(fn):
                def wrapped():
                    return np.array([0.5, 0.5])
                return wrapped
            return wrapper

        monkeypatch.setattr(mock_qml, "qnode", _fake_qnode)

        # Patch import of qiskit_aer to fail -> falls back to Statevector
        with patch.dict(sys.modules, {"qiskit_aer": None}):
            result = PennyLaneBackend.verify_against_qiskit(circ, shots=1000)

        assert "match" in result
        assert "max_diff" in result
        assert "qiskit_probs" in result
        assert "pennylane_probs" in result

    def test_verify_keys(self) -> None:
        """Result dict has the expected keys."""
        from qufin.backends.pennylane_backend import PennyLaneBackend

        circ = _FakeQiskitCircuit(n_qubits=1)

        # Patch both backends to avoid real execution
        with (
            patch.object(
                PennyLaneBackend,
                "verify_against_qiskit",
                return_value={
                    "match": True,
                    "qiskit_probs": {"0": 0.5, "1": 0.5},
                    "pennylane_probs": {"0": 0.5, "1": 0.5},
                    "max_diff": 0.0,
                },
            ),
        ):
            result = PennyLaneBackend.verify_against_qiskit(circ)
            assert set(result.keys()) == {
                "match",
                "qiskit_probs",
                "pennylane_probs",
                "max_diff",
            }
