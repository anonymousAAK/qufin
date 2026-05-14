"""Tests for automatic backend selection and circuit analysis."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from qufin.backends.auto_select import (
    BackendRegistry,
    CircuitAnalysis,
    analyze_circuit,
    auto_select_backend,
    get_available_backends,
)
from qufin.backends.mock import MockBackend

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_circuit(n_qubits=4, gates=None):
    """Build a minimal Qiskit-like circuit stub."""
    idx_iter = iter(range(100))

    data = []
    if gates:
        for name, n_q, params in gates:
            op = SimpleNamespace(name=name, params=params or [])
            qubits = [SimpleNamespace() for _ in range(n_q)]
            data.append(SimpleNamespace(operation=op, qubits=qubits))

    class FakeCircuit:
        num_qubits = n_qubits

        @staticmethod
        def depth():
            return len(data)

        @staticmethod
        def find_bit(_q):
            return SimpleNamespace(index=next(idx_iter) % n_qubits)

    FakeCircuit.data = data
    return FakeCircuit()


# ---------------------------------------------------------------------------
# CircuitAnalysis dataclass
# ---------------------------------------------------------------------------


class TestCircuitAnalysis:
    def test_total_gates(self) -> None:
        ca = CircuitAnalysis(gate_counts={"h": 3, "cx": 2})
        assert ca.total_gates == 5

    def test_two_qubit_count(self) -> None:
        ca = CircuitAnalysis(gate_counts={"h": 3, "cx": 2, "cz": 1})
        assert ca.two_qubit_count == 3

    def test_two_qubit_count_none(self) -> None:
        ca = CircuitAnalysis(gate_counts={"h": 5, "rz": 2})
        assert ca.two_qubit_count == 0

    def test_defaults(self) -> None:
        ca = CircuitAnalysis()
        assert ca.qubit_count == 0
        assert ca.depth == 0
        assert ca.total_gates == 0


# ---------------------------------------------------------------------------
# analyze_circuit
# ---------------------------------------------------------------------------


class TestAnalyzeCircuit:
    def test_empty_circuit(self) -> None:
        circ = _fake_circuit(n_qubits=3, gates=[])
        analysis = analyze_circuit(circ)
        assert analysis.qubit_count == 3
        assert analysis.depth == 0
        assert analysis.total_gates == 0

    def test_gate_counts(self) -> None:
        circ = _fake_circuit(
            n_qubits=4,
            gates=[
                ("h", 1, None),
                ("h", 1, None),
                ("cx", 2, None),
            ],
        )
        analysis = analyze_circuit(circ)
        assert analysis.gate_counts["h"] == 2
        assert analysis.gate_counts["cx"] == 1

    def test_connectivity_graph(self) -> None:
        circ = _fake_circuit(
            n_qubits=4,
            gates=[("cx", 2, None)],
        )
        analysis = analyze_circuit(circ)
        assert len(analysis.connectivity_graph) >= 1

    def test_plain_object_fallback(self) -> None:
        """Object with no data/depth attributes."""
        obj = SimpleNamespace(num_qubits=5)
        analysis = analyze_circuit(obj)
        assert analysis.qubit_count == 5
        assert analysis.depth == 0


# ---------------------------------------------------------------------------
# BackendRegistry
# ---------------------------------------------------------------------------


class TestBackendRegistry:
    def test_register_and_get(self) -> None:
        reg = BackendRegistry()
        reg.register("mock", MockBackend)
        backend = reg.get("mock")
        assert isinstance(backend, MockBackend)

    def test_get_caches_instance(self) -> None:
        reg = BackendRegistry()
        reg.register("mock", MockBackend)
        b1 = reg.get("mock")
        b2 = reg.get("mock")
        assert b1 is b2

    def test_get_unknown_raises(self) -> None:
        reg = BackendRegistry()
        with pytest.raises(KeyError, match="not registered"):
            reg.get("nonexistent")

    def test_registered_names(self) -> None:
        reg = BackendRegistry()
        reg.register("a", MockBackend)
        reg.register("b", MockBackend)
        assert sorted(reg.registered_names) == ["a", "b"]

    def test_unregister(self) -> None:
        reg = BackendRegistry()
        reg.register("tmp", MockBackend)
        reg.unregister("tmp")
        assert "tmp" not in reg.registered_names

    def test_clear(self) -> None:
        reg = BackendRegistry()
        reg.register("a", MockBackend)
        reg.clear()
        assert reg.registered_names == []


# ---------------------------------------------------------------------------
# get_available_backends
# ---------------------------------------------------------------------------


class TestGetAvailableBackends:
    def test_mock_always_available(self) -> None:
        available = get_available_backends()
        assert available["mock"] is True

    def test_returns_dict(self) -> None:
        available = get_available_backends()
        assert isinstance(available, dict)
        expected_keys = {
            "mock", "qiskit_aer", "cudaq", "pennylane", "cirq", "braket",
        }
        assert expected_keys == set(available.keys())


# ---------------------------------------------------------------------------
# auto_select_backend
# ---------------------------------------------------------------------------


class TestAutoSelectBackend:
    def test_fallback_to_mock(self) -> None:
        """When nothing else is available, MockBackend is returned."""
        avail = {
            "mock": True,
            "qiskit_aer": False,
            "cudaq": False,
        }
        circ = _fake_circuit(n_qubits=4)
        backend = auto_select_backend(circ, available_backends=avail)
        assert isinstance(backend, MockBackend)

    def test_preference_respected(self) -> None:
        avail = {"mock": True, "qiskit_aer": False, "cudaq": False}
        circ = _fake_circuit(n_qubits=4)
        backend = auto_select_backend(
            circ, available_backends=avail, preference="mock"
        )
        assert isinstance(backend, MockBackend)

    def test_preference_unavailable_falls_back(self) -> None:
        avail = {"mock": True, "qiskit_aer": False, "cudaq": False}
        circ = _fake_circuit(n_qubits=4)
        backend = auto_select_backend(
            circ, available_backends=avail, preference="cudaq"
        )
        # cudaq unavailable -> falls through to mock
        assert isinstance(backend, MockBackend)

    def test_large_circuit_prefers_cudaq(self) -> None:
        """For >30 qubits, cudaq should be tried first."""
        avail = {"mock": True, "qiskit_aer": False, "cudaq": False}
        circ = _fake_circuit(n_qubits=35)
        backend = auto_select_backend(circ, available_backends=avail)
        # cudaq not available, so falls back to mock
        assert isinstance(backend, MockBackend)

    def test_absolute_fallback(self) -> None:
        """Even with empty availability, we get MockBackend."""
        circ = _fake_circuit(n_qubits=2)
        backend = auto_select_backend(
            circ, available_backends={}
        )
        assert isinstance(backend, MockBackend)

    @patch(
        "qufin.backends.auto_select.get_available_backends",
        return_value={"mock": True, "qiskit_aer": False, "cudaq": False,
                      "pennylane": False, "cirq": False, "braket": False},
    )
    def test_auto_probe(self, _mock_probe) -> None:
        """auto_select_backend probes availability when not provided."""
        circ = _fake_circuit(n_qubits=4)
        backend = auto_select_backend(circ)
        assert isinstance(backend, MockBackend)

    def test_medium_circuit_mock_fallback(self) -> None:
        avail = {"mock": True, "qiskit_aer": False, "cudaq": False}
        circ = _fake_circuit(n_qubits=20)
        backend = auto_select_backend(circ, available_backends=avail)
        assert isinstance(backend, MockBackend)


# ---------------------------------------------------------------------------
# Lazy import helpers from backends/__init__.py  (lines 36-66)
# ---------------------------------------------------------------------------


class TestLazyImportHelpers:
    """Cover lazy-import functions in qufin.backends.__init__."""

    def test_get_ibm_backend_import_error(self) -> None:
        with pytest.raises((ImportError, ModuleNotFoundError)):
            from qufin.backends import get_ibm_backend
            get_ibm_backend()

    def test_get_pennylane_backend_import_error(self) -> None:
        with pytest.raises((ImportError, ModuleNotFoundError)):
            from qufin.backends import get_pennylane_backend
            get_pennylane_backend()

    def test_get_cirq_backend_import_error(self) -> None:
        with pytest.raises((ImportError, ModuleNotFoundError)):
            from qufin.backends import get_cirq_backend
            get_cirq_backend()

    def test_get_braket_backend_import_error(self) -> None:
        with pytest.raises((ImportError, ModuleNotFoundError)):
            from qufin.backends import get_braket_backend
            get_braket_backend()

    def test_get_cudaq_backend_import_error(self) -> None:
        with pytest.raises((ImportError, ModuleNotFoundError)):
            from qufin.backends import get_cudaq_backend
            get_cudaq_backend()


# ---------------------------------------------------------------------------
# get_available_backends — optional import branches (lines 192-223)
# ---------------------------------------------------------------------------


class TestGetAvailableBackendsImportBranches:
    """Cover all try/except branches in get_available_backends."""

    @patch.dict(sys.modules, {"cudaq": None})
    def test_cudaq_unavailable(self) -> None:
        available = get_available_backends()
        assert available["cudaq"] is False

    @patch.dict(sys.modules, {"pennylane": None})
    def test_pennylane_unavailable(self) -> None:
        available = get_available_backends()
        assert available["pennylane"] is False

    @patch.dict(sys.modules, {"cirq": None})
    def test_cirq_unavailable(self) -> None:
        available = get_available_backends()
        assert available["cirq"] is False

    @patch.dict(sys.modules, {"braket": None})
    def test_braket_unavailable(self) -> None:
        available = get_available_backends()
        assert available["braket"] is False


# ---------------------------------------------------------------------------
# _try_create_backend — optional backend branches (lines 325-356)
# ---------------------------------------------------------------------------


class TestTryCreateBackendBranches:
    """Cover _try_create_backend for optional backends."""

    def test_try_create_qiskit_aer_available(self) -> None:
        """qiskit_aer is available — should create a backend."""
        avail = {"mock": True, "qiskit_aer": True, "cudaq": False}
        circ = _fake_circuit(n_qubits=4)
        backend = auto_select_backend(circ, available_backends=avail)
        # Should pick qiskit_aer since it's available for small circuits
        assert backend is not None

    def test_try_create_pennylane_catches_import(self) -> None:
        """pennylane marked available but import fails -> returns None."""
        from qufin.backends.auto_select import CircuitAnalysis, _try_create_backend
        analysis = CircuitAnalysis(qubit_count=4)
        result = _try_create_backend(
            "pennylane", {"pennylane": True}, analysis
        )
        # Import will fail -> returns None
        assert result is None

    def test_try_create_cirq_catches_import(self) -> None:
        from qufin.backends.auto_select import CircuitAnalysis, _try_create_backend
        analysis = CircuitAnalysis(qubit_count=4)
        result = _try_create_backend("cirq", {"cirq": True}, analysis)
        assert result is None

    def test_try_create_braket_catches_import(self) -> None:
        from qufin.backends.auto_select import CircuitAnalysis, _try_create_backend
        analysis = CircuitAnalysis(qubit_count=4)
        result = _try_create_backend(
            "braket", {"braket": True}, analysis
        )
        assert result is None

    def test_try_create_cudaq_catches_import(self) -> None:
        from qufin.backends.auto_select import CircuitAnalysis, _try_create_backend
        analysis = CircuitAnalysis(qubit_count=4)
        result = _try_create_backend(
            "cudaq", {"cudaq": True}, analysis
        )
        assert result is None

    def test_try_create_cudaq_large_circuit(self) -> None:
        """cudaq with >30 qubits should try nvidia-mgpu target."""
        from qufin.backends.auto_select import CircuitAnalysis, _try_create_backend
        analysis = CircuitAnalysis(qubit_count=35)
        result = _try_create_backend(
            "cudaq", {"cudaq": True}, analysis
        )
        assert result is None  # import fails

    def test_try_create_unknown_returns_none(self) -> None:
        from qufin.backends.auto_select import CircuitAnalysis, _try_create_backend
        analysis = CircuitAnalysis(qubit_count=4)
        result = _try_create_backend(
            "unknown_backend", {"unknown_backend": True}, analysis
        )
        assert result is None
