"""Tests for the Quantinuum (H-Series) backend.

All pytket imports are mocked so tests run without the SDK installed.
"""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Mock the pytket SDK before any qufin import that touches it
# ---------------------------------------------------------------------------


def _make_pytket_mocks() -> dict[str, ModuleType]:
    """Create a minimal mock tree for pytket + pytket-quantinuum."""
    pytket = ModuleType("pytket")
    pytket_extensions = ModuleType("pytket.extensions")
    pytket_extensions_quantinuum = ModuleType("pytket.extensions.quantinuum")
    pytket_qasm = ModuleType("pytket.qasm")

    # Mock the QuantinuumBackend class from pytket
    mock_qt_backend_cls = MagicMock(name="QuantinuumBackend")
    mock_qt_backend_inst = MagicMock(name="QuantinuumBackendInstance")
    mock_qt_backend_cls.return_value = mock_qt_backend_inst
    pytket_extensions_quantinuum.QuantinuumBackend = mock_qt_backend_cls  # type: ignore[attr-defined]

    # Mock circuit_from_qasm_str
    pytket_qasm.circuit_from_qasm_str = MagicMock(  # type: ignore[attr-defined]
        name="circuit_from_qasm_str",
        return_value=MagicMock(name="TketCircuit"),
    )

    pytket.extensions = pytket_extensions  # type: ignore[attr-defined]
    pytket_extensions.quantinuum = pytket_extensions_quantinuum  # type: ignore[attr-defined]
    pytket.qasm = pytket_qasm  # type: ignore[attr-defined]

    return {
        "pytket": pytket,
        "pytket.extensions": pytket_extensions,
        "pytket.extensions.quantinuum": pytket_extensions_quantinuum,
        "pytket.qasm": pytket_qasm,
    }


_pytket_mocks = _make_pytket_mocks()


@pytest.fixture(autouse=True)
def _patch_pytket():
    """Patch pytket modules for every test in this file."""
    with patch.dict(sys.modules, _pytket_mocks):
        yield


# ---------------------------------------------------------------------------
# Helper to get a fresh import of the module under test
# ---------------------------------------------------------------------------


def _get_module():
    from qufin.backends import quantinuum_backend
    return quantinuum_backend


# ===========================================================================
# Test: QuantinuumConfig dataclass
# ===========================================================================


class TestQuantinuumConfig:
    def test_default_values(self) -> None:
        mod = _get_module()
        cfg = mod.QuantinuumConfig()
        assert cfg.device_name == "H1-1"
        assert cfg.shots == 1024

    def test_custom_values(self) -> None:
        mod = _get_module()
        cfg = mod.QuantinuumConfig(device_name="H2-1", shots=4096)
        assert cfg.device_name == "H2-1"
        assert cfg.shots == 4096


# ===========================================================================
# Test: QUANTINUUM_DEVICES mapping
# ===========================================================================


class TestDeviceMapping:
    def test_h1_in_mapping(self) -> None:
        mod = _get_module()
        assert "H1-1" in mod.QUANTINUUM_DEVICES
        assert mod.QUANTINUUM_DEVICES["H1-1"] == 20

    def test_h2_in_mapping(self) -> None:
        mod = _get_module()
        assert "H2-1" in mod.QUANTINUUM_DEVICES
        assert mod.QUANTINUUM_DEVICES["H2-1"] == 56

    def test_emulators_in_mapping(self) -> None:
        mod = _get_module()
        assert "H1-1E" in mod.QUANTINUUM_DEVICES
        assert "H2-1E" in mod.QUANTINUUM_DEVICES

    def test_emulator_same_qubits_as_hardware(self) -> None:
        mod = _get_module()
        assert mod.QUANTINUUM_DEVICES["H1-1"] == mod.QUANTINUUM_DEVICES["H1-1E"]
        assert mod.QUANTINUUM_DEVICES["H2-1"] == mod.QUANTINUUM_DEVICES["H2-1E"]


# ===========================================================================
# Test: QuantinuumBackend creation
# ===========================================================================


class TestQuantinuumBackendInit:
    def test_default_config(self) -> None:
        mod = _get_module()
        backend = mod.QuantinuumBackend()
        assert backend.backend_id == "quantinuum:H1-1"

    def test_custom_config(self) -> None:
        mod = _get_module()
        cfg = mod.QuantinuumConfig(device_name="H2-1")
        backend = mod.QuantinuumBackend(config=cfg)
        assert backend.backend_id == "quantinuum:H2-1"

    def test_unknown_device_raises(self) -> None:
        mod = _get_module()
        cfg = mod.QuantinuumConfig(device_name="H99-1")
        with pytest.raises(ValueError, match="Unknown Quantinuum device"):
            mod.QuantinuumBackend(config=cfg)

    def test_is_simulator_hardware(self) -> None:
        mod = _get_module()
        backend = mod.QuantinuumBackend()
        assert backend.is_simulator() is False

    def test_is_simulator_emulator(self) -> None:
        mod = _get_module()
        cfg = mod.QuantinuumConfig(device_name="H1-1E")
        backend = mod.QuantinuumBackend(config=cfg)
        assert backend.is_simulator() is True


# ===========================================================================
# Test: QuantinuumBackend.run
# ===========================================================================


class TestQuantinuumBackendRun:
    def _make_backend_with_mocked_run(self):
        """Create a backend with mocked tket_backend methods."""
        mod = _get_module()
        backend = mod.QuantinuumBackend()

        compiled = MagicMock(name="compiled_circuit")
        backend._tket_backend.get_compiled_circuit.return_value = compiled

        handle = MagicMock(name="result_handle")
        backend._tket_backend.process_circuit.return_value = handle

        mock_result = MagicMock(name="result")
        # get_counts returns dict of tuple->int
        mock_result.get_counts.return_value = {
            (0, 0): 500,
            (1, 1): 524,
        }
        backend._tket_backend.get_result.return_value = mock_result

        return backend

    def test_run_with_tket_circuit(self) -> None:
        backend = self._make_backend_with_mocked_run()
        mock_circuit = MagicMock()
        # No num_qubits attribute => treated as tket circuit
        del mock_circuit.num_qubits

        result = backend.run(mock_circuit, shots=1024)
        assert result.shots == 1024
        assert result.backend_id == "quantinuum:H1-1"
        assert "00" in result.counts
        assert "11" in result.counts

    def test_run_with_qiskit_circuit(self) -> None:
        backend = self._make_backend_with_mocked_run()
        mock_circuit = MagicMock()
        mock_circuit.num_qubits = 2
        # Patch _convert_qiskit
        backend._convert_qiskit = MagicMock(return_value=MagicMock())

        result = backend.run(mock_circuit, shots=1024)
        backend._convert_qiskit.assert_called_once_with(mock_circuit)
        assert result.shots == 1024

    def test_run_counts_are_strings(self) -> None:
        backend = self._make_backend_with_mocked_run()
        mock_circuit = MagicMock()
        del mock_circuit.num_qubits

        result = backend.run(mock_circuit, shots=1024)
        for key in result.counts:
            assert isinstance(key, str)
            assert all(c in "01" for c in key)

    def test_run_metadata_contains_device(self) -> None:
        backend = self._make_backend_with_mocked_run()
        mock_circuit = MagicMock()
        del mock_circuit.num_qubits

        result = backend.run(mock_circuit, shots=1024)
        assert result.metadata["device"] == "H1-1"
        assert result.metadata["max_qubits"] == 20


# ===========================================================================
# Test: QuantinuumBackend.statevector
# ===========================================================================


class TestQuantinuumBackendStatevector:
    def test_statevector_raises(self) -> None:
        mod = _get_module()
        backend = mod.QuantinuumBackend()
        with pytest.raises(NotImplementedError, match="not available on Quantinuum"):
            backend.statevector(MagicMock())


# ===========================================================================
# Test: ImportError when SDK missing
# ===========================================================================


class TestImportError:
    def test_missing_sdk_raises_helpful_error(self) -> None:
        """When pytket is not installed, constructor raises ImportError."""
        saved = {}
        for key in list(sys.modules.keys()):
            if key.startswith("pytket"):
                saved[key] = sys.modules.pop(key)
        for key in list(sys.modules.keys()):
            if "quantinuum_backend" in key:
                saved[key] = sys.modules.pop(key)

        try:
            with pytest.raises(ImportError, match="pip install qufin\\[quantinuum\\]"):
                from qufin.backends.quantinuum_backend import QuantinuumBackend
                QuantinuumBackend()
        finally:
            sys.modules.update(saved)
