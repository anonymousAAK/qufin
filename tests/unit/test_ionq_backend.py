"""Tests for the IonQ backend.

All braket imports are mocked so tests run without the SDK installed.
"""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Mock the braket SDK before any qufin import that touches it
# ---------------------------------------------------------------------------


def _make_braket_mocks() -> dict[str, ModuleType]:
    """Create a minimal mock tree for amazon-braket-sdk."""
    braket = ModuleType("braket")
    braket_aws = ModuleType("braket.aws")
    braket_devices = ModuleType("braket.devices")
    braket_circuits = ModuleType("braket.circuits")

    mock_device = MagicMock(name="AwsDevice")
    braket_aws.AwsDevice = mock_device  # type: ignore[attr-defined]

    braket_devices.LocalSimulator = MagicMock(name="LocalSimulator")  # type: ignore[attr-defined]

    braket_circuits.Circuit = MagicMock(name="BraketCircuit")  # type: ignore[attr-defined]

    braket.aws = braket_aws  # type: ignore[attr-defined]
    braket.devices = braket_devices  # type: ignore[attr-defined]
    braket.circuits = braket_circuits  # type: ignore[attr-defined]

    return {
        "braket": braket,
        "braket.aws": braket_aws,
        "braket.devices": braket_devices,
        "braket.circuits": braket_circuits,
    }


_braket_mocks = _make_braket_mocks()


@pytest.fixture(autouse=True)
def _patch_braket():
    """Patch braket modules for every test in this file."""
    with patch.dict(sys.modules, _braket_mocks):
        yield


# ---------------------------------------------------------------------------
# Helper to get a fresh import of the module under test
# ---------------------------------------------------------------------------


def _get_module():
    from qufin.backends import ionq_backend
    return ionq_backend


# ===========================================================================
# Test: IonQConfig dataclass
# ===========================================================================


class TestIonQConfig:
    def test_default_values(self) -> None:
        mod = _get_module()
        cfg = mod.IonQConfig()
        assert cfg.device_name == "Aria-1"
        assert cfg.shots == 1024
        assert cfg.api_key is None
        assert cfg.s3_bucket is None
        assert cfg.s3_prefix == "qufin-ionq-results"

    def test_custom_values(self) -> None:
        mod = _get_module()
        cfg = mod.IonQConfig(
            device_name="Forte-1",
            shots=2048,
            api_key="test-key",
            s3_bucket="my-bucket",
        )
        assert cfg.device_name == "Forte-1"
        assert cfg.shots == 2048
        assert cfg.api_key == "test-key"
        assert cfg.s3_bucket == "my-bucket"


# ===========================================================================
# Test: IONQ_DEVICE_ARNS mapping
# ===========================================================================


class TestDeviceArns:
    def test_aria1_in_mapping(self) -> None:
        mod = _get_module()
        assert "Aria-1" in mod.IONQ_DEVICE_ARNS

    def test_forte1_in_mapping(self) -> None:
        mod = _get_module()
        assert "Forte-1" in mod.IONQ_DEVICE_ARNS

    def test_arns_are_strings(self) -> None:
        mod = _get_module()
        for arn in mod.IONQ_DEVICE_ARNS.values():
            assert isinstance(arn, str)
            assert arn.startswith("arn:aws:braket")


# ===========================================================================
# Test: IonQBackend creation
# ===========================================================================


class TestIonQBackendInit:
    def test_default_config(self) -> None:
        mod = _get_module()
        backend = mod.IonQBackend()
        assert backend.backend_id == "ionq:Aria-1"

    def test_custom_config(self) -> None:
        mod = _get_module()
        cfg = mod.IonQConfig(device_name="Forte-1")
        backend = mod.IonQBackend(config=cfg)
        assert backend.backend_id == "ionq:Forte-1"

    def test_unknown_device_raises(self) -> None:
        mod = _get_module()
        cfg = mod.IonQConfig(device_name="Unknown-99")
        with pytest.raises(ValueError, match="Unknown IonQ device"):
            mod.IonQBackend(config=cfg)

    def test_is_not_simulator(self) -> None:
        mod = _get_module()
        backend = mod.IonQBackend()
        assert backend.is_simulator() is False


# ===========================================================================
# Test: IonQBackend.run
# ===========================================================================


class TestIonQBackendRun:
    def _setup_device_mock(self, backend, counts):
        """Helper to set up mock device run results."""
        mock_result = MagicMock()
        mock_result.measurement_counts = counts
        mock_task = MagicMock()
        mock_task.result.return_value = mock_result
        backend._device.run.return_value = mock_task

    def test_run_braket_circuit(self) -> None:
        mod = _get_module()
        backend = mod.IonQBackend()

        # Duck-typed as a Braket circuit (has qubits attr, no num_qubits)
        mock_circuit = MagicMock()
        del mock_circuit.num_qubits
        mock_circuit.qubits = [0, 1]

        self._setup_device_mock(backend, {"00": 500, "11": 524})

        result = backend.run(mock_circuit, shots=1024)
        assert result.shots == 1024
        assert result.backend_id == "ionq:Aria-1"
        assert sum(result.counts.values()) == 1024

    def test_run_unsupported_type_raises(self) -> None:
        mod = _get_module()
        backend = mod.IonQBackend()

        with pytest.raises(TypeError, match="Unsupported circuit type"):
            backend.run("not a circuit", shots=100)

    def test_run_returns_circuit_result(self) -> None:
        mod = _get_module()
        backend = mod.IonQBackend()

        mock_circuit = MagicMock()
        mock_circuit.num_qubits = 2
        backend._convert_qiskit = MagicMock(return_value=MagicMock())

        self._setup_device_mock(backend, {"0": 700, "1": 324})

        result = backend.run(mock_circuit, shots=1024)
        assert isinstance(result.counts, dict)
        assert result.metadata["device"] == "Aria-1"

    def test_run_metadata_contains_arn(self) -> None:
        mod = _get_module()
        backend = mod.IonQBackend()

        mock_circuit = MagicMock()
        mock_circuit.num_qubits = 2
        backend._convert_qiskit = MagicMock(return_value=MagicMock())

        self._setup_device_mock(backend, {"00": 1024})

        result = backend.run(mock_circuit, shots=1024)
        assert "device_arn" in result.metadata


# ===========================================================================
# Test: IonQBackend.statevector
# ===========================================================================


class TestIonQBackendStatevector:
    def test_statevector_raises(self) -> None:
        mod = _get_module()
        backend = mod.IonQBackend()
        with pytest.raises(NotImplementedError, match="not available on IonQ"):
            backend.statevector(MagicMock())


# ===========================================================================
# Test: ImportError when SDK missing
# ===========================================================================


class TestImportError:
    def test_missing_sdk_raises_helpful_error(self) -> None:
        """When braket is not installed, constructor raises ImportError."""
        # Remove braket from sys.modules to simulate missing SDK
        saved = {}
        for key in list(sys.modules.keys()):
            if key.startswith("braket"):
                saved[key] = sys.modules.pop(key)
        # Also remove cached ionq_backend module
        for key in list(sys.modules.keys()):
            if "ionq_backend" in key:
                saved[key] = sys.modules.pop(key)

        try:
            with pytest.raises(ImportError, match="pip install qufin\\[ionq\\]"):
                from qufin.backends.ionq_backend import IonQBackend
                IonQBackend()
        finally:
            sys.modules.update(saved)
