"""Tests for the Cirq backend (v0.3.0 enhancements).

All cirq imports are mocked so tests run without cirq installed.
"""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers: build a mock cirq module
# ---------------------------------------------------------------------------

def _make_mock_cirq() -> ModuleType:
    """Return a fake ``cirq`` module with just enough API surface."""
    cirq = ModuleType("cirq")

    # Simulator mocks
    cirq.Simulator = MagicMock  # type: ignore[attr-defined]
    cirq.DensityMatrixSimulator = MagicMock  # type: ignore[attr-defined]

    # Gate factories
    cirq.X = MagicMock()  # type: ignore[attr-defined]
    cirq.Y = MagicMock()  # type: ignore[attr-defined]
    cirq.Z = MagicMock()  # type: ignore[attr-defined]
    cirq.CNOT = MagicMock(return_value=MagicMock())  # type: ignore[attr-defined]
    cirq.CZ = MagicMock(return_value=MagicMock())  # type: ignore[attr-defined]

    def _mock_rz(angle: float) -> MagicMock:
        gate = MagicMock()
        gate.on = MagicMock(return_value=MagicMock())
        return gate

    cirq.rz = _mock_rz  # type: ignore[attr-defined]

    # LineQubit
    class _LQ:
        def __init__(self, i: int) -> None:
            self.i = i

        @staticmethod
        def range(n: int) -> list[Any]:
            return [_LQ(i) for i in range(n)]

    cirq.LineQubit = _LQ  # type: ignore[attr-defined]

    # Circuit
    class _Circuit:
        def __init__(self, *_args: Any, **_kw: Any) -> None:
            self._ops: list[Any] = []

        def append(self, ops: Any) -> None:
            self._ops.append(ops)

        def __add__(self, other: Any) -> _Circuit:
            c = _Circuit()
            c._ops = [*self._ops, other]
            return c

        def all_operations(self) -> list[Any]:
            return self._ops

        def all_qubits(self) -> set[Any]:
            return set()

    cirq.Circuit = _Circuit  # type: ignore[attr-defined]

    # measure
    cirq.measure = MagicMock(return_value=MagicMock())  # type: ignore[attr-defined]

    # Noise model helpers
    cirq.ConstantQubitNoiseModel = MagicMock  # type: ignore[attr-defined]

    class _DepolarizingChannel:
        def __init__(self, p: float) -> None:
            self.p = p

    cirq.DepolarizingChannel = _DepolarizingChannel  # type: ignore[attr-defined]
    cirq.depolarize = MagicMock(  # type: ignore[attr-defined]
        side_effect=_DepolarizingChannel
    )

    # SqrtIswapTargetGateset / optimize_for_target_gateset
    cirq.SqrtIswapTargetGateset = MagicMock  # type: ignore[attr-defined]
    cirq.optimize_for_target_gateset = MagicMock(  # type: ignore[attr-defined]
        return_value=_Circuit()
    )
    cirq.decompose = MagicMock(return_value=[])  # type: ignore[attr-defined]

    # contrib.qasm_import
    contrib = ModuleType("cirq.contrib")
    qasm_import = ModuleType("cirq.contrib.qasm_import")
    qasm_import.circuit_from_qasm = MagicMock(  # type: ignore[attr-defined]
        return_value=_Circuit()
    )
    contrib.qasm_import = qasm_import  # type: ignore[attr-defined]
    cirq.contrib = contrib  # type: ignore[attr-defined]

    # transformers (fallback path in decompose_for_sycamore)
    transformers = ModuleType("cirq.transformers")
    transformers.optimize_for_target_gateset = MagicMock(  # type: ignore[attr-defined]
        return_value=_Circuit()
    )
    cirq.transformers = transformers  # type: ignore[attr-defined]

    return cirq


@pytest.fixture(autouse=True)
def _inject_mock_cirq(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject a mock cirq module into sys.modules before every test."""
    mock_cirq = _make_mock_cirq()
    monkeypatch.setitem(sys.modules, "cirq", mock_cirq)
    monkeypatch.setitem(sys.modules, "cirq.contrib", mock_cirq.contrib)
    monkeypatch.setitem(
        sys.modules, "cirq.contrib.qasm_import", mock_cirq.contrib.qasm_import
    )
    monkeypatch.setitem(
        sys.modules, "cirq.transformers", mock_cirq.transformers
    )


# ---------------------------------------------------------------------------
# Import under test (must come AFTER mock injection at module level is fine
# because the fixture is autouse)
# ---------------------------------------------------------------------------

def _get_backend_module():
    """Import the cirq_backend module lazily."""
    from qufin.backends import cirq_backend
    return cirq_backend


# ===========================================================================
# Test: GoogleHardwareConfig dataclass
# ===========================================================================

class TestGoogleHardwareConfig:
    def test_default_values(self) -> None:
        mod = _get_backend_module()
        cfg = mod.GoogleHardwareConfig()
        assert cfg.processor_id == "sycamore"
        assert cfg.max_qubits == 53
        assert cfg.region == "us-central1"

    def test_custom_values(self) -> None:
        mod = _get_backend_module()
        cfg = mod.GoogleHardwareConfig(
            processor_id="willow", max_qubits=105, gate_set="google_v2"
        )
        assert cfg.processor_id == "willow"
        assert cfg.max_qubits == 105
        assert cfg.gate_set == "google_v2"

    def test_predefined_sycamore(self) -> None:
        mod = _get_backend_module()
        cfg = mod.SYCAMORE_CONFIG
        assert cfg.processor_id == "sycamore"
        assert cfg.gate_set == "sqrt_iswap"

    def test_predefined_willow(self) -> None:
        mod = _get_backend_module()
        cfg = mod.WILLOW_CONFIG
        assert cfg.processor_id == "willow"
        assert cfg.max_qubits == 105


# ===========================================================================
# Test: CirqBackend basics
# ===========================================================================

class TestCirqBackendBasics:
    def test_backend_id_default(self) -> None:
        mod = _get_backend_module()
        backend = mod.CirqBackend()
        assert backend.backend_id == "cirq:simulator"

    def test_backend_id_with_hardware(self) -> None:
        mod = _get_backend_module()
        cfg = mod.GoogleHardwareConfig(processor_id="willow")
        backend = mod.CirqBackend(hardware_config=cfg)
        assert backend.backend_id == "cirq:willow"

    def test_is_simulator_true(self) -> None:
        mod = _get_backend_module()
        backend = mod.CirqBackend()
        assert backend.is_simulator() is True

    def test_is_simulator_false_with_hardware(self) -> None:
        mod = _get_backend_module()
        cfg = mod.GoogleHardwareConfig(processor_id="sycamore")
        backend = mod.CirqBackend(hardware_config=cfg)
        assert backend.is_simulator() is False

    def test_noise_char_initially_none(self) -> None:
        mod = _get_backend_module()
        backend = mod.CirqBackend()
        assert backend.noise_characterization is None


# ===========================================================================
# Test: get_hardware_info
# ===========================================================================

class TestGetHardwareInfo:
    def test_raises_without_config(self) -> None:
        mod = _get_backend_module()
        backend = mod.CirqBackend()
        with pytest.raises(RuntimeError, match="No hardware config"):
            backend.get_hardware_info()

    def test_returns_dict_with_config(self) -> None:
        mod = _get_backend_module()
        cfg = mod.GoogleHardwareConfig(
            processor_id="sycamore",
            project_id="my-project",
        )
        backend = mod.CirqBackend(hardware_config=cfg)
        info = backend.get_hardware_info()
        assert info["processor_id"] == "sycamore"
        assert info["max_qubits"] == 53
        assert "Quantum Engine" in info["access_path"]
        assert info["project_id"] == "my-project"


# ===========================================================================
# Test: with_device_noise factory
# ===========================================================================

class TestWithDeviceNoise:
    def test_factory_creates_noisy_backend(self) -> None:
        mod = _get_backend_module()
        backend = mod.CirqBackend.with_device_noise(
            processor_id="sycamore",
            depolarize_rate=0.01,
        )
        assert "noisy" in backend.backend_id
        assert backend._depolarize_rate == 0.01

    def test_factory_unknown_processor(self) -> None:
        mod = _get_backend_module()
        backend = mod.CirqBackend.with_device_noise(
            processor_id="unknown_chip",
        )
        assert "noisy-unknown_chip" in backend.backend_id


# ===========================================================================
# Test: XEB fidelity
# ===========================================================================

class TestXebFidelity:
    def test_xeb_returns_float(self) -> None:
        mod = _get_backend_module()
        backend = mod.CirqBackend()

        # Mock the simulator to return plausible results
        sim_result = MagicMock()
        sim_result.final_state_vector = np.array(
            [0.5, 0.5, 0.5, 0.5], dtype=np.complex128
        )
        # Ensure every Simulator() instance returns same simulate result
        mock_sim = MagicMock()
        mock_sim.simulate = MagicMock(return_value=sim_result)
        backend._cirq.Simulator = MagicMock(return_value=mock_sim)

        meas_bits = np.array([[0, 0]] * 500 + [[1, 1]] * 500)
        run_result = MagicMock()
        run_result.measurements = {"m": meas_bits}
        backend._simulator.run = MagicMock(return_value=run_result)

        fid = backend.xeb_fidelity(
            num_circuits=2, cycle_depths=[2]
        )
        assert isinstance(fid, float)
        assert 0.0 <= fid <= 1.0

    def test_xeb_stores_characterization(self) -> None:
        mod = _get_backend_module()
        backend = mod.CirqBackend()

        sim_result = MagicMock()
        sim_result.final_state_vector = np.array(
            [1.0, 0.0, 0.0, 0.0], dtype=np.complex128
        )
        mock_sim = MagicMock()
        mock_sim.simulate = MagicMock(return_value=sim_result)
        backend._cirq.Simulator = MagicMock(return_value=mock_sim)

        meas_bits = np.array([[0, 0]] * 1000)
        run_result = MagicMock()
        run_result.measurements = {"m": meas_bits}
        backend._simulator.run = MagicMock(return_value=run_result)

        backend.xeb_fidelity(num_circuits=1, cycle_depths=[5])
        char = backend.noise_characterization
        assert char is not None
        assert char.num_circuits == 1
        assert 5 in char.cycle_depths


# ===========================================================================
# Test: characterize_noise
# ===========================================================================

class TestCharacterizeNoise:
    def test_returns_noise_char(self) -> None:
        mod = _get_backend_module()
        backend = mod.CirqBackend()

        sim_result = MagicMock()
        sim_result.final_state_vector = np.array(
            [0.5, 0.5, 0.5, 0.5], dtype=np.complex128
        )
        mock_sim = MagicMock()
        mock_sim.simulate = MagicMock(return_value=sim_result)
        backend._cirq.Simulator = MagicMock(return_value=mock_sim)

        meas_bits = np.array([[0, 0]] * 250 + [[0, 1]] * 250 +
                             [[1, 0]] * 250 + [[1, 1]] * 250)
        run_result = MagicMock()
        run_result.measurements = {"m": meas_bits}
        backend._simulator.run = MagicMock(return_value=run_result)

        char = backend.characterize_noise(
            n_qubits=2, num_circuits=2, cycle_depths=[2, 5]
        )
        assert isinstance(char, mod.NoiseCharacterization)
        assert char.single_qubit_error >= 0.0
        assert char.two_qubit_error >= 0.0


# ===========================================================================
# Test: ZZ interaction builder
# ===========================================================================

class TestBuildZZInteraction:
    def test_creates_circuit(self) -> None:
        mod = _get_backend_module()
        backend = mod.CirqBackend()
        circuit = backend.build_zz_interaction(
            qubit_pairs=[(0, 1), (1, 2)],
            angles=[0.5, 0.3],
        )
        # Should have appended operations (3 per pair: CNOT, Rz, CNOT)
        assert len(circuit._ops) == 6

    def test_single_pair(self) -> None:
        mod = _get_backend_module()
        backend = mod.CirqBackend()
        circuit = backend.build_zz_interaction(
            qubit_pairs=[(0, 1)],
            angles=[np.pi / 4],
        )
        assert len(circuit._ops) == 3


# ===========================================================================
# Test: decompose_for_sycamore
# ===========================================================================

class TestDecomposeForSycamore:
    def test_calls_optimize(self) -> None:
        mod = _get_backend_module()
        backend = mod.CirqBackend()
        import cirq
        input_circuit = cirq.Circuit()
        result = backend.decompose_for_sycamore(input_circuit)
        assert result is not None


# ===========================================================================
# Test: NoiseCharacterization dataclass
# ===========================================================================

class TestNoiseCharacterization:
    def test_defaults(self) -> None:
        mod = _get_backend_module()
        nc = mod.NoiseCharacterization()
        assert nc.xeb_fidelity == 0.0
        assert nc.num_circuits == 0
        assert nc.raw_fidelities == {}

    def test_custom(self) -> None:
        mod = _get_backend_module()
        nc = mod.NoiseCharacterization(
            xeb_fidelity=0.95,
            single_qubit_error=0.001,
            two_qubit_error=0.01,
            num_circuits=50,
        )
        assert nc.xeb_fidelity == 0.95
        assert nc.two_qubit_error == 0.01


# ===========================================================================
# Test: PROCESSOR_REGISTRY
# ===========================================================================

class TestProcessorRegistry:
    def test_contains_sycamore(self) -> None:
        mod = _get_backend_module()
        assert "sycamore" in mod.PROCESSOR_REGISTRY

    def test_contains_willow(self) -> None:
        mod = _get_backend_module()
        assert "willow" in mod.PROCESSOR_REGISTRY
