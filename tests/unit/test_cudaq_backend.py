"""Tests for CUDA-Q backend (all cudaq imports mocked)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_cudaq():
    """Build a fake ``cudaq`` module with the minimum API surface."""
    mock = MagicMock()
    mock.num_available_gpus.return_value = 2

    target_a = SimpleNamespace(name="nvidia")
    target_b = SimpleNamespace(name="nvidia-mgpu")
    mock.get_targets.return_value = [target_a, target_b]

    # make_kernel returns a kernel stub
    kernel = MagicMock()
    kernel.qalloc.return_value = MagicMock()
    mock.make_kernel.return_value = kernel

    # sample returns a dict-like result
    sample_result = MagicMock()
    sample_result.__iter__ = MagicMock(return_value=iter(["00", "11"]))
    sample_result.count.side_effect = lambda bs: {"00": 520, "11": 504}.get(
        bs, 0
    )
    mock.sample.return_value = sample_result

    # get_state returns a flat array
    mock.get_state.return_value = np.array(
        [1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)], dtype=np.complex128
    )

    return mock


def _fake_circuit(n_qubits=2):
    """Minimal Qiskit-like circuit stub."""
    bit_info = SimpleNamespace(index=0)

    class FakeCircuit:
        num_qubits = n_qubits
        data = []

        @staticmethod
        def find_bit(_q):
            return bit_info

        @staticmethod
        def depth():
            return 0

    return FakeCircuit()


def _fake_circuit_with_gates():
    """Stub circuit with a few representative gates."""
    idx = iter(range(10))

    def _make_inst(name, n_qubits, params=None):
        op = SimpleNamespace(name=name, params=params or [])
        qubits = [SimpleNamespace() for _ in range(n_qubits)]
        return SimpleNamespace(operation=op, qubits=qubits)

    h_inst = _make_inst("h", 1)
    cx_inst = _make_inst("cx", 2)
    rx_inst = _make_inst("rx", 1, params=[1.57])
    barrier_inst = _make_inst("barrier", 1)

    class FakeCircuit:
        num_qubits = 3
        data = [h_inst, cx_inst, rx_inst, barrier_inst]

        @staticmethod
        def find_bit(_q):
            return SimpleNamespace(index=next(idx) % 3)

        @staticmethod
        def depth():
            return 4

    return FakeCircuit()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCudaQBackendImportGuard:
    """Tests that run without cudaq installed."""

    def test_cudaq_available_flag_false(self) -> None:
        from qufin.backends.cudaq_backend import CUDAQ_AVAILABLE

        # In CI cudaq won't be installed
        assert isinstance(CUDAQ_AVAILABLE, bool)

    def test_require_cudaq_raises(self) -> None:
        from qufin.backends.cudaq_backend import _require_cudaq

        with (
            patch("qufin.backends.cudaq_backend.CUDAQ_AVAILABLE", False),
            pytest.raises(ImportError, match="CUDA-Q is required"),
        ):
            _require_cudaq()

    def test_constructor_raises_without_cudaq(self) -> None:
        with patch("qufin.backends.cudaq_backend.CUDAQ_AVAILABLE", False):
            from qufin.backends.cudaq_backend import CudaQBackend

            with pytest.raises(ImportError):
                CudaQBackend()


class TestCudaQBackendMocked:
    """Tests with a fully mocked cudaq module."""

    @pytest.fixture(autouse=True)
    def _patch_cudaq(self):
        self.mock_cudaq = _make_mock_cudaq()
        with (
            patch("qufin.backends.cudaq_backend.cudaq", self.mock_cudaq),
            patch("qufin.backends.cudaq_backend.CUDAQ_AVAILABLE", True),
        ):
            yield

    def _make_backend(self, **kwargs):
        from qufin.backends.cudaq_backend import CudaQBackend

        return CudaQBackend(**kwargs)

    # -- basic properties --

    def test_backend_id(self) -> None:
        backend = self._make_backend(target="nvidia")
        assert backend.backend_id == "cudaq-nvidia"

    def test_is_simulator(self) -> None:
        backend = self._make_backend()
        assert backend.is_simulator()

    def test_set_target_called(self) -> None:
        self._make_backend(target="nvidia-mgpu")
        self.mock_cudaq.set_target.assert_called_with("nvidia-mgpu")

    def test_seed_set(self) -> None:
        self._make_backend(seed=99)
        self.mock_cudaq.set_random_seed.assert_called_with(99)

    def test_seed_none_skips(self) -> None:
        self._make_backend(seed=None)
        self.mock_cudaq.set_random_seed.assert_not_called()

    # -- run --

    def test_run_returns_circuit_result(self) -> None:
        backend = self._make_backend()
        result = backend.run(_fake_circuit(), shots=1024)
        assert result.shots == 1024
        assert result.backend_id == "cudaq-nvidia"
        assert "00" in result.counts

    def test_run_calls_sample(self) -> None:
        backend = self._make_backend()
        backend.run(_fake_circuit(), shots=512)
        self.mock_cudaq.sample.assert_called_once()

    def test_run_auto_mgpu_for_large_circuits(self) -> None:
        backend = self._make_backend(
            target="nvidia", multi_gpu_threshold=5
        )
        backend.run(_fake_circuit(n_qubits=10), shots=100)
        self.mock_cudaq.set_target.assert_called_with("nvidia-mgpu")

    # -- statevector --

    def test_statevector_shape(self) -> None:
        backend = self._make_backend()
        sv = backend.statevector(_fake_circuit())
        assert sv.dtype == np.complex128
        assert len(sv) == 4

    def test_statevector_calls_get_state(self) -> None:
        backend = self._make_backend()
        backend.statevector(_fake_circuit())
        self.mock_cudaq.get_state.assert_called_once()

    # -- benchmark --

    def test_benchmark_simulation(self) -> None:
        backend = self._make_backend()
        result = backend.benchmark_simulation(n_qubits=4, depth=3)
        assert result["n_qubits"] == 4
        assert result["depth"] == 3
        assert "elapsed_seconds" in result
        assert result["elapsed_seconds"] >= 0

    # -- GPU info --

    def test_get_gpu_info(self) -> None:
        from qufin.backends.cudaq_backend import CudaQBackend

        info = CudaQBackend.get_gpu_info()
        assert info["available"] is True
        assert info["num_gpus"] == 2
        assert "nvidia" in info["targets"]

    # -- translation --

    def test_translate_with_gates(self) -> None:
        backend = self._make_backend()
        circ = _fake_circuit_with_gates()
        _kernel, n = backend._translate_circuit(circ)
        assert n == 3

    def test_translate_unsupported_gate(self) -> None:
        backend = self._make_backend()
        inst = SimpleNamespace(
            operation=SimpleNamespace(name="foogate", params=[]),
            qubits=[SimpleNamespace()],
        )

        class BadCircuit:
            num_qubits = 1
            data = [inst]

            @staticmethod
            def find_bit(_q):
                return SimpleNamespace(index=0)

        with pytest.raises(ValueError, match="Unsupported gate"):
            backend._translate_circuit(BadCircuit())


class TestGetGpuInfoUnavailable:
    def test_returns_unavailable(self) -> None:
        with patch("qufin.backends.cudaq_backend.CUDAQ_AVAILABLE", False):
            from qufin.backends.cudaq_backend import CudaQBackend

            info = CudaQBackend.get_gpu_info()
            assert info["available"] is False
            assert info["num_gpus"] == 0
