"""Tests for the enhanced Braket backend (targets, cost, hybrid jobs).

All braket SDK imports are mocked since the SDK is not installed.
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

    # LocalSimulator mock
    braket_devices.LocalSimulator = MagicMock(name="LocalSimulator")

    # AwsDevice mock
    braket_aws.AwsDevice = MagicMock(name="AwsDevice")

    # AwsQuantumJob mock
    braket_aws.AwsQuantumJob = MagicMock(name="AwsQuantumJob")

    # Circuit mock
    braket_circuits.Circuit = MagicMock(name="BraketCircuit")

    braket.aws = braket_aws
    braket.devices = braket_devices
    braket.circuits = braket_circuits

    return {
        "braket": braket,
        "braket.aws": braket_aws,
        "braket.devices": braket_devices,
        "braket.circuits": braket_circuits,
    }


_braket_mocks = _make_braket_mocks()


@pytest.fixture(autouse=True)
def _patch_braket():
    """Patch braket modules for every test."""
    with patch.dict(sys.modules, _braket_mocks):
        yield


# Now import the module under test (deferred so fixture is active at collection)
# We re-import inside helpers to ensure the mock is in place.


def _import_module():
    """Import braket_backend under mocked braket SDK."""
    import importlib

    # Force re-import so the mocked braket modules are picked up
    mod_name = "qufin.backends.braket_backend"
    if mod_name in sys.modules:
        return importlib.reload(sys.modules[mod_name])
    return importlib.import_module(mod_name)


# ===================================================================
# Target dataclass tests
# ===================================================================


class TestIonQTarget:
    def test_aria_arn(self) -> None:
        mod = _import_module()
        assert "ionq" in mod.IONQ_ARIA.arn.lower()

    def test_forte_arn(self) -> None:
        mod = _import_module()
        assert "Forte" in mod.IONQ_FORTE.arn

    def test_all_to_all_topology(self) -> None:
        mod = _import_module()
        assert mod.IONQ_ARIA.topology == mod.Topology.ALL_TO_ALL
        assert mod.IONQ_FORTE.topology == mod.Topology.ALL_TO_ALL

    def test_no_swap_overhead(self) -> None:
        mod = _import_module()
        assert mod.IONQ_ARIA.swap_overhead == 1.0
        assert mod.IONQ_FORTE.swap_overhead == 1.0

    def test_max_qubits(self) -> None:
        mod = _import_module()
        assert mod.IONQ_ARIA.max_qubits == 25
        assert mod.IONQ_FORTE.max_qubits == 32

    def test_frozen(self) -> None:
        mod = _import_module()
        with pytest.raises(AttributeError):
            mod.IONQ_ARIA.name = "changed"  # type: ignore[misc]


class TestRigettiTarget:
    def test_ankaa_arn(self) -> None:
        mod = _import_module()
        assert "rigetti" in mod.RIGETTI_ANKAA.arn.lower()

    def test_grid_topology(self) -> None:
        mod = _import_module()
        assert mod.RIGETTI_ANKAA.topology == mod.Topology.GRID

    def test_swap_overhead_gt_1(self) -> None:
        mod = _import_module()
        assert mod.RIGETTI_ANKAA.swap_overhead > 1.0

    def test_max_qubits(self) -> None:
        mod = _import_module()
        assert mod.RIGETTI_ANKAA.max_qubits == 84


class TestIQMTarget:
    def test_garnet_arn(self) -> None:
        mod = _import_module()
        assert "iqm" in mod.IQM_GARNET.arn.lower()

    def test_grid_topology(self) -> None:
        mod = _import_module()
        assert mod.IQM_GARNET.topology == mod.Topology.GRID


# ===================================================================
# Cost estimation tests
# ===================================================================


class TestCostEstimation:
    def test_ionq_cost(self) -> None:
        mod = _import_module()
        est = mod.estimate_cost(mod.IONQ_ARIA, shots=100)
        assert est.total_usd == pytest.approx(0.30 + 100 * 0.01, abs=1e-6)

    def test_rigetti_cost(self) -> None:
        mod = _import_module()
        est = mod.estimate_cost(mod.RIGETTI_ANKAA, shots=1000)
        expected = 0.30 + 1000 * 0.00035
        assert est.total_usd == pytest.approx(expected, abs=1e-6)

    def test_iqm_cost(self) -> None:
        mod = _import_module()
        est = mod.estimate_cost(mod.IQM_GARNET, shots=2000)
        expected = 0.30 + 2000 * 0.00045
        assert est.total_usd == pytest.approx(expected, abs=1e-6)

    def test_multi_task_cost(self) -> None:
        mod = _import_module()
        est = mod.estimate_cost(mod.IONQ_FORTE, shots=100, n_tasks=5)
        per_task = 0.30 + 100 * 0.01
        assert est.total_usd == pytest.approx(5 * per_task, abs=1e-6)

    def test_unknown_device_raises(self) -> None:
        mod = _import_module()
        fake = mod.IonQTarget(name="fake", arn="arn:aws:braket:fake/device")
        with pytest.raises(ValueError, match="No pricing data"):
            mod.estimate_cost(fake, shots=100)

    def test_cost_estimate_fields(self) -> None:
        mod = _import_module()
        est = mod.estimate_cost(mod.IONQ_ARIA, shots=50)
        assert est.device_name == "IonQ Aria"
        assert est.device_arn == mod.IONQ_ARIA.arn
        assert est.shots == 50
        assert est.per_task_usd == 0.30
        assert est.per_shot_usd == 0.01


# ===================================================================
# SWAP overhead / circuit analysis tests
# ===================================================================


class TestSwapOverhead:
    def test_ionq_no_swaps(self) -> None:
        mod = _import_module()
        result = mod.analyze_swap_overhead(mod.IONQ_ARIA, n_qubits=10, n_two_qubit_gates=20)
        assert result["estimated_swaps"] == 0
        assert result["total_two_qubit_gates"] == 20
        assert result["overhead_factor"] == 1.0
        assert result["topology"] == "all_to_all"

    def test_rigetti_has_swaps(self) -> None:
        mod = _import_module()
        result = mod.analyze_swap_overhead(
            mod.RIGETTI_ANKAA, n_qubits=16, n_two_qubit_gates=30
        )
        assert result["estimated_swaps"] > 0
        assert result["total_two_qubit_gates"] > 30
        assert result["overhead_factor"] > 1.0
        assert result["topology"] == "grid"

    def test_exceeds_max_qubits_raises(self) -> None:
        mod = _import_module()
        with pytest.raises(ValueError, match="supports at most"):
            mod.analyze_swap_overhead(mod.IONQ_ARIA, n_qubits=100, n_two_qubit_gates=10)

    def test_zero_two_qubit_gates(self) -> None:
        mod = _import_module()
        result = mod.analyze_swap_overhead(mod.IONQ_ARIA, n_qubits=5, n_two_qubit_gates=0)
        assert result["estimated_swaps"] == 0
        assert result["overhead_factor"] == 0.0


# ===================================================================
# Hybrid job tests (all mocked)
# ===================================================================


class TestHybridJobs:
    def _make_backend(self):
        mod = _import_module()
        backend = mod.BraketBackend.__new__(mod.BraketBackend)
        backend._device_arn = mod.IONQ_ARIA.arn
        backend._s3_bucket = "my-bucket"
        backend._s3_prefix = "qufin-results"
        backend._is_local = False
        backend._device = MagicMock()
        return backend

    def test_submit_hybrid_job(self) -> None:
        backend = self._make_backend()
        mock_job = MagicMock()
        mock_job.arn = "arn:aws:braket:us-east-1:123:job/test-job"
        mock_job.state.return_value = "QUEUED"

        with patch.object(
            _braket_mocks["braket.aws"].AwsQuantumJob,
            "create",
            return_value=mock_job,
        ):
            handle = backend.submit_hybrid_job("my_script.py")

        assert handle.job_arn == mock_job.arn
        assert handle.status == "QUEUED"

    def test_submit_hybrid_job_local_raises(self) -> None:
        backend = self._make_backend()
        backend._is_local = True
        with pytest.raises(RuntimeError, match="Hybrid jobs require"):
            backend.submit_hybrid_job("script.py")

    def test_poll_job_status(self) -> None:
        backend = self._make_backend()
        mock_job = MagicMock()
        mock_job.state.return_value = "RUNNING"

        with patch.object(
            _braket_mocks["braket.aws"],
            "AwsQuantumJob",
            return_value=mock_job,
        ):
            handle = backend.poll_job_status("arn:aws:braket:job/test")

        assert handle.status == "RUNNING"

    def test_poll_job_completed_includes_result(self) -> None:
        backend = self._make_backend()
        mock_job = MagicMock()
        mock_job.state.return_value = "COMPLETED"
        mock_job.result.return_value = {"energy": -1.5}

        with patch.object(
            _braket_mocks["braket.aws"],
            "AwsQuantumJob",
            return_value=mock_job,
        ):
            handle = backend.poll_job_status("arn:aws:braket:job/done")

        assert handle.status == "COMPLETED"
        assert handle.metadata["result"] == {"energy": -1.5}

    def test_get_job_result_completed(self) -> None:
        backend = self._make_backend()
        mock_job = MagicMock()
        mock_job.state.return_value = "COMPLETED"
        mock_job.result.return_value = {"answer": 42}

        with patch.object(
            _braket_mocks["braket.aws"],
            "AwsQuantumJob",
            return_value=mock_job,
        ):
            result = backend.get_job_result("arn:aws:braket:job/done")

        assert result["status"] == "COMPLETED"
        assert result["result"] == {"answer": 42}

    def test_get_job_result_failed_raises(self) -> None:
        backend = self._make_backend()
        mock_job = MagicMock()
        mock_job.state.return_value = "FAILED"

        with (
            patch.object(
                _braket_mocks["braket.aws"],
                "AwsQuantumJob",
                return_value=mock_job,
            ),
            pytest.raises(RuntimeError, match="FAILED"),
        ):
            backend.get_job_result("arn:aws:braket:job/bad")

    def test_get_job_result_pending(self) -> None:
        backend = self._make_backend()
        mock_job = MagicMock()
        mock_job.state.return_value = "RUNNING"

        with patch.object(
            _braket_mocks["braket.aws"],
            "AwsQuantumJob",
            return_value=mock_job,
        ):
            result = backend.get_job_result("arn:aws:braket:job/pending")

        assert result["status"] == "RUNNING"
        assert "not yet completed" in result["message"]


# ===================================================================
# Backend instance-level cost estimation
# ===================================================================


class TestBackendCostMethod:
    def test_instance_cost_raises_for_local(self) -> None:
        mod = _import_module()
        backend = mod.BraketBackend.__new__(mod.BraketBackend)
        backend._is_local = True
        backend._device_arn = None
        with pytest.raises(ValueError, match="local simulator"):
            backend.estimate_cost(shots=100)
