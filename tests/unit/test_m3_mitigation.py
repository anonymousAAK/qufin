"""Tests for M3 (matrix-free) measurement mitigation."""

from __future__ import annotations

import numpy as np
import pytest

from qufin.backends.m3_mitigation import (
    CalibrationData,
    M3Config,
    M3Mitigator,
    iterative_correction,
    tensored_calibration,
)
from qufin.backends.mock import MockBackend


class TestM3Config:
    """Test M3Config dataclass."""

    def test_default_config(self) -> None:
        cfg = M3Config()
        assert cfg.n_calibration_shots == 8192
        assert cfg.method == "direct"
        assert cfg.max_iterations == 25
        assert cfg.convergence_tol == 1e-6

    def test_custom_config(self) -> None:
        cfg = M3Config(
            n_calibration_shots=4096,
            method="iterative",
            max_iterations=50,
            convergence_tol=1e-8,
        )
        assert cfg.n_calibration_shots == 4096
        assert cfg.method == "iterative"


class TestTensoredCalibration:
    """Test tensored (per-qubit) calibration."""

    def test_calibration_returns_correct_structure(self) -> None:
        backend = MockBackend(default_counts={"0": 950, "1": 50})
        cal = tensored_calibration(2, backend, shots=1000)
        assert cal.n_qubits == 2
        assert len(cal.qubit_matrices) == 2
        assert cal.shots == 1000

    def test_calibration_matrices_are_2x2(self) -> None:
        backend = MockBackend(default_counts={"0": 1000})
        cal = tensored_calibration(3, backend, shots=1000)
        for mat in cal.qubit_matrices:
            assert mat.shape == (2, 2)

    def test_calibration_columns_sum_to_one(self) -> None:
        backend = MockBackend(default_counts={"0": 900, "1": 100})
        cal = tensored_calibration(2, backend, shots=1000)
        for mat in cal.qubit_matrices:
            for col in range(2):
                col_sum = mat[:, col].sum()
                assert abs(col_sum - 1.0) < 1e-10

    def test_calibration_circuit_count(self) -> None:
        """M3 uses 2 circuits per qubit, not 2^n total."""
        # This is implicit: tensored_calibration runs 2 circuits per qubit
        # We verify the overhead_estimate matches
        mitigator = M3Mitigator()
        overhead = mitigator.overhead_estimate(5)
        assert overhead["m3_calibration_circuits"] == 10
        assert overhead["full_calibration_circuits"] == 32


class TestM3Mitigator:
    """Test the M3Mitigator class."""

    def test_not_calibrated_initially(self) -> None:
        m = M3Mitigator()
        assert not m.is_calibrated

    def test_apply_raises_before_calibration(self) -> None:
        m = M3Mitigator()
        with pytest.raises(RuntimeError, match="not calibrated"):
            m.apply({"00": 500, "11": 500}, shots=1000)

    def test_calibrate_sets_calibrated(self) -> None:
        backend = MockBackend(default_counts={"0": 900, "1": 100})
        m = M3Mitigator()
        m.calibrate(backend, n_qubits=2)
        assert m.is_calibrated
        assert m.calibration_data is not None

    def test_direct_correction_returns_valid_probs(self) -> None:
        backend = MockBackend(default_counts={"0": 950, "1": 50})
        m = M3Mitigator(M3Config(method="direct"))
        m.calibrate(backend, n_qubits=2)

        result = m.apply({"00": 800, "01": 100, "10": 50, "11": 50}, shots=1000)
        probs = result["mitigated_probs"]
        total = sum(probs.values())
        assert abs(total - 1.0) < 0.01

    def test_iterative_correction_returns_valid_probs(self) -> None:
        backend = MockBackend(default_counts={"0": 950, "1": 50})
        m = M3Mitigator(M3Config(method="iterative"))
        m.calibrate(backend, n_qubits=2)

        result = m.apply({"00": 800, "01": 100, "10": 50, "11": 50}, shots=1000)
        probs = result["mitigated_probs"]
        total = sum(probs.values())
        assert abs(total - 1.0) < 0.01

    def test_method_label_in_result(self) -> None:
        backend = MockBackend(default_counts={"0": 950, "1": 50})
        m = M3Mitigator(M3Config(method="direct"))
        m.calibrate(backend, n_qubits=1)
        result = m.apply({"0": 900, "1": 100}, shots=1000)
        assert result["method"] == "m3_direct"


class TestIterativeCorrection:
    """Test iterative Bayesian correction directly."""

    def test_converges_for_identity_calibration(self) -> None:
        """With perfect calibration (identity), output should match input."""
        cal_data = CalibrationData(
            qubit_matrices=[np.eye(2), np.eye(2)],
            n_qubits=2,
            shots=1000,
        )
        counts = {"00": 500, "11": 500}
        result = iterative_correction(counts, cal_data, max_iter=10)
        assert abs(result.get("00", 0) - 0.5) < 0.01
        assert abs(result.get("11", 0) - 0.5) < 0.01

    def test_iterative_probs_sum_to_one(self) -> None:
        cal_data = CalibrationData(
            qubit_matrices=[
                np.array([[0.95, 0.05], [0.05, 0.95]]),
                np.array([[0.95, 0.05], [0.05, 0.95]]),
            ],
            n_qubits=2,
            shots=1000,
        )
        counts = {"00": 700, "01": 100, "10": 100, "11": 100}
        result = iterative_correction(counts, cal_data, max_iter=50)
        total = sum(result.values())
        assert abs(total - 1.0) < 0.01


class TestOverheadEstimate:
    """Test overhead estimation."""

    def test_overhead_scaling(self) -> None:
        m = M3Mitigator()
        o2 = m.overhead_estimate(2)
        o10 = m.overhead_estimate(10)
        # M3 scales linearly, full scales exponentially
        assert o2["m3_calibration_circuits"] == 4
        assert o10["m3_calibration_circuits"] == 20
        assert o10["full_calibration_circuits"] == 1024

    def test_speedup_factor_increases(self) -> None:
        m = M3Mitigator()
        s5 = m.overhead_estimate(5)["speedup_factor"]
        s10 = m.overhead_estimate(10)["speedup_factor"]
        assert s10 > s5
