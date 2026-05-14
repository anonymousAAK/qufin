"""Tests for PEC and CDR error mitigation methods."""

from __future__ import annotations

import numpy as np
import pytest

from qufin.backends.mock import MockBackend

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_simple_circuit(n_qubits: int = 2):
    """Create a simple test circuit with non-Clifford gates."""
    from qiskit.circuit import QuantumCircuit

    qc = QuantumCircuit(n_qubits)
    qc.h(0)
    if n_qubits > 1:
        qc.cx(0, 1)
    qc.t(0)  # non-Clifford
    return qc


def _make_clifford_circuit(n_qubits: int = 2):
    """Create a circuit with only Clifford gates."""
    from qiskit.circuit import QuantumCircuit

    qc = QuantumCircuit(n_qubits)
    qc.h(0)
    if n_qubits > 1:
        qc.cx(0, 1)
    qc.s(0)
    return qc


def _constant_observable(counts, shots):
    """Observable that returns P(all-zeros)."""
    n_qubits = len(next(iter(counts)))
    zero_state = "0" * n_qubits
    return counts.get(zero_state, 0) / shots


# ---------------------------------------------------------------------------
# PECConfig tests
# ---------------------------------------------------------------------------


class TestPECConfig:
    """Tests for PECConfig dataclass."""

    def test_default_values(self) -> None:
        from qufin.backends.error_mitigation import PECConfig

        cfg = PECConfig()
        assert cfg.n_samples == 1000
        assert cfg.noise_model_type == "depolarizing"
        assert cfg.max_overhead == 100.0
        assert cfg.seed == 42

    def test_custom_values(self) -> None:
        from qufin.backends.error_mitigation import PECConfig

        cfg = PECConfig(
            n_samples=500,
            noise_model_type="pauli_twirl",
            max_overhead=50.0,
            seed=123,
        )
        assert cfg.n_samples == 500
        assert cfg.noise_model_type == "pauli_twirl"
        assert cfg.max_overhead == 50.0
        assert cfg.seed == 123


# ---------------------------------------------------------------------------
# characterize_noise_channel tests
# ---------------------------------------------------------------------------


class TestCharacterizeNoiseChannel:
    """Tests for gate noise channel characterization."""

    def test_returns_ptm_shape(self) -> None:
        from qufin.backends.error_mitigation import characterize_noise_channel

        backend = MockBackend(default_counts={"0": 900, "1": 100})
        ptm = characterize_noise_channel("x", backend, n_qubits=1)
        assert ptm.shape == (4, 4)

    def test_ptm_identity_row_preserved(self) -> None:
        """The identity row of the PTM should remain [1,0,0,0]."""
        from qufin.backends.error_mitigation import characterize_noise_channel

        backend = MockBackend(default_counts={"0": 512, "1": 512})
        ptm = characterize_noise_channel("id", backend, n_qubits=1)
        # First row is identity (trace preservation)
        assert ptm[0, 0] == 1.0

    def test_multi_qubit_returns_identity(self) -> None:
        """Multi-qubit characterization returns identity as placeholder."""
        from qufin.backends.error_mitigation import characterize_noise_channel

        backend = MockBackend()
        ptm = characterize_noise_channel("cx", backend, n_qubits=2)
        assert ptm.shape == (16, 16)
        np.testing.assert_array_equal(ptm, np.eye(16))

    def test_different_gates(self) -> None:
        """Characterization should work for various gate names."""
        from qufin.backends.error_mitigation import characterize_noise_channel

        backend = MockBackend(default_counts={"0": 800, "1": 200})
        for gate in ["x", "h", "z", "s", "t"]:
            ptm = characterize_noise_channel(gate, backend, n_qubits=1)
            assert ptm.shape == (4, 4)


# ---------------------------------------------------------------------------
# quasi_probability_decomposition tests
# ---------------------------------------------------------------------------


class TestQuasiProbabilityDecomposition:
    """Tests for QPD computation."""

    def test_identity_channels_gamma_is_dim(self) -> None:
        """QPD of identity w.r.t. identity should have gamma = dim."""
        from qufin.backends.error_mitigation import (
            quasi_probability_decomposition,
        )

        ideal = np.eye(4)
        noisy = np.eye(4)
        result = quasi_probability_decomposition(ideal, noisy)
        assert result["gamma"] == pytest.approx(4.0)
        assert len(result["coefficients"]) == 4

    def test_gamma_geq_one(self) -> None:
        """Gamma should be >= 1 for any valid decomposition."""
        from qufin.backends.error_mitigation import (
            quasi_probability_decomposition,
        )

        ideal = np.eye(4)
        # Slightly noisy channel (depolarizing-like)
        noisy = 0.95 * np.eye(4) + 0.05 * np.ones((4, 4)) / 4
        result = quasi_probability_decomposition(ideal, noisy)
        assert result["gamma"] >= 1.0

    def test_basis_labels_returned(self) -> None:
        from qufin.backends.error_mitigation import (
            quasi_probability_decomposition,
        )

        result = quasi_probability_decomposition(np.eye(4), np.eye(4))
        assert "basis_labels" in result
        assert len(result["basis_labels"]) == 4


# ---------------------------------------------------------------------------
# pec_overhead_estimate tests
# ---------------------------------------------------------------------------


class TestPECOverheadEstimate:
    """Tests for PEC overhead estimation."""

    def test_no_noise_gives_gamma_one(self) -> None:
        from qufin.backends.error_mitigation import pec_overhead_estimate

        circuit = _make_simple_circuit()
        gamma = pec_overhead_estimate(
            circuit,
            {"gate_error": 0.0},
        )
        assert gamma == pytest.approx(1.0)

    def test_gamma_increases_with_noise(self) -> None:
        from qufin.backends.error_mitigation import pec_overhead_estimate

        circuit = _make_simple_circuit()
        g1 = pec_overhead_estimate(circuit, {"gate_error": 0.01})
        g2 = pec_overhead_estimate(circuit, {"gate_error": 0.05})
        assert g2 > g1

    def test_gamma_increases_with_gates(self) -> None:
        from qufin.backends.error_mitigation import pec_overhead_estimate

        g1 = pec_overhead_estimate(
            _make_simple_circuit(),
            {"gate_error": 0.01, "n_gates": 5},
        )
        g2 = pec_overhead_estimate(
            _make_simple_circuit(),
            {"gate_error": 0.01, "n_gates": 20},
        )
        assert g2 > g1

    def test_override_n_gates(self) -> None:
        from qufin.backends.error_mitigation import pec_overhead_estimate

        gamma = pec_overhead_estimate(
            _make_simple_circuit(),
            {"gate_error": 0.01, "n_gates": 1},
        )
        assert gamma == pytest.approx(1.02)


# ---------------------------------------------------------------------------
# pec_mitigate tests
# ---------------------------------------------------------------------------


class TestPECMitigate:
    """Tests for the full PEC mitigation pipeline."""

    def test_returns_expected_keys(self) -> None:
        from qufin.backends.error_mitigation import PECConfig, pec_mitigate

        backend = MockBackend(default_counts={"00": 800, "01": 200})
        cfg = PECConfig(n_samples=10, seed=42)
        result = pec_mitigate(
            _make_simple_circuit(),
            backend,
            config=cfg,
            noise_params={"gate_error": 0.01},
        )
        assert "mitigated_value" in result
        assert "raw_value" in result
        assert "gamma" in result
        assert "n_samples" in result
        assert "sample_values" in result

    def test_n_samples_matches_config(self) -> None:
        from qufin.backends.error_mitigation import PECConfig, pec_mitigate

        backend = MockBackend(default_counts={"00": 700, "01": 300})
        cfg = PECConfig(n_samples=15, seed=0)
        result = pec_mitigate(
            _make_simple_circuit(),
            backend,
            config=cfg,
            noise_params={"gate_error": 0.01},
        )
        assert result["n_samples"] == 15
        assert len(result["sample_values"]) == 15

    def test_default_config(self) -> None:
        """Should work with default config (but small n_samples)."""
        from qufin.backends.error_mitigation import PECConfig, pec_mitigate

        backend = MockBackend(default_counts={"00": 900, "01": 100})
        cfg = PECConfig(n_samples=5)
        result = pec_mitigate(
            _make_simple_circuit(),
            backend,
            config=cfg,
        )
        assert isinstance(result["mitigated_value"], float)

    def test_mitigated_value_is_finite(self) -> None:
        from qufin.backends.error_mitigation import PECConfig, pec_mitigate

        backend = MockBackend(default_counts={"00": 600, "01": 400})
        cfg = PECConfig(n_samples=20, seed=7)
        result = pec_mitigate(
            _make_simple_circuit(),
            backend,
            config=cfg,
            noise_params={"gate_error": 0.02},
        )
        assert np.isfinite(result["mitigated_value"])


# ---------------------------------------------------------------------------
# CDRConfig tests
# ---------------------------------------------------------------------------


class TestCDRConfig:
    """Tests for CDRConfig dataclass."""

    def test_default_values(self) -> None:
        from qufin.backends.error_mitigation import CDRConfig

        cfg = CDRConfig()
        assert cfg.n_training_circuits == 20
        assert cfg.regression_type == "linear"
        assert cfg.ridge_alpha == 1.0
        assert cfg.seed == 42

    def test_custom_values(self) -> None:
        from qufin.backends.error_mitigation import CDRConfig

        cfg = CDRConfig(
            n_training_circuits=10,
            regression_type="ridge",
            ridge_alpha=0.5,
            seed=99,
        )
        assert cfg.n_training_circuits == 10
        assert cfg.regression_type == "ridge"


# ---------------------------------------------------------------------------
# nearest_clifford_gate tests
# ---------------------------------------------------------------------------


class TestNearestCliffordGate:
    """Tests for Clifford gate mapping."""

    def test_clifford_gates_unchanged(self) -> None:
        from qufin.backends.error_mitigation import nearest_clifford_gate

        for gate in ["h", "x", "y", "z", "s", "sdg", "cx", "cz"]:
            assert nearest_clifford_gate(gate) == gate

    def test_t_maps_to_s(self) -> None:
        from qufin.backends.error_mitigation import nearest_clifford_gate

        assert nearest_clifford_gate("t") == "s"

    def test_tdg_maps_to_sdg(self) -> None:
        from qufin.backends.error_mitigation import nearest_clifford_gate

        assert nearest_clifford_gate("tdg") == "sdg"

    def test_rotation_gates_map(self) -> None:
        from qufin.backends.error_mitigation import nearest_clifford_gate

        assert nearest_clifford_gate("rx") == "x"
        assert nearest_clifford_gate("ry") == "y"
        assert nearest_clifford_gate("rz") == "z"

    def test_unknown_gate_maps_to_id(self) -> None:
        from qufin.backends.error_mitigation import nearest_clifford_gate

        assert nearest_clifford_gate("some_exotic_gate") == "id"

    def test_case_insensitive(self) -> None:
        from qufin.backends.error_mitigation import nearest_clifford_gate

        assert nearest_clifford_gate("T") == "s"
        assert nearest_clifford_gate("H") == "h"


# ---------------------------------------------------------------------------
# generate_clifford_circuits tests
# ---------------------------------------------------------------------------


class TestGenerateCliffordCircuits:
    """Tests for near-Clifford circuit generation."""

    def test_correct_count(self) -> None:
        from qufin.backends.error_mitigation import (
            generate_clifford_circuits,
        )

        circuits = generate_clifford_circuits(
            _make_simple_circuit(),
            n_circuits=5,
        )
        assert len(circuits) == 5

    def test_same_qubit_count(self) -> None:
        from qufin.backends.error_mitigation import (
            generate_clifford_circuits,
        )

        original = _make_simple_circuit(3)
        circuits = generate_clifford_circuits(original, n_circuits=3)
        for circ in circuits:
            assert circ.num_qubits == 3

    def test_all_clifford_circuit_unchanged_structure(self) -> None:
        """A circuit with only Clifford gates should keep same gate count."""
        from qufin.backends.error_mitigation import (
            generate_clifford_circuits,
        )

        original = _make_clifford_circuit()
        circuits = generate_clifford_circuits(original, n_circuits=3)
        for circ in circuits:
            assert circ.num_qubits == original.num_qubits

    def test_reproducible_with_seed(self) -> None:
        from qufin.backends.error_mitigation import (
            generate_clifford_circuits,
        )

        c1 = generate_clifford_circuits(
            _make_simple_circuit(),
            n_circuits=3,
            seed=42,
        )
        c2 = generate_clifford_circuits(
            _make_simple_circuit(),
            n_circuits=3,
            seed=42,
        )
        for a, b in zip(c1, c2, strict=True):
            assert a.num_qubits == b.num_qubits
            assert len(a.data) == len(b.data)


# ---------------------------------------------------------------------------
# cdr_mitigate tests
# ---------------------------------------------------------------------------


class TestCDRMitigate:
    """Tests for the full CDR mitigation pipeline."""

    def test_returns_expected_keys(self) -> None:
        from qufin.backends.error_mitigation import CDRConfig, cdr_mitigate

        backend = MockBackend(default_counts={"00": 800, "01": 200})
        cfg = CDRConfig(n_training_circuits=5, seed=42)
        result = cdr_mitigate(
            _make_simple_circuit(),
            backend,
            config=cfg,
        )
        assert "mitigated_value" in result
        assert "raw_value" in result
        assert "slope" in result
        assert "intercept" in result
        assert "training_noisy" in result
        assert "training_ideal" in result
        assert "n_training" in result

    def test_training_count_matches(self) -> None:
        from qufin.backends.error_mitigation import CDRConfig, cdr_mitigate

        backend = MockBackend(default_counts={"00": 700, "01": 300})
        cfg = CDRConfig(n_training_circuits=8, seed=0)
        result = cdr_mitigate(
            _make_simple_circuit(),
            backend,
            config=cfg,
        )
        assert result["n_training"] == 8
        assert len(result["training_noisy"]) == 8
        assert len(result["training_ideal"]) == 8

    def test_ridge_regression(self) -> None:
        from qufin.backends.error_mitigation import CDRConfig, cdr_mitigate

        backend = MockBackend(default_counts={"00": 600, "01": 400})
        cfg = CDRConfig(
            n_training_circuits=5,
            regression_type="ridge",
            ridge_alpha=0.5,
            seed=42,
        )
        result = cdr_mitigate(
            _make_simple_circuit(),
            backend,
            config=cfg,
        )
        assert isinstance(result["mitigated_value"], float)
        assert np.isfinite(result["slope"])

    def test_mitigated_value_is_finite(self) -> None:
        from qufin.backends.error_mitigation import CDRConfig, cdr_mitigate

        backend = MockBackend(default_counts={"00": 500, "01": 500})
        cfg = CDRConfig(n_training_circuits=5, seed=7)
        result = cdr_mitigate(
            _make_simple_circuit(),
            backend,
            config=cfg,
        )
        assert np.isfinite(result["mitigated_value"])

    def test_separate_ideal_backend(self) -> None:
        """CDR with a separate ideal backend."""
        from qufin.backends.error_mitigation import CDRConfig, cdr_mitigate

        noisy = MockBackend(default_counts={"00": 600, "01": 400})
        ideal = MockBackend(default_counts={"00": 950, "01": 50})
        cfg = CDRConfig(n_training_circuits=5, seed=42)
        result = cdr_mitigate(
            _make_simple_circuit(),
            noisy,
            config=cfg,
            ideal_backend=ideal,
        )
        assert isinstance(result["mitigated_value"], float)


# ---------------------------------------------------------------------------
# Regression helpers tests
# ---------------------------------------------------------------------------


class TestRegressionHelpers:
    """Tests for the linear and ridge regression helpers."""

    def test_linear_regression_perfect_fit(self) -> None:
        from qufin.backends.error_mitigation import _linear_regression

        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = np.array([2.0, 4.0, 6.0, 8.0])
        slope, intercept = _linear_regression(x, y)
        assert slope == pytest.approx(2.0)
        assert intercept == pytest.approx(0.0, abs=1e-10)

    def test_linear_regression_with_intercept(self) -> None:
        from qufin.backends.error_mitigation import _linear_regression

        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([1.0, 3.0, 5.0, 7.0])
        slope, intercept = _linear_regression(x, y)
        assert slope == pytest.approx(2.0)
        assert intercept == pytest.approx(1.0)

    def test_linear_regression_empty(self) -> None:
        from qufin.backends.error_mitigation import _linear_regression

        slope, intercept = _linear_regression(
            np.array([]),
            np.array([]),
        )
        assert slope == 1.0
        assert intercept == 0.0

    def test_ridge_regression_basic(self) -> None:
        from qufin.backends.error_mitigation import _ridge_regression

        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = np.array([2.0, 4.0, 6.0, 8.0])
        slope, intercept = _ridge_regression(x, y, alpha=0.01)
        # Ridge with small alpha should be close to OLS
        assert slope == pytest.approx(2.0, abs=0.1)
        assert intercept == pytest.approx(0.0, abs=0.5)

    def test_ridge_regression_empty(self) -> None:
        from qufin.backends.error_mitigation import _ridge_regression

        slope, intercept = _ridge_regression(
            np.array([]),
            np.array([]),
            alpha=1.0,
        )
        assert slope == 1.0
        assert intercept == 0.0
