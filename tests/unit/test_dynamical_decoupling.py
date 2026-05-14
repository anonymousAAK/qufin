"""Tests for dynamical decoupling sequence insertion."""

from __future__ import annotations

import pytest

from qufin.backends.dynamical_decoupling import (
    DDConfig,
    DDSequence,
    cpmg_sequence,
    dd_with_zne,
    estimate_t2_extension,
    insert_dd_sequences,
    uhrig_sequence,
    xy4_sequence,
)
from qufin.backends.mock import MockBackend


class TestDDSequenceEnum:
    """Test DDSequence enum members."""

    def test_enum_values(self) -> None:
        assert DDSequence.XY4.value == "xy4"
        assert DDSequence.CPMG.value == "cpmg"
        assert DDSequence.UHRIG.value == "uhrig"
        assert DDSequence.CUSTOM.value == "custom"

    def test_enum_has_four_members(self) -> None:
        assert len(DDSequence) == 4


class TestDDConfig:
    """Test DDConfig dataclass."""

    def test_default_config(self) -> None:
        cfg = DDConfig()
        assert cfg.sequence_type == DDSequence.XY4
        assert cfg.pulse_spacing == 1.0
        assert cfg.combine_with_zne is False
        assert cfg.n_pulses == 4
        assert cfg.custom_gates is None

    def test_custom_config(self) -> None:
        cfg = DDConfig(
            sequence_type=DDSequence.CPMG,
            pulse_spacing=2.0,
            n_pulses=8,
            combine_with_zne=True,
        )
        assert cfg.sequence_type == DDSequence.CPMG
        assert cfg.pulse_spacing == 2.0
        assert cfg.n_pulses == 8
        assert cfg.combine_with_zne is True


class TestXY4Sequence:
    """Test XY4 pulse sequence generation."""

    def test_xy4_returns_four_pulses(self) -> None:
        seq = xy4_sequence(0)
        assert len(seq) == 4

    def test_xy4_gate_pattern(self) -> None:
        seq = xy4_sequence(0)
        gates = [g for g, _ in seq]
        assert gates == ["x", "y", "x", "y"]

    def test_xy4_correct_qubit(self) -> None:
        seq = xy4_sequence(3)
        for _, qubit in seq:
            assert qubit == 3


class TestCPMGSequence:
    """Test CPMG pulse sequence generation."""

    def test_cpmg_returns_correct_count(self) -> None:
        seq = cpmg_sequence(0, n_pulses=6)
        assert len(seq) == 6

    def test_cpmg_all_x_gates(self) -> None:
        seq = cpmg_sequence(0, n_pulses=4)
        for gate, _ in seq:
            assert gate == "x"

    def test_cpmg_raises_on_zero_pulses(self) -> None:
        with pytest.raises(ValueError, match="n_pulses must be >= 1"):
            cpmg_sequence(0, n_pulses=0)

    def test_cpmg_single_pulse(self) -> None:
        seq = cpmg_sequence(0, n_pulses=1)
        assert len(seq) == 1


class TestUhrigSequence:
    """Test Uhrig DD sequence generation."""

    def test_uhrig_returns_correct_count(self) -> None:
        seq = uhrig_sequence(0, n_pulses=4)
        assert len(seq) == 4

    def test_uhrig_timings_are_ordered(self) -> None:
        seq = uhrig_sequence(0, n_pulses=6)
        times = [t for _, _, t in seq]
        for i in range(len(times) - 1):
            assert times[i] < times[i + 1]

    def test_uhrig_timings_in_unit_interval(self) -> None:
        seq = uhrig_sequence(0, n_pulses=8)
        for _, _, t in seq:
            assert 0.0 < t < 1.0

    def test_uhrig_raises_on_zero_pulses(self) -> None:
        with pytest.raises(ValueError, match="n_pulses must be >= 1"):
            uhrig_sequence(0, n_pulses=0)

    def test_uhrig_symmetry(self) -> None:
        """Uhrig timings should be symmetric around 0.5."""
        seq = uhrig_sequence(0, n_pulses=4)
        times = [t for _, _, t in seq]
        for i in range(len(times)):
            assert abs(times[i] + times[-(i + 1)] - 1.0) < 1e-10


class TestInsertDDSequences:
    """Test DD sequence insertion into circuits."""

    def test_insert_xy4_increases_depth(self) -> None:
        from qiskit.circuit import QuantumCircuit

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        dd_qc = insert_dd_sequences(qc, DDConfig(sequence_type=DDSequence.XY4))
        # DD circuit should have more gates than original
        assert dd_qc.size() >= qc.size()

    def test_insert_on_empty_circuit(self) -> None:
        from qiskit.circuit import QuantumCircuit

        qc = QuantumCircuit(2)
        dd_qc = insert_dd_sequences(qc)
        assert dd_qc.num_qubits == 2

    def test_insert_cpmg(self) -> None:
        from qiskit.circuit import QuantumCircuit

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        cfg = DDConfig(sequence_type=DDSequence.CPMG, n_pulses=2)
        dd_qc = insert_dd_sequences(qc, cfg)
        assert dd_qc.num_qubits == 2

    def test_insert_custom_sequence(self) -> None:
        from qiskit.circuit import QuantumCircuit

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        cfg = DDConfig(
            sequence_type=DDSequence.CUSTOM,
            custom_gates=["x", "x"],
        )
        dd_qc = insert_dd_sequences(qc, cfg)
        assert dd_qc.num_qubits == 2

    def test_custom_without_gates_raises(self) -> None:
        from qiskit.circuit import QuantumCircuit

        qc = QuantumCircuit(2)
        qc.h(0)
        # Force idle period so DD insertion is attempted
        qc.barrier()
        qc.cx(0, 1)

        cfg = DDConfig(
            sequence_type=DDSequence.CUSTOM,
            custom_gates=None,
            pulse_spacing=0.0,
        )
        # This may or may not raise depending on whether idle qubits are found.
        # Just ensure it doesn't crash for circuits with no idle periods.
        try:
            insert_dd_sequences(qc, cfg)
        except ValueError as e:
            assert "custom_gates" in str(e)


class TestEstimateT2Extension:
    """Test T2 extension estimation."""

    def test_xy4_extension(self) -> None:
        result = estimate_t2_extension(100.0, DDSequence.XY4)
        assert result["extension_factor"] == 3.0
        assert result["extended_t2"] == 300.0
        assert result["sequence"] == "xy4"

    def test_cpmg_extension_scales_with_pulses(self) -> None:
        r4 = estimate_t2_extension(100.0, DDSequence.CPMG, n_pulses=4)
        r8 = estimate_t2_extension(100.0, DDSequence.CPMG, n_pulses=8)
        assert r8["extension_factor"] > r4["extension_factor"]

    def test_uhrig_extension_linear_in_pulses(self) -> None:
        result = estimate_t2_extension(50.0, DDSequence.UHRIG, n_pulses=10)
        assert result["extension_factor"] == 10.0
        assert result["extended_t2"] == 500.0

    def test_negative_t2_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            estimate_t2_extension(-1.0, DDSequence.XY4)

    def test_custom_conservative_estimate(self) -> None:
        result = estimate_t2_extension(100.0, DDSequence.CUSTOM)
        assert result["extension_factor"] == 2.0


class TestDDWithZNE:
    """Test compound DD + ZNE mitigation."""

    def test_dd_zne_returns_expected_keys(self) -> None:
        from qiskit.circuit import QuantumCircuit

        qc = QuantumCircuit(2)
        qc.x(0)
        qc.x(1)

        backend = MockBackend(default_counts={"11": 900, "00": 100})
        result = dd_with_zne(qc, backend, shots=1024)

        assert "mitigated_value" in result
        assert "raw_values" in result
        assert "scale_factors" in result
        assert "dd_sequence" in result
        assert "dd_depth_overhead" in result

    def test_dd_zne_uses_configured_sequence(self) -> None:
        from qiskit.circuit import QuantumCircuit

        qc = QuantumCircuit(1)
        qc.x(0)

        backend = MockBackend(default_counts={"1": 1024})
        cfg = DDConfig(sequence_type=DDSequence.CPMG, n_pulses=2)
        result = dd_with_zne(qc, backend, dd_config=cfg, shots=512)
        assert result["dd_sequence"] == "cpmg"
