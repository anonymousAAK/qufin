"""Tests for QAOA mixer circuits."""

from __future__ import annotations

import pytest

from qufin.portfolio.mixers import (
    DickeInitialState,
    GroverMixer,
    XMixer,
    XYFullMixer,
    XYRingMixer,
    get_mixer,
)


class TestXMixer:
    def test_circuit_creation(self) -> None:
        mixer = XMixer(4)
        circ = mixer.circuit(0.5)
        assert circ.num_qubits == 4

    def test_not_hamming_preserving(self) -> None:
        mixer = XMixer(4)
        assert mixer.preserves_hamming_weight is False


class TestXYRingMixer:
    def test_circuit_creation(self) -> None:
        mixer = XYRingMixer(4)
        circ = mixer.circuit(0.5)
        assert circ.num_qubits == 4

    def test_hamming_preserving(self) -> None:
        mixer = XYRingMixer(4)
        assert mixer.preserves_hamming_weight is True


class TestXYFullMixer:
    def test_circuit_creation(self) -> None:
        mixer = XYFullMixer(4)
        circ = mixer.circuit(0.5)
        assert circ.num_qubits == 4

    def test_hamming_preserving(self) -> None:
        mixer = XYFullMixer(4)
        assert mixer.preserves_hamming_weight is True

    def test_more_gates_than_ring(self) -> None:
        ring = XYRingMixer(6)
        full = XYFullMixer(6)
        ring_circ = ring.circuit(0.5)
        full_circ = full.circuit(0.5)
        # Full should have more gates (all-to-all vs ring)
        assert len(full_circ.data) > len(ring_circ.data)


class TestGroverMixer:
    def test_circuit_creation(self) -> None:
        mixer = GroverMixer(3)
        circ = mixer.circuit(0.5)
        assert circ.num_qubits == 3

    def test_not_hamming_preserving(self) -> None:
        mixer = GroverMixer(3)
        assert mixer.preserves_hamming_weight is False


class TestDickeInitialState:
    def test_circuit_creation(self) -> None:
        dicke = DickeInitialState(5, 2)
        circ = dicke.circuit()
        assert circ.num_qubits == 5

    def test_k_greater_than_n_raises(self) -> None:
        with pytest.raises(ValueError, match="k=5 > n_qubits=3"):
            DickeInitialState(3, 5)

    def test_k_equals_n(self) -> None:
        dicke = DickeInitialState(3, 3)
        circ = dicke.circuit()
        assert circ.num_qubits == 3


class TestGetMixer:
    def test_x(self) -> None:
        mixer = get_mixer("x", 4)
        assert isinstance(mixer, XMixer)

    def test_xy_ring(self) -> None:
        mixer = get_mixer("xy_ring", 4)
        assert isinstance(mixer, XYRingMixer)

    def test_xy_full(self) -> None:
        mixer = get_mixer("xy_full", 4)
        assert isinstance(mixer, XYFullMixer)

    def test_grover(self) -> None:
        mixer = get_mixer("grover", 4)
        assert isinstance(mixer, GroverMixer)

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown mixer"):
            get_mixer("unknown", 4)
