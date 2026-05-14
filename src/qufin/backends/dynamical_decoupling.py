"""Dynamical Decoupling (DD) sequence insertion for decoherence suppression.

Implements DD pulse sequences that suppress low-frequency noise during
idle periods in quantum circuits. Supported sequences:

- **XY4**: Four-pulse sequence (X-Y-X-Y) that suppresses both dephasing
  and amplitude-damping noise.
- **CPMG**: Carr-Purcell-Meiboom-Gill sequence of evenly spaced pi-pulses.
- **Uhrig**: Uhrig Dynamical Decoupling with optimally timed pulses for
  pure dephasing noise.

Can be combined with ZNE for compound mitigation.

References
----------
Viola, Knill, Lloyd, "Dynamical Decoupling of Open Quantum Systems",
  PRL 82:2417 (1999).
Uhrig, "Keeping a Quantum Bit Alive by Optimized Pi-Pulse Sequences",
  PRL 98:100504 (2007).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from qufin.backends.base import Backend


class DDSequence(Enum):
    """Supported dynamical decoupling sequences."""

    XY4 = "xy4"
    CPMG = "cpmg"
    UHRIG = "uhrig"
    CUSTOM = "custom"


@dataclass
class DDConfig:
    """Configuration for dynamical decoupling insertion.

    Parameters
    ----------
    sequence_type : DDSequence
        Which DD sequence to use.
    pulse_spacing : float
        Minimum idle time (in gate units) before inserting DD pulses.
        Default 1.0 means any idle period of 1 or more gate slots gets DD.
    n_pulses : int
        Number of pi-pulses for CPMG/Uhrig sequences. Ignored for XY4.
    combine_with_zne : bool
        Whether to combine DD with ZNE for compound mitigation.
    custom_gates : list[str] | None
        Gate names for CUSTOM sequence type (e.g., ["x", "y", "x", "y"]).
    """

    sequence_type: DDSequence = DDSequence.XY4
    pulse_spacing: float = 1.0
    n_pulses: int = 4
    combine_with_zne: bool = False
    custom_gates: list[str] | None = None


def xy4_sequence(qubit: int) -> list[tuple[str, int]]:
    """Return XY4 dynamical decoupling gate sequence.

    The XY4 sequence applies X-Y-X-Y pulses, which suppress both
    dephasing (T2) and amplitude damping (T1) noise.

    Parameters
    ----------
    qubit : int
        Qubit index to apply the sequence on.

    Returns
    -------
    List of (gate_name, qubit) tuples.
    """
    return [("x", qubit), ("y", qubit), ("x", qubit), ("y", qubit)]


def cpmg_sequence(qubit: int, n_pulses: int = 4) -> list[tuple[str, int]]:
    """Return CPMG dynamical decoupling gate sequence.

    Carr-Purcell-Meiboom-Gill sequence: evenly spaced pi-X pulses.
    Effective at suppressing dephasing noise.

    Parameters
    ----------
    qubit : int
        Qubit index.
    n_pulses : int
        Number of pi-pulses (must be even for proper refocusing).

    Returns
    -------
    List of (gate_name, qubit) tuples.
    """
    if n_pulses < 1:
        raise ValueError(f"n_pulses must be >= 1, got {n_pulses}")
    return [("x", qubit)] * n_pulses


def uhrig_sequence(
    qubit: int, n_pulses: int = 4
) -> list[tuple[str, int, float]]:
    """Return Uhrig DD sequence with optimal timing.

    Uhrig DD uses non-uniform pulse spacing optimized for suppressing
    pure dephasing noise. The j-th pulse (1-indexed) is placed at
    fractional time t_j = sin^2(j * pi / (2*(n+1))).

    Parameters
    ----------
    qubit : int
        Qubit index.
    n_pulses : int
        Number of pi-pulses.

    Returns
    -------
    List of (gate_name, qubit, fractional_time) tuples, where
    fractional_time is in [0, 1] indicating when in the idle period
    the pulse should be applied.
    """
    if n_pulses < 1:
        raise ValueError(f"n_pulses must be >= 1, got {n_pulses}")
    pulses = []
    for j in range(1, n_pulses + 1):
        t_j = np.sin(j * np.pi / (2 * (n_pulses + 1))) ** 2
        pulses.append(("x", qubit, float(t_j)))
    return pulses


def _get_idle_qubits(circuit: Any, moment_idx: int) -> list[int]:
    """Identify qubits that are idle at a given circuit moment.

    Walks through circuit instructions and finds qubits not involved
    in any gate at the specified depth layer.

    Parameters
    ----------
    circuit : QuantumCircuit
        The quantum circuit to analyze.
    moment_idx : int
        The depth layer index to check.

    Returns
    -------
    List of idle qubit indices.
    """
    n_qubits = circuit.num_qubits
    active_qubits: set[int] = set()

    # Build a simple layer model: assign each instruction a depth
    qubit_depth = [0] * n_qubits
    layers: dict[int, list[set[int]]] = {}

    for instruction in circuit.data:
        qubits = [circuit.find_bit(q).index for q in instruction.qubits]
        depth = max(qubit_depth[q] for q in qubits)
        if depth not in layers:
            layers[depth] = []
        qubit_set = set(qubits)
        layers[depth].append(qubit_set)
        for q in qubits:
            qubit_depth[q] = depth + 1

    if moment_idx in layers:
        for qubit_set in layers[moment_idx]:
            active_qubits |= qubit_set

    idle = [q for q in range(n_qubits) if q not in active_qubits]
    return idle


def _apply_gate(circuit: Any, gate_name: str, qubit: int) -> None:
    """Apply a named gate to a circuit.

    Parameters
    ----------
    circuit : QuantumCircuit
        Circuit to modify.
    gate_name : str
        One of "x", "y", "z", "h", "id".
    qubit : int
        Target qubit index.
    """
    gate_map = {
        "x": circuit.x,
        "y": circuit.y,
        "z": circuit.z,
        "h": circuit.h,
        "id": circuit.id,
    }
    if gate_name not in gate_map:
        raise ValueError(
            f"Unknown gate '{gate_name}'. Supported: {list(gate_map.keys())}"
        )
    gate_map[gate_name](qubit)


def insert_dd_sequences(circuit: Any, config: DDConfig | None = None) -> Any:
    """Insert dynamical decoupling sequences into idle periods.

    Analyzes the circuit for idle qubit periods and inserts the
    configured DD pulse sequence to suppress decoherence.

    Parameters
    ----------
    circuit : QuantumCircuit
        Transpiled quantum circuit (without measurements).
    config : DDConfig | None
        DD configuration. Uses XY4 defaults if None.

    Returns
    -------
    New QuantumCircuit with DD sequences inserted.
    """
    from qiskit.circuit import QuantumCircuit

    if config is None:
        config = DDConfig()

    n_qubits = circuit.num_qubits
    depth = circuit.depth()

    if depth == 0:
        return circuit.copy()

    # Build the DD-enhanced circuit
    dd_circuit = QuantumCircuit(n_qubits)

    # Replay original circuit and insert DD during idle periods
    # Simple approach: rebuild gate-by-gate, adding DD after each layer
    qubit_depth = [0] * n_qubits
    layer_instructions: dict[int, list] = {}

    for instruction in circuit.data:
        qargs = instruction.qubits
        qubit_indices = [circuit.find_bit(q).index for q in qargs]
        layer = max(qubit_depth[q] for q in qubit_indices)
        if layer not in layer_instructions:
            layer_instructions[layer] = []
        layer_instructions[layer].append(instruction)
        for q in qubit_indices:
            qubit_depth[q] = layer + 1

    max_depth = max(qubit_depth)
    qubit_placed = [0] * n_qubits

    for layer_idx in range(max_depth):
        # Apply gates in this layer
        if layer_idx in layer_instructions:
            for inst in layer_instructions[layer_idx]:
                qargs = [circuit.find_bit(q).index for q in inst.qubits]
                cargs = [circuit.find_bit(c).index for c in inst.clbits]
                if cargs:
                    dd_circuit.append(
                        inst.operation,
                        qargs,
                        cargs,
                    )
                else:
                    dd_circuit.append(inst.operation, qargs)
                for q in qargs:
                    qubit_placed[q] = layer_idx + 1

        # Insert DD on idle qubits
        for q in range(n_qubits):
            if qubit_placed[q] <= layer_idx:
                idle_duration = layer_idx - qubit_placed[q] + 1
                if idle_duration >= config.pulse_spacing:
                    _insert_dd_on_qubit(dd_circuit, q, config)

    return dd_circuit


def _insert_dd_on_qubit(circuit: Any, qubit: int, config: DDConfig) -> None:
    """Insert a single DD sequence on the given qubit.

    Parameters
    ----------
    circuit : QuantumCircuit
        Circuit to modify in-place.
    qubit : int
        Target qubit.
    config : DDConfig
        DD configuration specifying the sequence type.
    """
    if config.sequence_type == DDSequence.XY4:
        seq = xy4_sequence(qubit)
        for gate_name, q in seq:
            _apply_gate(circuit, gate_name, q)
    elif config.sequence_type == DDSequence.CPMG:
        seq = cpmg_sequence(qubit, config.n_pulses)
        for gate_name, q in seq:
            _apply_gate(circuit, gate_name, q)
    elif config.sequence_type == DDSequence.UHRIG:
        seq = uhrig_sequence(qubit, config.n_pulses)
        for gate_name, q, _t in seq:
            _apply_gate(circuit, gate_name, q)
    elif config.sequence_type == DDSequence.CUSTOM:
        if config.custom_gates is None:
            raise ValueError(
                "custom_gates must be provided for CUSTOM sequence type"
            )
        for gate_name in config.custom_gates:
            _apply_gate(circuit, gate_name, qubit)
    else:
        raise ValueError(f"Unknown sequence type: {config.sequence_type}")


def estimate_t2_extension(
    original_t2: float,
    dd_sequence: DDSequence,
    n_pulses: int = 4,
) -> dict[str, float]:
    """Estimate T2 coherence time improvement from DD sequences.

    Uses analytical models for the expected T2 extension factor
    based on the DD sequence type and number of pulses.

    Parameters
    ----------
    original_t2 : float
        Original T2 coherence time (in microseconds).
    dd_sequence : DDSequence
        DD sequence type.
    n_pulses : int
        Number of pulses (for CPMG/Uhrig).

    Returns
    -------
    Dict with keys: original_t2, extended_t2, extension_factor, sequence.
    """
    if original_t2 <= 0:
        raise ValueError(f"original_t2 must be positive, got {original_t2}")

    # Extension factors based on published results:
    # - XY4 typically extends T2 by 2-5x
    # - CPMG scales as n_pulses^(2/3) for 1/f noise
    # - Uhrig is optimal for pure dephasing, scales as n_pulses
    if dd_sequence == DDSequence.XY4:
        factor = 3.0  # Conservative XY4 estimate
    elif dd_sequence == DDSequence.CPMG:
        factor = float(n_pulses ** (2 / 3))
    elif dd_sequence == DDSequence.UHRIG:
        factor = float(n_pulses)
    elif dd_sequence == DDSequence.CUSTOM:
        factor = 2.0  # Conservative default for custom sequences
    else:
        factor = 1.0

    extended_t2 = original_t2 * factor

    return {
        "original_t2": original_t2,
        "extended_t2": extended_t2,
        "extension_factor": factor,
        "sequence": dd_sequence.value,
    }


def dd_with_zne(
    circuit: Any,
    backend: Backend,
    dd_config: DDConfig | None = None,
    zne_scale_factors: list[float] | None = None,
    shots: int = 4096,
    observable_fn: Any = None,
) -> dict[str, Any]:
    """Compound mitigation: Dynamical Decoupling + Zero-Noise Extrapolation.

    First inserts DD sequences to suppress idle-time decoherence,
    then applies ZNE on the DD-protected circuit.

    Parameters
    ----------
    circuit : QuantumCircuit
        Circuit WITHOUT measurements.
    backend : Backend
        Backend to execute on.
    dd_config : DDConfig | None
        DD configuration. Uses XY4 defaults if None.
    zne_scale_factors : list[float] | None
        Scale factors for ZNE. Default [1, 3, 5].
    shots : int
        Shots per execution.
    observable_fn : callable | None
        Observable function for ZNE. Default: P(all-zeros).

    Returns
    -------
    Dict with keys: mitigated_value, raw_values, scale_factors,
    dd_sequence, dd_depth_overhead.
    """
    from qufin.backends.error_mitigation import zne_extrapolate

    if dd_config is None:
        dd_config = DDConfig()

    # Step 1: Insert DD sequences
    dd_circuit = insert_dd_sequences(circuit, dd_config)

    # Step 2: Apply ZNE on the DD-protected circuit
    zne_result = zne_extrapolate(
        dd_circuit,
        backend,
        scale_factors=zne_scale_factors,
        shots=shots,
        observable_fn=observable_fn,
    )

    return {
        "mitigated_value": zne_result["mitigated_value"],
        "raw_values": zne_result["raw_values"],
        "scale_factors": zne_result["scale_factors"],
        "dd_sequence": dd_config.sequence_type.value,
        "dd_depth_overhead": dd_circuit.depth() - circuit.depth(),
    }
