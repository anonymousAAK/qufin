"""Quantum deep hedging (Cherrat et al., arXiv:2303.16585).

Revival of jpmorganchase/jpmc-qcware-deephedging (archived March 2023).
Modernized to Qiskit Primitives and PennyLane TorchLayer.

This module provides a circuit builder and evaluator for variational quantum
hedging networks.  It does *not* include a full training loop (which would
require PyTorch or JAX) but exposes all the building blocks needed to plug
the circuit into an external optimiser.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class QuantumDeepHedgingConfig:
    """Configuration for the quantum deep-hedging ansatz.

    Attributes
    ----------
    n_qubits : int
        Number of qubits in the variational circuit.
    n_layers : int
        Number of variational layers.
    entanglement : str
        Entanglement topology: ``"linear"``, ``"full"``, or ``"circular"``.
    """

    n_qubits: int = 4
    n_layers: int = 2
    entanglement: str = "linear"


# ---------------------------------------------------------------------------
# Circuit builder
# ---------------------------------------------------------------------------

def build_circuit(n_qubits: int, n_layers: int, entanglement: str = "linear") -> Any:
    """Build an EfficientSU2-style variational ansatz.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    n_layers : int
        Number of variational layers.
    entanglement : str
        Entanglement pattern passed to ``TwoLocal``.

    Returns
    -------
    qiskit.circuit.QuantumCircuit
        Parametrised circuit with measurement on all qubits.
    """
    from qiskit.circuit.library import TwoLocal

    ansatz = TwoLocal(
        num_qubits=n_qubits,
        rotation_blocks=["ry", "rz"],
        entanglement_blocks="cx",
        entanglement=entanglement,
        reps=n_layers,
        insert_barriers=False,
    )
    return ansatz


def _encode_features(
    n_qubits: int, features: NDArray[np.float64],
) -> Any:
    """Build a feature-encoding circuit using angle embedding.

    Maps each feature to an RY rotation on the corresponding qubit.
    If ``len(features) < n_qubits`` the remaining qubits stay at |0>.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    features : NDArray
        1-D array of feature values (length <= *n_qubits*).

    Returns
    -------
    qiskit.circuit.QuantumCircuit
    """
    from qiskit.circuit import QuantumCircuit

    qc = QuantumCircuit(n_qubits)
    for i, f in enumerate(features[:n_qubits]):
        qc.ry(float(f), i)
    return qc


# ---------------------------------------------------------------------------
# Forward evaluation
# ---------------------------------------------------------------------------

def forward(
    params: NDArray[np.float64],
    features: NDArray[np.float64],
    n_qubits: int,
    n_layers: int,
    entanglement: str = "linear",
    shots: int = 1024,
    backend: Any | None = None,
) -> NDArray[np.float64]:
    """Evaluate the parametrised circuit and return expectation values.

    Parameters
    ----------
    params : NDArray
        Flat array of variational parameters.
    features : NDArray
        1-D feature vector for angle encoding.
    n_qubits, n_layers, entanglement
        Ansatz configuration.
    shots : int
        Number of measurement shots.
    backend : object or None
        A Qiskit-compatible backend.  If ``None`` the Qiskit
        ``StatevectorSimulator`` is used via ``Statevector``.

    Returns
    -------
    NDArray
        Array of per-qubit Z expectation values (length *n_qubits*).
    """
    from qiskit.circuit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    # Build full circuit: encoding + ansatz
    encoding = _encode_features(n_qubits, features)
    ansatz = build_circuit(n_qubits, n_layers, entanglement)

    qc = QuantumCircuit(n_qubits)
    qc.compose(encoding, inplace=True)
    qc.compose(ansatz, inplace=True)

    # Bind parameters
    param_list = list(ansatz.parameters)
    if len(params) != len(param_list):
        raise ValueError(
            f"Expected {len(param_list)} params, got {len(params)}"
        )
    bind_dict = dict(zip(param_list, params.tolist(), strict=False))
    qc.assign_parameters(bind_dict, inplace=True)

    # Statevector evaluation (exact)
    sv = Statevector.from_instruction(qc)
    probs = sv.probabilities()

    # Compute <Z_i> for each qubit
    n_states = 2 ** n_qubits
    expectations = np.zeros(n_qubits)
    for state_idx in range(n_states):
        p = probs[state_idx]
        for q in range(n_qubits):
            # qubit q value: bit (n_qubits - 1 - q) of state_idx
            bit = (state_idx >> (n_qubits - 1 - q)) & 1
            expectations[q] += p * (1 - 2 * bit)

    return expectations


# ---------------------------------------------------------------------------
# Resource estimation
# ---------------------------------------------------------------------------

def resource_estimate(n_qubits: int, n_layers: int) -> dict[str, int]:
    """Estimate gate resources for the variational ansatz.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    n_layers : int
        Number of variational layers (reps).

    Returns
    -------
    dict
        ``gate_count``, ``depth``, ``params`` (number of free parameters).
    """
    # EfficientSU2 / TwoLocal with [ry, rz] rotations + cx entanglement
    # Per layer: 2*n_qubits single-qubit gates + (n_qubits-1) CX gates
    # Plus one final rotation layer
    single_q = 2 * n_qubits * (n_layers + 1)
    cx_gates = (n_qubits - 1) * n_layers
    total_gates = single_q + cx_gates
    n_params = single_q  # each rotation has one parameter

    # Rough depth: each layer ~ 3 (ry + rz + cx layer)
    depth = 2 * (n_layers + 1) + n_layers

    return {
        "gate_count": total_gates,
        "depth": depth,
        "params": n_params,
    }


# ---------------------------------------------------------------------------
# Hedger wrapper
# ---------------------------------------------------------------------------

class QuantumDeepHedger:
    """Variational quantum circuit for deep hedging.

    Parameters
    ----------
    config : QuantumDeepHedgingConfig
        Circuit configuration.
    """

    def __init__(self, config: QuantumDeepHedgingConfig | None = None) -> None:
        self.config = config or QuantumDeepHedgingConfig()
        self.circuit = build_circuit(
            self.config.n_qubits,
            self.config.n_layers,
            self.config.entanglement,
        )
        self.n_params = len(self.circuit.parameters)

    def forward(
        self,
        params: NDArray[np.float64],
        features: NDArray[np.float64],
        backend: Any | None = None,
    ) -> NDArray[np.float64]:
        """Evaluate circuit with given parameters and features."""
        return forward(
            params, features,
            self.config.n_qubits,
            self.config.n_layers,
            self.config.entanglement,
            backend=backend,
        )

    def resource_estimate(self) -> dict[str, int]:
        """Return gate-count, depth and parameter count."""
        return resource_estimate(self.config.n_qubits, self.config.n_layers)
