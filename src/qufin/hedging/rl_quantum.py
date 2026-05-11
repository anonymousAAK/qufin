"""Quantum RL policy networks for hedging.

Provides a parametrised quantum circuit acting as a stochastic policy
for REINFORCE-style reinforcement learning.  The circuit maps a
classical state (encoded via angle embedding) to action probabilities
derived from measurement outcome distributions.
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
class QuantumPolicyConfig:
    """Configuration for the quantum policy circuit.

    Attributes
    ----------
    n_qubits : int
        Number of qubits in the policy circuit.
    n_layers : int
        Number of variational layers.
    n_actions : int
        Size of the discrete action space.
    """

    n_qubits: int = 4
    n_layers: int = 2
    n_actions: int = 3


# ---------------------------------------------------------------------------
# Circuit builder
# ---------------------------------------------------------------------------

def build_policy_circuit(
    n_qubits: int,
    n_layers: int,
    n_actions: int,
) -> Any:
    """Build a variational policy circuit.

    The circuit consists of:
    1. A placeholder for angle-encoded state features (applied at runtime).
    2. An ``EfficientSU2``-style trainable ansatz.
    3. Measurement on the first ``ceil(log2(n_actions))`` qubits.

    Parameters
    ----------
    n_qubits : int
        Total qubits.
    n_layers : int
        Variational reps.
    n_actions : int
        Number of discrete actions.

    Returns
    -------
    qiskit.circuit.QuantumCircuit
        Parametrised circuit (without measurements; those are handled
        at evaluation time).
    """
    from qiskit.circuit.library import TwoLocal

    ansatz = TwoLocal(
        num_qubits=n_qubits,
        rotation_blocks=["ry", "rz"],
        entanglement_blocks="cx",
        entanglement="linear",
        reps=n_layers,
        insert_barriers=False,
    )
    return ansatz


def _encode_state(n_qubits: int, state: NDArray[np.float64]) -> Any:
    """Angle-encode a classical state vector into RY rotations."""
    from qiskit.circuit import QuantumCircuit

    qc = QuantumCircuit(n_qubits)
    for i, s in enumerate(state[:n_qubits]):
        qc.ry(float(s), i)
    return qc


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

class QuantumPolicy:
    """Parametrised quantum circuit as a discrete-action policy.

    Parameters
    ----------
    config : QuantumPolicyConfig
        Policy configuration.
    """

    def __init__(self, config: QuantumPolicyConfig | None = None) -> None:
        self.config = config or QuantumPolicyConfig()
        self.ansatz = build_policy_circuit(
            self.config.n_qubits,
            self.config.n_layers,
            self.config.n_actions,
        )
        self.n_params = len(self.ansatz.parameters)
        self._n_measure = max(1, int(np.ceil(np.log2(self.config.n_actions))))

    # ------------------------------------------------------------------
    def select_action(
        self,
        state: NDArray[np.float64],
        params: NDArray[np.float64],
        backend: Any | None = None,
    ) -> NDArray[np.float64]:
        """Return action probabilities for the given state.

        Parameters
        ----------
        state : NDArray
            1-D classical state vector (length <= *n_qubits*).
        params : NDArray
            Flat variational parameter vector.
        backend : object or None
            Qiskit backend; ``None`` uses exact statevector simulation.

        Returns
        -------
        NDArray
            Action probability vector of length *n_actions*.
        """
        from qiskit.circuit import QuantumCircuit
        from qiskit.quantum_info import Statevector

        n_q = self.config.n_qubits
        n_actions = self.config.n_actions

        # Build circuit
        encoding = _encode_state(n_q, state)
        qc = QuantumCircuit(n_q)
        qc.compose(encoding, inplace=True)
        qc.compose(self.ansatz, inplace=True)

        # Bind parameters
        param_symbols = list(self.ansatz.parameters)
        if len(params) != len(param_symbols):
            raise ValueError(
                f"Expected {len(param_symbols)} params, got {len(params)}"
            )
        bind_dict = dict(zip(param_symbols, params.tolist(), strict=False))
        qc.assign_parameters(bind_dict, inplace=True)

        if backend is not None:
            # Shot-based evaluation
            from qiskit import transpile

            meas_qc = qc.copy()
            meas_qc.measure_all()
            transpiled = transpile(meas_qc, backend)
            job = backend.run(transpiled, shots=1024)
            counts = job.result().get_counts()
            probs = self._counts_to_action_probs(counts, n_actions)
        else:
            # Exact statevector
            sv = Statevector.from_instruction(qc)
            full_probs = sv.probabilities()
            probs = self._probs_to_action_probs(full_probs, n_actions)

        return probs

    # ------------------------------------------------------------------
    def _probs_to_action_probs(
        self, full_probs: NDArray, n_actions: int,
    ) -> NDArray[np.float64]:
        """Coarse-grain statevector probabilities into action probs."""
        n_states = len(full_probs)
        action_probs = np.zeros(n_actions)

        # Map each computational basis state to an action by modular indexing
        for idx in range(n_states):
            action = idx % n_actions
            action_probs[action] += full_probs[idx]

        # Normalise (should already sum to 1, but be safe)
        total = action_probs.sum()
        if total > 0:
            action_probs /= total
        else:
            action_probs[:] = 1.0 / n_actions

        return action_probs

    def _counts_to_action_probs(
        self, counts: dict[str, int], n_actions: int,
    ) -> NDArray[np.float64]:
        """Convert shot counts to action probabilities."""
        action_counts = np.zeros(n_actions)
        total_shots = sum(counts.values())

        for bitstring, count in counts.items():
            idx = int(bitstring.replace(" ", ""), 2)
            action = idx % n_actions
            action_counts[action] += count

        probs = action_counts / total_shots
        return probs

    # ------------------------------------------------------------------
    def log_prob(
        self,
        state: NDArray[np.float64],
        action: int,
        params: NDArray[np.float64],
        backend: Any | None = None,
    ) -> float:
        """Log-probability of a specific action (for REINFORCE gradient).

        Parameters
        ----------
        state : NDArray
            State vector.
        action : int
            Chosen action index.
        params : NDArray
            Variational parameters.
        backend : object or None
            Qiskit backend.

        Returns
        -------
        float
            ``log pi(action | state)``.
        """
        probs = self.select_action(state, params, backend)
        p = float(probs[action])
        return float(np.log(max(p, 1e-12)))
