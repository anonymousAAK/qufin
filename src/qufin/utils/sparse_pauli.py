"""Efficient sparse Pauli operator construction for QUBO Hamiltonians.

Builds Hamiltonians using Qiskit's ``SparsePauliOp`` representation,
avoiding dense matrix construction for large problems.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def qubo_to_sparse_pauli(Q: NDArray[np.float64]) -> Any:
    """Convert a QUBO matrix to a SparsePauliOp Hamiltonian.

    Maps the QUBO objective ``x^T Q x`` to an Ising Hamiltonian
    using the substitution ``x_i = (1 - Z_i) / 2``.

    Parameters
    ----------
    Q : ndarray, shape (n, n)
        QUBO matrix (upper-triangular or symmetric).

    Returns
    -------
    SparsePauliOp
        Sparse Pauli representation of the Ising Hamiltonian.
    """
    from qiskit.quantum_info import SparsePauliOp

    n = Q.shape[0]
    labels: list[str] = []
    coeffs: list[complex] = []

    identity = "I" * n
    offset = 0.0

    # Diagonal terms: Q_ii * x_i = Q_ii * (1 - Z_i) / 2
    for i in range(n):
        if abs(Q[i, i]) < 1e-15:
            continue
        offset += Q[i, i] / 2.0
        label = list(identity)
        label[n - 1 - i] = "Z"
        labels.append("".join(label))
        coeffs.append(-Q[i, i] / 2.0)

    # Off-diagonal terms: Q_ij * x_i * x_j = Q_ij * (1 - Z_i)(1 - Z_j) / 4
    for i in range(n):
        for j in range(i + 1, n):
            q_ij = Q[i, j] + Q[j, i]
            if abs(q_ij) < 1e-15:
                continue
            offset += q_ij / 4.0
            # -Z_i / 4
            label_i = list(identity)
            label_i[n - 1 - i] = "Z"
            labels.append("".join(label_i))
            coeffs.append(-q_ij / 4.0)
            # -Z_j / 4
            label_j = list(identity)
            label_j[n - 1 - j] = "Z"
            labels.append("".join(label_j))
            coeffs.append(-q_ij / 4.0)
            # +Z_i Z_j / 4
            label_ij = list(identity)
            label_ij[n - 1 - i] = "Z"
            label_ij[n - 1 - j] = "Z"
            labels.append("".join(label_ij))
            coeffs.append(q_ij / 4.0)

    # Add identity (offset) term
    labels.append(identity)
    coeffs.append(offset)

    op = SparsePauliOp(labels, np.array(coeffs, dtype=complex))
    return op.simplify()


def sparse_pauli_to_matrix(op: Any) -> NDArray[np.complex128]:
    """Convert a SparsePauliOp to its dense matrix representation.

    Useful for small problems where dense simulation is desired.
    """
    mat = op.to_matrix()
    # Handle both sparse and dense returns across Qiskit versions
    if hasattr(mat, "todense"):
        return np.array(mat.todense())
    return np.asarray(mat, dtype=np.complex128)


def diagonal_expectation(
    op: Any, bitstring: str,
) -> float:
    """Evaluate expectation of a diagonal (Z-only) operator on a bitstring.

    Parameters
    ----------
    op : SparsePauliOp
        Operator (must be diagonal / Z-only + identity).
    bitstring : str
        Computational basis state, e.g. ``"0110"``.

    Returns
    -------
    float
        The expectation value.
    """
    n = len(bitstring)
    total = 0.0
    for label, coeff in zip(op.paulis.to_labels(), op.coeffs, strict=True):
        val = 1.0
        for q in range(n):
            pauli_char = label[n - 1 - q]
            if pauli_char == "Z":
                bit = int(bitstring[q])
                val *= 1 - 2 * bit  # |0> -> +1, |1> -> -1
            elif pauli_char != "I":
                # Non-diagonal term — skip
                val = 0.0
                break
        total += float(np.real(coeff)) * val
    return total
