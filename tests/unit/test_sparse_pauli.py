"""Unit tests for sparse Pauli operator construction."""

from __future__ import annotations

import numpy as np

from qufin.utils.sparse_pauli import (
    diagonal_expectation,
    qubo_to_sparse_pauli,
    sparse_pauli_to_matrix,
)


def _simple_qubo(n: int = 3) -> np.ndarray:
    """Create a simple QUBO matrix for testing."""
    Q = np.zeros((n, n))
    for i in range(n):
        Q[i, i] = -(i + 1) * 0.1
    for i in range(n - 1):
        Q[i, i + 1] = 0.05
    return Q


class TestQuboToSparsePauli:
    def test_returns_sparse_pauli_op(self) -> None:
        from qiskit.quantum_info import SparsePauliOp

        Q = _simple_qubo(3)
        op = qubo_to_sparse_pauli(Q)
        assert isinstance(op, SparsePauliOp)

    def test_correct_num_qubits(self) -> None:
        Q = _simple_qubo(4)
        op = qubo_to_sparse_pauli(Q)
        assert op.num_qubits == 4

    def test_identity_qubo(self) -> None:
        """Diagonal QUBO should produce Z and I terms only."""
        Q = np.diag([1.0, 2.0])
        op = qubo_to_sparse_pauli(Q)
        labels = [str(p) for p in op.paulis]
        for label in labels:
            assert all(c in ("I", "Z") for c in label)

    def test_zero_matrix(self) -> None:
        Q = np.zeros((2, 2))
        op = qubo_to_sparse_pauli(Q)
        # Should be zero operator or empty
        mat = sparse_pauli_to_matrix(op)
        np.testing.assert_allclose(mat, np.zeros((4, 4)), atol=1e-12)

    def test_single_qubit(self) -> None:
        Q = np.array([[1.0]])
        op = qubo_to_sparse_pauli(Q)
        assert op.num_qubits == 1

    def test_symmetric_qubo(self) -> None:
        """Symmetric QUBO and upper-triangular should give same Hamiltonian."""
        Q_upper = np.array([[1.0, 0.5], [0.0, 2.0]])
        Q_sym = np.array([[1.0, 0.25], [0.25, 2.0]])
        op1 = qubo_to_sparse_pauli(Q_upper)
        op2 = qubo_to_sparse_pauli(Q_sym)
        m1 = sparse_pauli_to_matrix(op1)
        m2 = sparse_pauli_to_matrix(op2)
        np.testing.assert_allclose(m1, m2, atol=1e-12)

    def test_energy_matches_brute_force(self) -> None:
        """Verify that diagonal elements match QUBO evaluation for all bitstrings."""
        Q = np.array([[1.0, 0.3], [0.3, -0.5]])
        op = qubo_to_sparse_pauli(Q)
        mat = sparse_pauli_to_matrix(op)

        for b in range(4):
            # Qiskit uses little-endian ordering: index b corresponds
            # to reversed bitstring for qubit assignment
            bs_le = format(b, "02b")  # little-endian label
            bs_be = bs_le[::-1]  # big-endian for QUBO evaluation
            x = np.array([int(c) for c in bs_be], dtype=float)
            qubo_val = x @ Q @ x
            ham_val = np.real(mat[b, b])
            assert abs(qubo_val - ham_val) < 1e-10, f"Mismatch for {bs_le}"

    def test_hermitian(self) -> None:
        Q = _simple_qubo(3)
        op = qubo_to_sparse_pauli(Q)
        mat = sparse_pauli_to_matrix(op)
        np.testing.assert_allclose(mat, mat.conj().T, atol=1e-12)


class TestSparsePauliToMatrix:
    def test_returns_ndarray(self) -> None:
        Q = _simple_qubo(2)
        op = qubo_to_sparse_pauli(Q)
        mat = sparse_pauli_to_matrix(op)
        assert isinstance(mat, np.ndarray)

    def test_shape(self) -> None:
        Q = _simple_qubo(3)
        op = qubo_to_sparse_pauli(Q)
        mat = sparse_pauli_to_matrix(op)
        assert mat.shape == (8, 8)

    def test_dtype(self) -> None:
        Q = _simple_qubo(2)
        op = qubo_to_sparse_pauli(Q)
        mat = sparse_pauli_to_matrix(op)
        assert np.issubdtype(mat.dtype, np.complexfloating)


class TestDiagonalExpectation:
    def test_all_zeros(self) -> None:
        Q = np.array([[1.0, 0.0], [0.0, 2.0]])
        op = qubo_to_sparse_pauli(Q)
        val = diagonal_expectation(op, "00")
        # x = [0, 0], QUBO = 0
        assert abs(val - 0.0) < 1e-10

    def test_all_ones(self) -> None:
        Q = np.array([[1.0, 0.0], [0.0, 2.0]])
        op = qubo_to_sparse_pauli(Q)
        val = diagonal_expectation(op, "11")
        # x = [1, 1], QUBO = 1 + 2 = 3
        assert abs(val - 3.0) < 1e-10

    def test_matches_qubo(self) -> None:
        Q = np.array([[0.5, 0.2], [0.2, -0.3]])
        op = qubo_to_sparse_pauli(Q)
        for bs in ["00", "01", "10", "11"]:
            x = np.array([int(c) for c in bs], dtype=float)
            expected = float(x @ Q @ x)
            actual = diagonal_expectation(op, bs)
            assert abs(actual - expected) < 1e-10, f"Mismatch for {bs}"

    def test_three_qubits(self) -> None:
        Q = _simple_qubo(3)
        op = qubo_to_sparse_pauli(Q)
        x = np.array([1, 0, 1], dtype=float)
        expected = float(x @ Q @ x)
        actual = diagonal_expectation(op, "101")
        assert abs(actual - expected) < 1e-10
