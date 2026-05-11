"""Property-based tests for portfolio optimization."""

from __future__ import annotations

import numpy as np
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from qufin.portfolio.encodings import decode_binary, decode_one_hot
from qufin.portfolio.qubo import PortfolioQUBO

n_assets_st = st.integers(min_value=3, max_value=10)


@given(n=n_assets_st, seed=st.integers(min_value=0, max_value=10000))
@settings(max_examples=50)
def test_qubo_all_zeros_is_zero(n: int, seed: int) -> None:
    """All-zeros bitstring always evaluates to 0."""
    rng = np.random.default_rng(seed)
    mu = rng.normal(0.001, 0.01, n)
    A = rng.normal(size=(100, n))
    cov = np.cov(A, rowvar=False)
    qubo = PortfolioQUBO(mu=mu, cov=cov)
    assert qubo.evaluate("0" * n) == 0.0


@given(n=n_assets_st, seed=st.integers(min_value=0, max_value=10000))
@settings(max_examples=50)
def test_qubo_matrix_symmetric(n: int, seed: int) -> None:
    """QUBO matrix Q is symmetric (Q_ij = Q_ji)."""
    rng = np.random.default_rng(seed)
    mu = rng.normal(0.001, 0.01, n)
    A = rng.normal(size=(100, n))
    cov = np.cov(A, rowvar=False)
    qubo = PortfolioQUBO(mu=mu, cov=cov, gamma=1.0)
    Q = qubo.build_matrix()
    np.testing.assert_array_almost_equal(Q, Q.T)


@given(n=n_assets_st, seed=st.integers(min_value=0, max_value=10000))
@settings(max_examples=50)
def test_qubo_evaluate_deterministic(n: int, seed: int) -> None:
    """Same bitstring always gives same value."""
    rng = np.random.default_rng(seed)
    mu = rng.normal(0.001, 0.01, n)
    A = rng.normal(size=(100, n))
    cov = np.cov(A, rowvar=False)
    qubo = PortfolioQUBO(mu=mu, cov=cov)
    bs = "1" * (n // 2) + "0" * (n - n // 2)
    assert qubo.evaluate(bs) == qubo.evaluate(bs)


@given(n=n_assets_st, seed=st.integers(min_value=0, max_value=10000))
@settings(max_examples=50)
def test_qubo_with_cardinality_symmetric(n: int, seed: int) -> None:
    """QUBO with cardinality constraint is still symmetric."""
    rng = np.random.default_rng(seed)
    mu = rng.normal(0.001, 0.01, n)
    A = rng.normal(size=(100, n))
    cov = np.cov(A, rowvar=False)
    K = max(1, n // 3)
    qubo = PortfolioQUBO(mu=mu, cov=cov, cardinality=K)
    Q = qubo.build_matrix()
    np.testing.assert_array_almost_equal(Q, Q.T)


@given(n=n_assets_st, seed=st.integers(min_value=0, max_value=10000))
@settings(max_examples=30)
def test_cardinality_penalty_correct_k_is_lower(n: int, seed: int) -> None:
    """With cardinality K, selecting exactly K assets should cost less
    than selecting K+1 (assuming penalty is large enough)."""
    rng = np.random.default_rng(seed)
    mu = rng.uniform(0.001, 0.01, n)
    A = rng.normal(size=(100, n))
    cov = np.cov(A, rowvar=False)
    K = max(1, n // 3)
    assume(n > K + 1)
    qubo = PortfolioQUBO(mu=mu, cov=cov, cardinality=K, gamma=0.1)
    bs_k = "1" * K + "0" * (n - K)
    bs_k1 = "1" * (K + 1) + "0" * (n - K - 1)
    assert qubo.evaluate(bs_k) < qubo.evaluate(bs_k1)


@given(n=st.integers(min_value=2, max_value=6), seed=st.integers(min_value=0, max_value=1000))
@settings(max_examples=30)
def test_decode_one_hot_weights_sum_to_one(n: int, seed: int) -> None:
    """Decoded one-hot weights sum to 1 (when at least one selected)."""
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, n)
    assume(bits.sum() > 0)
    bs = "".join(str(b) for b in bits)
    w = decode_one_hot(bs)
    assert abs(w.sum() - 1.0) < 1e-10


@given(
    n=st.integers(min_value=2, max_value=5),
    bits=st.integers(min_value=2, max_value=4),
    seed=st.integers(min_value=0, max_value=1000),
)
@settings(max_examples=30)
def test_decode_binary_weights_sum_to_one(n: int, bits: int, seed: int) -> None:
    """Decoded binary weights sum to 1 (when not all zeros)."""
    rng = np.random.default_rng(seed)
    total_qubits = n * bits
    bitvals = rng.integers(0, 2, total_qubits)
    assume(bitvals.sum() > 0)
    bs = "".join(str(b) for b in bitvals)
    w = decode_binary(bs, n_assets=n, bits_per_asset=bits)
    assert abs(w.sum() - 1.0) < 1e-10


@given(n=n_assets_st, seed=st.integers(min_value=0, max_value=10000))
@settings(max_examples=30)
def test_binary_qubo_matrix_symmetric(n: int, seed: int) -> None:
    """Binary-encoded QUBO matrix is symmetric."""
    rng = np.random.default_rng(seed)
    mu = rng.normal(0.001, 0.01, n)
    A = rng.normal(size=(100, n))
    cov = np.cov(A, rowvar=False)
    qubo = PortfolioQUBO(mu=mu, cov=cov, encoding="binary", bits_per_asset=2)
    Q = qubo.build_matrix()
    np.testing.assert_array_almost_equal(Q, Q.T)
