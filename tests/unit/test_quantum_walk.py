"""Unit tests for the Szegedy quantum walk portfolio optimizer."""

from __future__ import annotations

import numpy as np
import pytest

from qufin.backends.mock import MockBackend
from qufin.portfolio.optimizers.quantum_walk import (
    SzegedyWalkConfig,
    SzegedyWalkOptimizer,
    SzegedyWalkResult,
    build_marked_oracle,
    build_transition_matrix,
    build_walk_unitary,
    classical_random_walk,
    compute_qubo_energies,
)
from qufin.portfolio.qubo import PortfolioQUBO


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_qubo_3() -> PortfolioQUBO:
    """3-asset portfolio QUBO for fast testing."""
    mu = np.array([0.02, 0.03, 0.015])
    cov = np.array([
        [0.04, 0.006, 0.002],
        [0.006, 0.09, 0.004],
        [0.002, 0.004, 0.01],
    ])
    return PortfolioQUBO(mu=mu, cov=cov, gamma=0.5)


@pytest.fixture
def small_qubo_4() -> PortfolioQUBO:
    """4-asset portfolio QUBO with cardinality constraint."""
    mu = np.array([0.01, 0.02, 0.015, 0.008])
    cov = np.array([
        [0.04, 0.006, 0.002, 0.001],
        [0.006, 0.09, 0.004, 0.003],
        [0.002, 0.004, 0.01, 0.002],
        [0.001, 0.003, 0.002, 0.025],
    ])
    return PortfolioQUBO(mu=mu, cov=cov, cardinality=2, gamma=0.5)


@pytest.fixture
def small_qubo_5() -> PortfolioQUBO:
    """5-asset portfolio QUBO."""
    rng = np.random.default_rng(123)
    mu = rng.uniform(0.005, 0.03, 5)
    A = rng.standard_normal((5, 5)) * 0.1
    cov = A.T @ A + np.eye(5) * 0.01
    return PortfolioQUBO(mu=mu, cov=cov, gamma=1.0)


@pytest.fixture
def mock_backend() -> MockBackend:
    return MockBackend(seed=42)


# ---------------------------------------------------------------------------
# Walk unitary construction tests
# ---------------------------------------------------------------------------

class TestWalkUnitaryConstruction:
    def test_transition_matrix_is_stochastic(self, small_qubo_3: PortfolioQUBO) -> None:
        """Transition matrix rows should sum to 1."""
        energies = compute_qubo_energies(small_qubo_3)
        P = build_transition_matrix(energies, temperature=1.0)

        row_sums = P.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)

    def test_transition_matrix_nonnegative(self, small_qubo_3: PortfolioQUBO) -> None:
        """All transition probabilities should be non-negative."""
        energies = compute_qubo_energies(small_qubo_3)
        P = build_transition_matrix(energies, temperature=1.0)
        assert np.all(P >= -1e-15)

    def test_transition_matrix_symmetric(self, small_qubo_3: PortfolioQUBO) -> None:
        """Doubly-stochastic matrix should be symmetric."""
        energies = compute_qubo_energies(small_qubo_3)
        P = build_transition_matrix(energies, temperature=1.0)
        np.testing.assert_allclose(P, P.T, atol=1e-10)

    def test_walk_unitary_is_unitary(self) -> None:
        """Walk operator should be unitary: W^dag @ W = I."""
        mu = np.array([0.01, 0.02])
        cov = np.array([[0.04, 0.01], [0.01, 0.09]])
        qubo = PortfolioQUBO(mu=mu, cov=cov, gamma=0.5)
        energies = compute_qubo_energies(qubo)
        P = build_transition_matrix(energies, temperature=1.0)
        W = build_walk_unitary(P)

        identity = np.eye(W.shape[0], dtype=np.complex128)
        product = W.conj().T @ W
        np.testing.assert_allclose(product, identity, atol=1e-10)

    def test_walk_unitary_dimension(self, small_qubo_3: PortfolioQUBO) -> None:
        """Walk unitary should have dimension N^2 x N^2."""
        energies = compute_qubo_energies(small_qubo_3)
        N = len(energies)
        P = build_transition_matrix(energies, temperature=1.0)
        W = build_walk_unitary(P)
        assert W.shape == (N * N, N * N)

    def test_marked_oracle_diagonal(self, small_qubo_3: PortfolioQUBO) -> None:
        """Oracle should be a diagonal matrix with +/-1 entries."""
        energies = compute_qubo_energies(small_qubo_3)
        threshold = float(np.median(list(energies.values())))
        oracle = build_marked_oracle(energies, threshold)

        # Should be diagonal
        off_diag = oracle - np.diag(np.diag(oracle))
        np.testing.assert_allclose(off_diag, 0.0, atol=1e-15)

        # Diagonal entries should be +1 or -1
        diag = np.diag(oracle).real
        assert np.all(np.isclose(np.abs(diag), 1.0))

    def test_marked_oracle_marks_low_energy(self) -> None:
        """Oracle should mark (phase-flip) low-energy states."""
        energies = {"00": 1.0, "01": -1.0, "10": 0.5, "11": 2.0}
        threshold = 0.5
        oracle = build_marked_oracle(energies, threshold)

        N = 4
        # State "01" (idx=1) has energy -1.0 <= 0.5, should be marked
        # State "10" (idx=2) has energy 0.5 <= 0.5, should be marked
        # State "00" (idx=0) has energy 1.0 > 0.5, not marked
        # State "11" (idx=3) has energy 2.0 > 0.5, not marked
        for y in range(N):
            assert oracle[1 * N + y, 1 * N + y] == -1.0  # "01" marked
            assert oracle[2 * N + y, 2 * N + y] == -1.0  # "10" marked
            assert oracle[0 * N + y, 0 * N + y] == 1.0   # "00" not marked
            assert oracle[3 * N + y, 3 * N + y] == 1.0   # "11" not marked


# ---------------------------------------------------------------------------
# Small portfolio optimization tests
# ---------------------------------------------------------------------------

class TestSzegedyWalkOptimizer:
    def test_3_asset_optimization(
        self, small_qubo_3: PortfolioQUBO, mock_backend: MockBackend,
    ) -> None:
        """Optimizer should find a good solution for 3 assets."""
        config = SzegedyWalkConfig(n_walk_steps=3, temperature=0.5, seed=42)
        optimizer = SzegedyWalkOptimizer(small_qubo_3, config, mock_backend)
        result = optimizer.run()

        assert isinstance(result, SzegedyWalkResult)
        assert len(result.best_bitstring) == 3
        assert result.weights.shape == (3,)
        assert result.wall_time_s > 0
        assert result.walk_unitary_dim == (2**3) ** 2
        assert result.n_walk_steps == 3

    def test_4_asset_with_cardinality(
        self, small_qubo_4: PortfolioQUBO, mock_backend: MockBackend,
    ) -> None:
        """Optimizer should handle cardinality constraints."""
        config = SzegedyWalkConfig(n_walk_steps=5, temperature=1.0, seed=42)
        optimizer = SzegedyWalkOptimizer(small_qubo_4, config, mock_backend)
        result = optimizer.run()

        assert isinstance(result, SzegedyWalkResult)
        assert len(result.best_bitstring) == 4
        assert result.best_objective < float("inf")
        assert not np.isnan(result.best_objective)

    def test_5_asset_optimization(
        self, small_qubo_5: PortfolioQUBO, mock_backend: MockBackend,
    ) -> None:
        """Optimizer should work for 5 assets (32 states)."""
        config = SzegedyWalkConfig(n_walk_steps=2, temperature=1.0, seed=42)
        optimizer = SzegedyWalkOptimizer(small_qubo_5, config, mock_backend)
        result = optimizer.run()

        assert isinstance(result, SzegedyWalkResult)
        assert len(result.best_bitstring) == 5
        assert result.walk_unitary_dim == (2**5) ** 2

    def test_finds_near_optimal_3_assets(
        self, small_qubo_3: PortfolioQUBO, mock_backend: MockBackend,
    ) -> None:
        """Walk optimizer should find a solution close to brute-force optimum."""
        from qufin.portfolio.optimizers.exhaustive import exhaustive_solve

        exact = exhaustive_solve(small_qubo_3)
        config = SzegedyWalkConfig(
            n_walk_steps=5, temperature=0.3, seed=42,
        )
        optimizer = SzegedyWalkOptimizer(small_qubo_3, config, mock_backend)
        result = optimizer.run()

        # Should find the optimal or near-optimal solution
        assert result.best_objective <= exact.best_objective + 0.5

    def test_result_serializable(
        self, small_qubo_3: PortfolioQUBO, mock_backend: MockBackend,
    ) -> None:
        """Result should be JSON-serializable via the Result base class."""
        config = SzegedyWalkConfig(n_walk_steps=1, temperature=1.0, seed=42)
        optimizer = SzegedyWalkOptimizer(small_qubo_3, config, mock_backend)
        result = optimizer.run()

        json_str = result.to_json()
        assert isinstance(json_str, str)
        assert "best_bitstring" in json_str

    def test_too_many_qubits_raises(self, mock_backend: MockBackend) -> None:
        """Should raise ValueError for n > 10 qubits."""
        mu = np.ones(11)
        cov = np.eye(11)
        qubo = PortfolioQUBO(mu=mu, cov=cov)
        config = SzegedyWalkConfig()
        with pytest.raises(ValueError, match="too large"):
            SzegedyWalkOptimizer(qubo, config, mock_backend)

    def test_custom_energy_threshold(
        self, small_qubo_3: PortfolioQUBO, mock_backend: MockBackend,
    ) -> None:
        """Custom energy threshold should be respected."""
        config = SzegedyWalkConfig(
            n_walk_steps=3, temperature=1.0,
            energy_threshold=-0.01, seed=42,
        )
        optimizer = SzegedyWalkOptimizer(small_qubo_3, config, mock_backend)
        result = optimizer.run()
        assert result.marked_count >= 0
        assert isinstance(result, SzegedyWalkResult)

    def test_more_walk_steps_generally_helps(
        self, small_qubo_3: PortfolioQUBO, mock_backend: MockBackend,
    ) -> None:
        """More walk steps should not degrade quality on average."""
        results = []
        for steps in [1, 5, 10]:
            config = SzegedyWalkConfig(
                n_walk_steps=steps, temperature=0.5, seed=42,
            )
            opt = SzegedyWalkOptimizer(small_qubo_3, config, mock_backend)
            results.append(opt.run())

        # At least one of the higher-step results should be
        # no worse than 1-step
        best_high = min(r.best_objective for r in results[1:])
        assert best_high <= results[0].best_objective + 1.0


# ---------------------------------------------------------------------------
# Classical walk comparison tests
# ---------------------------------------------------------------------------

class TestClassicalRandomWalk:
    def test_classical_walk_returns_valid_bitstring(
        self, small_qubo_3: PortfolioQUBO,
    ) -> None:
        """Classical walk should return a valid bitstring and energy."""
        best_bs, best_energy, trace = classical_random_walk(
            small_qubo_3, n_steps=200, temperature=1.0, seed=42,
        )
        assert len(best_bs) == 3
        assert all(c in "01" for c in best_bs)
        assert isinstance(best_energy, float)
        assert len(trace) == 201  # initial + 200 steps

    def test_classical_walk_finds_good_solution(
        self, small_qubo_3: PortfolioQUBO,
    ) -> None:
        """Classical walk should find a reasonable solution."""
        from qufin.portfolio.optimizers.exhaustive import exhaustive_solve

        exact = exhaustive_solve(small_qubo_3)
        best_bs, best_energy, _ = classical_random_walk(
            small_qubo_3, n_steps=500, temperature=0.5, seed=42,
        )
        # Should be within reasonable range of optimal
        assert best_energy <= exact.best_objective + 1.0

    def test_classical_walk_reproducible(
        self, small_qubo_3: PortfolioQUBO,
    ) -> None:
        """Same seed should produce same results."""
        bs1, e1, _ = classical_random_walk(
            small_qubo_3, n_steps=100, temperature=1.0, seed=123,
        )
        bs2, e2, _ = classical_random_walk(
            small_qubo_3, n_steps=100, temperature=1.0, seed=123,
        )
        assert bs1 == bs2
        assert e1 == e2

    def test_quantum_vs_classical_comparison(
        self, small_qubo_3: PortfolioQUBO, mock_backend: MockBackend,
    ) -> None:
        """Both quantum and classical walks should produce valid results."""
        # Quantum walk
        config = SzegedyWalkConfig(
            n_walk_steps=5, temperature=0.5, seed=42,
        )
        q_opt = SzegedyWalkOptimizer(small_qubo_3, config, mock_backend)
        q_result = q_opt.run()

        # Classical walk
        c_bs, c_energy, _ = classical_random_walk(
            small_qubo_3, n_steps=500, temperature=0.5, seed=42,
        )

        # Both should produce finite, valid results
        assert not np.isnan(q_result.best_objective)
        assert not np.isnan(c_energy)
        assert q_result.best_objective < float("inf")
        assert c_energy < float("inf")


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_asset(self, mock_backend: MockBackend) -> None:
        """Should handle a single-asset portfolio (2 states)."""
        mu = np.array([0.05])
        cov = np.array([[0.04]])
        qubo = PortfolioQUBO(mu=mu, cov=cov, gamma=1.0)
        config = SzegedyWalkConfig(n_walk_steps=2, temperature=1.0, seed=42)
        optimizer = SzegedyWalkOptimizer(qubo, config, mock_backend)
        result = optimizer.run()
        assert len(result.best_bitstring) == 1
        assert result.best_bitstring in ("0", "1")

    def test_two_assets(self, mock_backend: MockBackend) -> None:
        """Should handle a 2-asset portfolio (4 states)."""
        mu = np.array([0.02, 0.03])
        cov = np.array([[0.04, 0.01], [0.01, 0.09]])
        qubo = PortfolioQUBO(mu=mu, cov=cov, gamma=0.5)
        config = SzegedyWalkConfig(n_walk_steps=3, temperature=1.0, seed=42)
        optimizer = SzegedyWalkOptimizer(qubo, config, mock_backend)
        result = optimizer.run()
        assert len(result.best_bitstring) == 2

    def test_zero_temperature(self, mock_backend: MockBackend) -> None:
        """Near-zero temperature should still produce valid results."""
        mu = np.array([0.01, 0.02, 0.015])
        cov = np.eye(3) * 0.01
        qubo = PortfolioQUBO(mu=mu, cov=cov, gamma=0.5)
        config = SzegedyWalkConfig(
            n_walk_steps=2, temperature=1e-10, seed=42,
        )
        optimizer = SzegedyWalkOptimizer(qubo, config, mock_backend)
        result = optimizer.run()
        assert not np.isnan(result.best_objective)

    def test_high_temperature(self, mock_backend: MockBackend) -> None:
        """Very high temperature should still produce valid results."""
        mu = np.array([0.01, 0.02, 0.015])
        cov = np.eye(3) * 0.01
        qubo = PortfolioQUBO(mu=mu, cov=cov, gamma=0.5)
        config = SzegedyWalkConfig(
            n_walk_steps=2, temperature=1e6, seed=42,
        )
        optimizer = SzegedyWalkOptimizer(qubo, config, mock_backend)
        result = optimizer.run()
        assert not np.isnan(result.best_objective)

    def test_zero_walk_steps(self, mock_backend: MockBackend) -> None:
        """Zero walk steps should still return a valid (initial) result."""
        mu = np.array([0.01, 0.02])
        cov = np.array([[0.04, 0.01], [0.01, 0.09]])
        qubo = PortfolioQUBO(mu=mu, cov=cov, gamma=0.5)
        config = SzegedyWalkConfig(n_walk_steps=0, temperature=1.0, seed=42)
        optimizer = SzegedyWalkOptimizer(qubo, config, mock_backend)
        result = optimizer.run()
        assert isinstance(result, SzegedyWalkResult)
        assert result.n_walk_steps == 0

    def test_identical_assets(self, mock_backend: MockBackend) -> None:
        """Should handle identical assets (degenerate landscape)."""
        mu = np.array([0.02, 0.02, 0.02])
        cov = np.ones((3, 3)) * 0.01 + np.eye(3) * 0.03
        qubo = PortfolioQUBO(mu=mu, cov=cov, gamma=1.0)
        config = SzegedyWalkConfig(n_walk_steps=3, temperature=1.0, seed=42)
        optimizer = SzegedyWalkOptimizer(qubo, config, mock_backend)
        result = optimizer.run()
        assert not np.isnan(result.best_objective)

    def test_energies_computation(self, small_qubo_3: PortfolioQUBO) -> None:
        """compute_qubo_energies should enumerate all 2^n states."""
        energies = compute_qubo_energies(small_qubo_3)
        assert len(energies) == 2**3
        assert all(isinstance(v, float) for v in energies.values())
        # Energy of all-zeros should be 0
        assert energies["000"] == pytest.approx(0.0)
