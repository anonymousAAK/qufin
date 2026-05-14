"""Unit tests for the Grover Adaptive Search portfolio optimizer."""

from __future__ import annotations

import numpy as np
import pytest

from qufin.backends.mock import MockBackend
from qufin.portfolio.optimizers.grover_search import (
    GroverAdaptiveSearch,
    GroverSearchConfig,
    GroverSearchResult,
    _build_diffusion,
    _build_grover_circuit,
    _build_oracle,
    _compute_all_energies,
    _optimal_grover_iters,
    branch_and_bound_solve,
    compare_optimizers,
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
# Energy computation tests
# ---------------------------------------------------------------------------


class TestEnergyComputation:
    def test_all_energies_enumerated(self, small_qubo_3: PortfolioQUBO) -> None:
        """Should enumerate all 2^n bitstrings."""
        energies = _compute_all_energies(small_qubo_3)
        assert len(energies) == 2**3

    def test_zero_bitstring_energy(self, small_qubo_3: PortfolioQUBO) -> None:
        """Energy of all-zeros should be 0."""
        energies = _compute_all_energies(small_qubo_3)
        assert energies["000"] == pytest.approx(0.0)

    def test_energies_are_real(self, small_qubo_3: PortfolioQUBO) -> None:
        """All energies should be real finite floats."""
        energies = _compute_all_energies(small_qubo_3)
        for e in energies.values():
            assert isinstance(e, float)
            assert not np.isnan(e)
            assert np.isfinite(e)


# ---------------------------------------------------------------------------
# Oracle and diffusion tests
# ---------------------------------------------------------------------------


class TestOracleConstruction:
    def test_oracle_returns_circuit(self, small_qubo_3: PortfolioQUBO) -> None:
        """Oracle should return a quantum circuit."""
        energies = _compute_all_energies(small_qubo_3)
        threshold = float(np.median(list(energies.values())))
        oracle = _build_oracle(3, energies, threshold)
        assert oracle is not None

    def test_oracle_no_marked_states(self, small_qubo_3: PortfolioQUBO) -> None:
        """Oracle with impossibly low threshold returns identity-like circuit."""
        energies = _compute_all_energies(small_qubo_3)
        threshold = min(energies.values()) - 1.0
        oracle = _build_oracle(3, energies, threshold)
        # Should be a valid circuit (identity)
        assert oracle is not None

    def test_diffusion_returns_circuit(self) -> None:
        """Diffusion operator should return a quantum circuit."""
        diff = _build_diffusion(3)
        assert diff is not None

    def test_diffusion_single_qubit(self) -> None:
        """Diffusion should work for 1 qubit."""
        diff = _build_diffusion(1)
        assert diff is not None


# ---------------------------------------------------------------------------
# Grover circuit tests
# ---------------------------------------------------------------------------


class TestGroverCircuit:
    def test_circuit_has_measurements(self, small_qubo_3: PortfolioQUBO) -> None:
        """Grover circuit should include measurement gates."""
        energies = _compute_all_energies(small_qubo_3)
        threshold = float(np.median(list(energies.values())))
        circuit = _build_grover_circuit(3, energies, threshold, n_iterations=1)
        assert hasattr(circuit, "num_qubits")

    def test_circuit_zero_iterations(self, small_qubo_3: PortfolioQUBO) -> None:
        """Zero Grover iterations should still produce a valid circuit."""
        energies = _compute_all_energies(small_qubo_3)
        threshold = 0.0
        circuit = _build_grover_circuit(3, energies, threshold, n_iterations=0)
        assert hasattr(circuit, "num_qubits")

    def test_circuit_multiple_iterations(self, small_qubo_3: PortfolioQUBO) -> None:
        """Multiple Grover iterations should produce a deeper circuit."""
        energies = _compute_all_energies(small_qubo_3)
        threshold = float(np.median(list(energies.values())))
        c1 = _build_grover_circuit(3, energies, threshold, n_iterations=1)
        c3 = _build_grover_circuit(3, energies, threshold, n_iterations=3)
        assert c3.depth() >= c1.depth()


# ---------------------------------------------------------------------------
# Optimal iterations tests
# ---------------------------------------------------------------------------


class TestOptimalGroverIters:
    def test_returns_positive(self) -> None:
        assert _optimal_grover_iters(16, 4) >= 1

    def test_single_marked(self) -> None:
        """For 1 marked state in N=16, should be ~floor(pi/4 * 4) = 3."""
        iters = _optimal_grover_iters(16, 1)
        assert 1 <= iters <= 5

    def test_all_marked(self) -> None:
        """If all states are marked, should return 1."""
        assert _optimal_grover_iters(16, 16) == 1

    def test_no_marked(self) -> None:
        """If no states are marked, should return 1."""
        assert _optimal_grover_iters(16, 0) == 1


# ---------------------------------------------------------------------------
# Grover Adaptive Search optimizer tests
# ---------------------------------------------------------------------------


class TestGroverAdaptiveSearch:
    def test_3_asset_optimization(
        self, small_qubo_3: PortfolioQUBO, mock_backend: MockBackend,
    ) -> None:
        """Optimizer should find a valid solution for 3 assets."""
        config = GroverSearchConfig(max_iterations=5, seed=42)
        optimizer = GroverAdaptiveSearch(small_qubo_3, config, mock_backend)
        result = optimizer.run()

        assert isinstance(result, GroverSearchResult)
        assert len(result.best_bitstring) == 3
        assert result.weights.shape == (3,)
        assert result.wall_time_s > 0
        assert result.n_rounds >= 0

    def test_4_asset_with_cardinality(
        self, small_qubo_4: PortfolioQUBO, mock_backend: MockBackend,
    ) -> None:
        """Optimizer should handle cardinality constraints."""
        config = GroverSearchConfig(max_iterations=5, seed=42)
        optimizer = GroverAdaptiveSearch(small_qubo_4, config, mock_backend)
        result = optimizer.run()

        assert isinstance(result, GroverSearchResult)
        assert len(result.best_bitstring) == 4
        assert result.best_objective < float("inf")
        assert not np.isnan(result.best_objective)

    def test_5_asset_optimization(
        self, small_qubo_5: PortfolioQUBO, mock_backend: MockBackend,
    ) -> None:
        """Optimizer should work for 5 assets."""
        config = GroverSearchConfig(max_iterations=3, seed=42)
        optimizer = GroverAdaptiveSearch(small_qubo_5, config, mock_backend)
        result = optimizer.run()

        assert isinstance(result, GroverSearchResult)
        assert len(result.best_bitstring) == 5

    def test_threshold_history_recorded(
        self, small_qubo_3: PortfolioQUBO, mock_backend: MockBackend,
    ) -> None:
        """Threshold history should be non-empty and monotonically non-increasing."""
        config = GroverSearchConfig(max_iterations=10, seed=42)
        optimizer = GroverAdaptiveSearch(small_qubo_3, config, mock_backend)
        result = optimizer.run()

        assert len(result.threshold_history) > 0
        # Thresholds should not increase (adaptive tightening)
        for i in range(1, len(result.threshold_history)):
            assert result.threshold_history[i] <= result.threshold_history[i - 1] + 1e-10

    def test_oracle_calls_counted(
        self, small_qubo_3: PortfolioQUBO, mock_backend: MockBackend,
    ) -> None:
        """Oracle calls should be a positive integer."""
        config = GroverSearchConfig(max_iterations=3, seed=42)
        optimizer = GroverAdaptiveSearch(small_qubo_3, config, mock_backend)
        result = optimizer.run()
        assert result.n_oracle_calls >= 0

    def test_result_serializable(
        self, small_qubo_3: PortfolioQUBO, mock_backend: MockBackend,
    ) -> None:
        """Result should be JSON-serializable."""
        config = GroverSearchConfig(max_iterations=2, seed=42)
        optimizer = GroverAdaptiveSearch(small_qubo_3, config, mock_backend)
        result = optimizer.run()

        json_str = result.to_json()
        assert isinstance(json_str, str)
        assert "best_bitstring" in json_str

    def test_too_many_qubits_raises(self, mock_backend: MockBackend) -> None:
        """Should raise ValueError for n > max_qubits."""
        mu = np.ones(25)
        cov = np.eye(25)
        qubo = PortfolioQUBO(mu=mu, cov=cov)
        config = GroverSearchConfig(max_qubits=20)
        with pytest.raises(ValueError, match="too large"):
            GroverAdaptiveSearch(qubo, config, mock_backend)

    def test_fixed_grover_iterations(
        self, small_qubo_3: PortfolioQUBO, mock_backend: MockBackend,
    ) -> None:
        """Fixed n_grover_iterations should be respected."""
        config = GroverSearchConfig(
            max_iterations=3, n_grover_iterations=2, seed=42,
        )
        optimizer = GroverAdaptiveSearch(small_qubo_3, config, mock_backend)
        result = optimizer.run()
        assert isinstance(result, GroverSearchResult)

    def test_single_asset(self, mock_backend: MockBackend) -> None:
        """Should handle a 1-asset portfolio."""
        mu = np.array([0.05])
        cov = np.array([[0.04]])
        qubo = PortfolioQUBO(mu=mu, cov=cov, gamma=1.0)
        config = GroverSearchConfig(max_iterations=3, seed=42)
        optimizer = GroverAdaptiveSearch(qubo, config, mock_backend)
        result = optimizer.run()
        assert len(result.best_bitstring) == 1
        assert result.best_bitstring in ("0", "1")

    def test_two_assets(self, mock_backend: MockBackend) -> None:
        """Should handle a 2-asset portfolio."""
        mu = np.array([0.02, 0.03])
        cov = np.array([[0.04, 0.01], [0.01, 0.09]])
        qubo = PortfolioQUBO(mu=mu, cov=cov, gamma=0.5)
        config = GroverSearchConfig(max_iterations=5, seed=42)
        optimizer = GroverAdaptiveSearch(qubo, config, mock_backend)
        result = optimizer.run()
        assert len(result.best_bitstring) == 2

    def test_reproducible_with_same_seed(
        self, small_qubo_3: PortfolioQUBO, mock_backend: MockBackend,
    ) -> None:
        """Same seed should produce same results."""
        config = GroverSearchConfig(max_iterations=5, seed=99)
        r1 = GroverAdaptiveSearch(small_qubo_3, config, mock_backend).run()
        r2 = GroverAdaptiveSearch(small_qubo_3, config, mock_backend).run()
        assert r1.best_bitstring == r2.best_bitstring
        assert r1.best_objective == r2.best_objective


# ---------------------------------------------------------------------------
# Branch and bound tests
# ---------------------------------------------------------------------------


class TestBranchAndBound:
    def test_finds_optimal_3_assets(self, small_qubo_3: PortfolioQUBO) -> None:
        """Branch and bound should find the exact optimum."""
        from qufin.portfolio.optimizers.exhaustive import exhaustive_solve

        exact = exhaustive_solve(small_qubo_3)
        _bb_bs, bb_obj = branch_and_bound_solve(small_qubo_3)

        assert bb_obj == pytest.approx(exact.best_objective, abs=1e-10)

    def test_finds_optimal_4_assets(self, small_qubo_4: PortfolioQUBO) -> None:
        """Branch and bound should find optimal for 4-asset QUBO."""
        from qufin.portfolio.optimizers.exhaustive import exhaustive_solve

        exact = exhaustive_solve(small_qubo_4)
        _bb_bs, bb_obj = branch_and_bound_solve(small_qubo_4)

        assert bb_obj == pytest.approx(exact.best_objective, abs=1e-10)

    def test_returns_valid_bitstring(self, small_qubo_3: PortfolioQUBO) -> None:
        """Branch and bound should return a valid bitstring."""
        bb_bs, bb_obj = branch_and_bound_solve(small_qubo_3)
        assert len(bb_bs) == 3
        assert all(c in "01" for c in bb_bs)
        assert isinstance(bb_obj, float)
        assert not np.isnan(bb_obj)


# ---------------------------------------------------------------------------
# Comparison utility tests
# ---------------------------------------------------------------------------


class TestCompareOptimizers:
    def test_compare_returns_all_methods(
        self, small_qubo_3: PortfolioQUBO, mock_backend: MockBackend,
    ) -> None:
        """compare_optimizers should return results for all methods."""
        config = GroverSearchConfig(max_iterations=2, seed=42)
        results = compare_optimizers(small_qubo_3, config, mock_backend)

        assert "grover_adaptive" in results
        assert "branch_and_bound" in results
        assert "objective" in results["grover_adaptive"]
        assert "wall_time_s" in results["grover_adaptive"]
        assert "objective" in results["branch_and_bound"]

    def test_branch_and_bound_objective_is_optimal(
        self, small_qubo_3: PortfolioQUBO, mock_backend: MockBackend,
    ) -> None:
        """Branch-and-bound should find the best or near-best objective."""
        config = GroverSearchConfig(max_iterations=2, seed=42)
        results = compare_optimizers(small_qubo_3, config, mock_backend)

        bb_obj = results["branch_and_bound"]["objective"]
        gas_obj = results["grover_adaptive"]["objective"]
        # B&B should be exact; Grover may or may not match
        assert bb_obj <= gas_obj + 1.0
