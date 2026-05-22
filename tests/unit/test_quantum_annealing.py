"""Unit tests for qufin.portfolio.optimizers.quantum_annealing module."""

from __future__ import annotations

import numpy as np
import pytest

from qufin.backends.mock import MockBackend
from qufin.portfolio.optimizers.quantum_annealing import (
    SQAConfig,
    SQAResult,
    _qubo_energy,
    random_restart_hill_climbing,
    simulated_quantum_annealing,
)
from qufin.portfolio.qubo import PortfolioQUBO

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_qubo() -> PortfolioQUBO:
    """3-asset portfolio QUBO."""
    mu = np.array([0.02, 0.03, 0.015])
    cov = np.array([
        [0.04, 0.006, 0.002],
        [0.006, 0.09, 0.004],
        [0.002, 0.004, 0.01],
    ])
    return PortfolioQUBO(mu=mu, cov=cov, gamma=0.5)


@pytest.fixture
def cardinality_qubo() -> PortfolioQUBO:
    """4-asset QUBO with cardinality constraint."""
    mu = np.array([0.01, 0.02, 0.015, 0.008])
    cov = np.array([
        [0.04, 0.006, 0.002, 0.001],
        [0.006, 0.09, 0.004, 0.003],
        [0.002, 0.004, 0.01, 0.002],
        [0.001, 0.003, 0.002, 0.025],
    ])
    return PortfolioQUBO(mu=mu, cov=cov, cardinality=2, gamma=0.5)


@pytest.fixture
def mock_backend() -> MockBackend:
    return MockBackend(seed=42)


@pytest.fixture
def fast_config() -> SQAConfig:
    return SQAConfig(n_sweeps=10, n_replicas=2, seed=42)


# ---------------------------------------------------------------------------
# SQAConfig tests
# ---------------------------------------------------------------------------


class TestSQAConfig:
    def test_defaults(self) -> None:
        cfg = SQAConfig()
        assert cfg.n_sweeps == 200
        assert cfg.n_replicas == 8
        assert cfg.beta == 2.0
        assert cfg.gamma_start == 3.0
        assert cfg.gamma_end == 0.01

    def test_custom(self) -> None:
        cfg = SQAConfig(n_sweeps=50, n_replicas=4, beta=5.0)
        assert cfg.n_sweeps == 50
        assert cfg.n_replicas == 4
        assert cfg.beta == 5.0


# ---------------------------------------------------------------------------
# QUBO energy tests
# ---------------------------------------------------------------------------


class TestQUBOEnergy:
    def test_zero_vector(self, small_qubo: PortfolioQUBO) -> None:
        Q = small_qubo.build_matrix()
        x = np.zeros(small_qubo.n_qubits)
        assert _qubo_energy(x, Q) == 0.0

    def test_energy_is_scalar(self, small_qubo: PortfolioQUBO) -> None:
        Q = small_qubo.build_matrix()
        x = np.ones(small_qubo.n_qubits)
        result = _qubo_energy(x, Q)
        assert isinstance(result, float)

    def test_energy_matches_manual(self) -> None:
        """2x2 diagonal Q: energy should be sum of diagonal entries for all-ones."""
        Q = np.diag([1.0, 2.0])
        x = np.array([1.0, 1.0])
        assert _qubo_energy(x, Q) == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# SQA solver tests
# ---------------------------------------------------------------------------


class TestSimulatedQuantumAnnealing:
    def test_returns_sqa_result(
        self, small_qubo: PortfolioQUBO, fast_config: SQAConfig
    ) -> None:
        result = simulated_quantum_annealing(small_qubo, fast_config)
        assert isinstance(result, SQAResult)

    def test_bitstring_length(
        self, small_qubo: PortfolioQUBO, fast_config: SQAConfig
    ) -> None:
        result = simulated_quantum_annealing(small_qubo, fast_config)
        assert len(result.best_bitstring) == small_qubo.n_qubits

    def test_energy_history_populated(
        self, small_qubo: PortfolioQUBO, fast_config: SQAConfig
    ) -> None:
        result = simulated_quantum_annealing(small_qubo, fast_config)
        assert len(result.energy_history) == fast_config.n_sweeps

    def test_wall_time_positive(
        self, small_qubo: PortfolioQUBO, fast_config: SQAConfig
    ) -> None:
        result = simulated_quantum_annealing(small_qubo, fast_config)
        assert result.wall_time_s > 0

    def test_objective_finite(
        self, small_qubo: PortfolioQUBO, fast_config: SQAConfig
    ) -> None:
        result = simulated_quantum_annealing(small_qubo, fast_config)
        assert np.isfinite(result.best_objective)

    def test_with_backend(
        self,
        small_qubo: PortfolioQUBO,
        fast_config: SQAConfig,
        mock_backend: MockBackend,
    ) -> None:
        result = simulated_quantum_annealing(small_qubo, fast_config, backend=mock_backend)
        assert "mock" in result.backend_id

    def test_without_backend(
        self, small_qubo: PortfolioQUBO, fast_config: SQAConfig
    ) -> None:
        result = simulated_quantum_annealing(small_qubo, fast_config)
        assert result.backend_id == "classical_sqa"

    def test_reproducibility(
        self, small_qubo: PortfolioQUBO
    ) -> None:
        cfg = SQAConfig(n_sweeps=5, n_replicas=2, seed=99)
        r1 = simulated_quantum_annealing(small_qubo, cfg)
        r2 = simulated_quantum_annealing(small_qubo, cfg)
        assert r1.best_objective == r2.best_objective
        assert r1.best_bitstring == r2.best_bitstring

    def test_cardinality_qubo(
        self, cardinality_qubo: PortfolioQUBO, fast_config: SQAConfig
    ) -> None:
        result = simulated_quantum_annealing(cardinality_qubo, fast_config)
        assert len(result.best_bitstring) == cardinality_qubo.n_qubits

    def test_weights_shape(
        self, small_qubo: PortfolioQUBO, fast_config: SQAConfig
    ) -> None:
        result = simulated_quantum_annealing(small_qubo, fast_config)
        assert result.weights is not None


# ---------------------------------------------------------------------------
# Hill climbing baseline tests
# ---------------------------------------------------------------------------


class TestRandomRestartHillClimbing:
    def test_returns_sqa_result(self, small_qubo: PortfolioQUBO) -> None:
        result = random_restart_hill_climbing(small_qubo, n_restarts=5, max_steps=10)
        assert isinstance(result, SQAResult)

    def test_bitstring_length(self, small_qubo: PortfolioQUBO) -> None:
        result = random_restart_hill_climbing(small_qubo, n_restarts=5, max_steps=10)
        assert len(result.best_bitstring) == small_qubo.n_qubits

    def test_objective_finite(self, small_qubo: PortfolioQUBO) -> None:
        result = random_restart_hill_climbing(small_qubo, n_restarts=5, max_steps=10)
        assert np.isfinite(result.best_objective)

    def test_backend_id(self, small_qubo: PortfolioQUBO) -> None:
        result = random_restart_hill_climbing(small_qubo, n_restarts=3, max_steps=5)
        assert result.backend_id == "classical_hill_climb"

    def test_reproducibility(self, small_qubo: PortfolioQUBO) -> None:
        r1 = random_restart_hill_climbing(small_qubo, n_restarts=5, seed=42)
        r2 = random_restart_hill_climbing(small_qubo, n_restarts=5, seed=42)
        assert r1.best_objective == r2.best_objective

    def test_energy_history_length(self, small_qubo: PortfolioQUBO) -> None:
        result = random_restart_hill_climbing(small_qubo, n_restarts=7, max_steps=5)
        assert len(result.energy_history) == 7
