"""Unit tests for VQE portfolio optimizer and warm-start strategies.

Tests VQE on a small 4-asset problem (verifiable against exhaustive solver),
exercises all entanglement types, and validates that warm-start initialization
produces valid results without degrading solution quality.
"""

from __future__ import annotations

import numpy as np
import pytest

from qufin.backends.qiskit_backend import QiskitAerBackend
from qufin.portfolio.optimizers.exhaustive import exhaustive_solve
from qufin.portfolio.optimizers.qaoa import QAOAConfig, QAOAPortfolio
from qufin.portfolio.optimizers.vqe import VQEConfig, VQEPortfolio, VQEResult
from qufin.portfolio.optimizers.warm_start import (
    WarmStartResult,
    warm_start_qaoa,
    warm_start_vqe,
)
from qufin.portfolio.qubo import PortfolioQUBO


@pytest.fixture
def aer_backend() -> QiskitAerBackend:
    return QiskitAerBackend(seed=42)


@pytest.fixture
def small_qubo() -> PortfolioQUBO:
    """4-asset portfolio with cardinality=2 for fast testing."""
    mu = np.array([0.01, 0.02, 0.015, 0.008])
    cov = np.array([
        [0.04, 0.006, 0.002, 0.001],
        [0.006, 0.09, 0.004, 0.003],
        [0.002, 0.004, 0.01, 0.002],
        [0.001, 0.003, 0.002, 0.025],
    ])
    return PortfolioQUBO(mu=mu, cov=cov, cardinality=2, gamma=0.5)


@pytest.fixture
def unconstrained_qubo() -> PortfolioQUBO:
    """4-asset portfolio without cardinality for simpler tests."""
    mu = np.array([0.01, 0.02, 0.015, 0.008])
    cov = np.eye(4) * 0.04
    return PortfolioQUBO(mu=mu, cov=cov, gamma=1.0)


class TestVQEConfig:
    def test_default_config(self) -> None:
        config = VQEConfig()
        assert config.reps == 3
        assert config.entanglement == "linear"
        assert config.optimizer == "COBYLA"
        assert config.maxiter == 300
        assert config.shots == 8192
        assert config.seed == 42
        assert config.cvar_alpha == 0.5
        assert config.initial_params is None
        assert config.rotation_blocks == ["ry", "rz"]

    def test_custom_config(self) -> None:
        config = VQEConfig(
            reps=2,
            entanglement="full",
            optimizer="SPSA",
            maxiter=100,
            shots=4096,
            seed=123,
            cvar_alpha=0.3,
        )
        assert config.reps == 2
        assert config.entanglement == "full"
        assert config.optimizer == "SPSA"
        assert config.seed == 123


class TestVQEResult:
    def test_result_inherits_from_result(self) -> None:
        result = VQEResult()
        assert hasattr(result, "value")
        assert hasattr(result, "wall_time_s")
        assert hasattr(result, "to_dict")
        assert hasattr(result, "to_json")

    def test_result_defaults(self) -> None:
        result = VQEResult()
        assert result.best_bitstring == ""
        assert result.best_objective == float("inf")
        assert result.feasible is False
        assert len(result.history) == 0


class TestVQEPortfolio:
    def test_n_params(
        self, small_qubo: PortfolioQUBO, aer_backend: QiskitAerBackend
    ) -> None:
        """Check parameter count formula: n_rot * n * (reps + 1)."""
        config = VQEConfig(reps=2, seed=42)
        solver = VQEPortfolio(small_qubo, config, aer_backend)
        # 2 rotation blocks (ry, rz) * 4 qubits * (2 + 1) layers = 24
        assert solver._n_params() == 24

    def test_n_params_single_rep(
        self, small_qubo: PortfolioQUBO, aer_backend: QiskitAerBackend
    ) -> None:
        config = VQEConfig(reps=1, seed=42)
        solver = VQEPortfolio(small_qubo, config, aer_backend)
        # 2 * 4 * 2 = 16
        assert solver._n_params() == 16

    def test_vqe_produces_valid_result(
        self, small_qubo: PortfolioQUBO, aer_backend: QiskitAerBackend
    ) -> None:
        """VQE should return a well-formed VQEResult."""
        config = VQEConfig(reps=1, shots=1024, maxiter=15, seed=42)
        solver = VQEPortfolio(small_qubo, config, aer_backend)
        result = solver.run()

        assert isinstance(result, VQEResult)
        assert len(result.best_bitstring) == 4
        assert all(c in "01" for c in result.best_bitstring)
        assert result.weights.shape == (4,)
        assert result.wall_time_s > 0
        assert result.optimal_params.shape[0] == solver._n_params()
        assert len(result.history) > 0
        assert result.backend_id == aer_backend.backend_id
        assert result.seed == 42
        assert not np.isnan(result.best_objective)

    def test_vqe_against_exhaustive(
        self, small_qubo: PortfolioQUBO, aer_backend: QiskitAerBackend
    ) -> None:
        """VQE objective should be in the right ballpark vs exact solution."""
        exact = exhaustive_solve(small_qubo)

        config = VQEConfig(
            reps=2, shots=4096, maxiter=50, seed=42, cvar_alpha=0.3,
        )
        solver = VQEPortfolio(small_qubo, config, aer_backend)
        result = solver.run()

        # Generous bound: VQE objective within factor of 3 or within 1.0
        assert (
            result.best_objective <= exact.best_objective * 3.0
            or result.best_objective <= exact.best_objective + 1.0
        )

    def test_vqe_entanglement_linear(
        self, small_qubo: PortfolioQUBO, aer_backend: QiskitAerBackend
    ) -> None:
        config = VQEConfig(
            reps=1, entanglement="linear", shots=512, maxiter=10, seed=42,
        )
        solver = VQEPortfolio(small_qubo, config, aer_backend)
        result = solver.run()
        assert len(result.best_bitstring) == 4
        assert not np.isnan(result.best_objective)

    def test_vqe_entanglement_circular(
        self, small_qubo: PortfolioQUBO, aer_backend: QiskitAerBackend
    ) -> None:
        config = VQEConfig(
            reps=1, entanglement="circular", shots=512, maxiter=10, seed=42,
        )
        solver = VQEPortfolio(small_qubo, config, aer_backend)
        result = solver.run()
        assert len(result.best_bitstring) == 4
        assert not np.isnan(result.best_objective)

    def test_vqe_entanglement_full(
        self, small_qubo: PortfolioQUBO, aer_backend: QiskitAerBackend
    ) -> None:
        config = VQEConfig(
            reps=1, entanglement="full", shots=512, maxiter=10, seed=42,
        )
        solver = VQEPortfolio(small_qubo, config, aer_backend)
        result = solver.run()
        assert len(result.best_bitstring) == 4
        assert not np.isnan(result.best_objective)

    def test_vqe_cvar_alpha(
        self, small_qubo: PortfolioQUBO, aer_backend: QiskitAerBackend
    ) -> None:
        """CVaR with alpha < 1 should produce a valid result."""
        config = VQEConfig(
            reps=1, shots=1024, maxiter=10, seed=42, cvar_alpha=0.2,
        )
        solver = VQEPortfolio(small_qubo, config, aer_backend)
        result = solver.run()
        assert isinstance(result.best_objective, float)
        assert not np.isnan(result.best_objective)

    def test_vqe_custom_initial_params(
        self, small_qubo: PortfolioQUBO, aer_backend: QiskitAerBackend
    ) -> None:
        """VQE should accept user-provided initial parameters."""
        config = VQEConfig(reps=1, shots=512, maxiter=5, seed=42)
        solver = VQEPortfolio(small_qubo, config, aer_backend)
        n_params = solver._n_params()

        custom_params = np.zeros(n_params)
        config_custom = VQEConfig(
            reps=1, shots=512, maxiter=5, seed=42,
            initial_params=custom_params,
        )
        solver_custom = VQEPortfolio(small_qubo, config_custom, aer_backend)
        result = solver_custom.run()
        assert isinstance(result, VQEResult)
        assert not np.isnan(result.best_objective)

    def test_vqe_weights_sum(
        self, small_qubo: PortfolioQUBO, aer_backend: QiskitAerBackend
    ) -> None:
        """Decoded weights should sum to 1 (if any asset selected)."""
        config = VQEConfig(reps=1, shots=1024, maxiter=15, seed=42)
        solver = VQEPortfolio(small_qubo, config, aer_backend)
        result = solver.run()

        if np.any(result.weights > 0):
            assert abs(np.sum(result.weights) - 1.0) < 1e-10

    def test_vqe_serialization(
        self, small_qubo: PortfolioQUBO, aer_backend: QiskitAerBackend
    ) -> None:
        """VQEResult should be JSON-serializable."""
        config = VQEConfig(reps=1, shots=512, maxiter=5, seed=42)
        solver = VQEPortfolio(small_qubo, config, aer_backend)
        result = solver.run()

        json_str = result.to_json()
        assert isinstance(json_str, str)
        assert "best_bitstring" in json_str

        d = result.to_dict()
        assert isinstance(d, dict)
        assert "best_objective" in d


class TestWarmStartVQE:
    def test_warm_start_produces_valid_result(
        self, small_qubo: PortfolioQUBO, aer_backend: QiskitAerBackend
    ) -> None:
        """Warm-started VQE should produce a valid VQEResult."""
        config = VQEConfig(reps=1, shots=1024, maxiter=15, seed=42)
        solver = VQEPortfolio(small_qubo, config, aer_backend)
        n_params = solver._n_params()

        ws = warm_start_vqe(small_qubo, n_params=n_params, seed=42)
        assert isinstance(ws, WarmStartResult)
        assert ws.initial_params.shape == (n_params,)

        config_ws = VQEConfig(
            reps=1, shots=1024, maxiter=15, seed=42,
            initial_params=ws.initial_params,
        )
        solver_ws = VQEPortfolio(small_qubo, config_ws, aer_backend)
        result = solver_ws.run()

        assert isinstance(result, VQEResult)
        assert not np.isnan(result.best_objective)
        assert len(result.best_bitstring) == 4

    def test_warm_start_does_not_hurt(
        self, small_qubo: PortfolioQUBO, aer_backend: QiskitAerBackend
    ) -> None:
        """Warm-started VQE should be no worse than cold-start (with tolerance)."""
        maxiter = 25
        shots = 2048

        # Cold start
        config_cold = VQEConfig(
            reps=1, shots=shots, maxiter=maxiter, seed=42, cvar_alpha=0.3,
        )
        solver_cold = VQEPortfolio(small_qubo, config_cold, aer_backend)
        result_cold = solver_cold.run()

        # Warm start
        n_params = solver_cold._n_params()
        ws = warm_start_vqe(small_qubo, n_params=n_params, seed=42)
        config_warm = VQEConfig(
            reps=1, shots=shots, maxiter=maxiter, seed=42, cvar_alpha=0.3,
            initial_params=ws.initial_params,
        )
        solver_warm = VQEPortfolio(small_qubo, config_warm, aer_backend)
        result_warm = solver_warm.run()

        # Warm-start should not be significantly worse
        # Allow generous tolerance since quantum optimization is stochastic
        assert result_warm.best_objective <= result_cold.best_objective + 1.0


class TestWarmStartQAOA:
    def test_warm_start_qaoa_produces_valid_result(
        self, small_qubo: PortfolioQUBO
    ) -> None:
        ws = warm_start_qaoa(small_qubo, p=2, seed=42)
        assert isinstance(ws, WarmStartResult)
        assert ws.initial_params.shape == (4,)  # 2 gammas + 2 betas
        assert len(ws.rounded_bitstring) == 4

    def test_warm_start_qaoa_runs(
        self, small_qubo: PortfolioQUBO, aer_backend: QiskitAerBackend
    ) -> None:
        """Warm-started QAOA should run end-to-end."""
        ws = warm_start_qaoa(small_qubo, p=2, seed=42)
        gammas = ws.initial_params[:2]
        betas = ws.initial_params[2:]

        config = QAOAConfig(
            p=2, mixer="x", shots=1024, maxiter=15, seed=42,
            initial_gammas=gammas,
            initial_betas=betas,
        )
        solver = QAOAPortfolio(small_qubo, config, aer_backend)
        result = solver.run()

        assert len(result.best_bitstring) == 4
        assert not np.isnan(result.best_objective)

    def test_warm_start_qaoa_does_not_hurt(
        self, small_qubo: PortfolioQUBO, aer_backend: QiskitAerBackend
    ) -> None:
        """Warm-started QAOA should be no worse than cold-start (with tolerance)."""
        p = 2
        maxiter = 20
        shots = 2048

        # Cold start
        config_cold = QAOAConfig(
            p=p, mixer="x", shots=shots, maxiter=maxiter, seed=42,
        )
        solver_cold = QAOAPortfolio(small_qubo, config_cold, aer_backend)
        result_cold = solver_cold.run()

        # Warm start
        ws = warm_start_qaoa(small_qubo, p=p, seed=42)
        config_warm = QAOAConfig(
            p=p, mixer="x", shots=shots, maxiter=maxiter, seed=42,
            initial_gammas=ws.initial_params[:p],
            initial_betas=ws.initial_params[p:],
        )
        solver_warm = QAOAPortfolio(small_qubo, config_warm, aer_backend)
        result_warm = solver_warm.run()

        # Warm-start should not be significantly worse
        assert result_warm.best_objective <= result_cold.best_objective + 1.0
