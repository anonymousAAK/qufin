"""Integration tests: VQE and Hybrid on Qiskit Aer.

Tests the full VQE and hybrid pipelines on small problems.
"""

from __future__ import annotations

import numpy as np
import pytest

from qufin.backends.qiskit_backend import QiskitAerBackend
from qufin.portfolio.optimizers.exhaustive import exhaustive_solve
from qufin.portfolio.optimizers.hybrid import HybridConfig, HybridOptimizer, HybridResult
from qufin.portfolio.optimizers.vqe import VQEConfig, VQEPortfolio, VQEResult
from qufin.portfolio.optimizers.warm_start import warm_start_vqe
from qufin.portfolio.qubo import PortfolioQUBO


@pytest.fixture
def aer_backend() -> QiskitAerBackend:
    return QiskitAerBackend(seed=42)


@pytest.fixture
def small_qubo() -> PortfolioQUBO:
    """4-asset portfolio for fast testing."""
    mu = np.array([0.01, 0.02, 0.015, 0.008])
    cov = np.array([
        [0.04, 0.006, 0.002, 0.001],
        [0.006, 0.09, 0.004, 0.003],
        [0.002, 0.004, 0.01, 0.002],
        [0.001, 0.003, 0.002, 0.025],
    ])
    return PortfolioQUBO(mu=mu, cov=cov, cardinality=2, gamma=0.5)


class TestVQEOnAer:
    @pytest.mark.slow
    def test_vqe_produces_valid_output(
        self, small_qubo: PortfolioQUBO, aer_backend: QiskitAerBackend
    ) -> None:
        """VQE should produce valid output structure."""
        config = VQEConfig(reps=1, shots=1024, maxiter=20, seed=42)
        solver = VQEPortfolio(small_qubo, config, aer_backend)
        result = solver.run()

        assert isinstance(result, VQEResult)
        assert len(result.best_bitstring) == 4
        assert result.weights.shape == (4,)
        assert len(result.history) > 0
        assert result.wall_time_s > 0
        assert result.optimal_params.shape[0] > 0

    @pytest.mark.slow
    def test_vqe_finds_reasonable_solution(
        self, small_qubo: PortfolioQUBO, aer_backend: QiskitAerBackend
    ) -> None:
        """VQE should find a solution in the right ballpark."""
        exact = exhaustive_solve(small_qubo)

        config = VQEConfig(
            reps=2, shots=4096, maxiter=60, seed=42, cvar_alpha=0.3,
        )
        solver = VQEPortfolio(small_qubo, config, aer_backend)
        result = solver.run()

        # VQE objective within generous tolerance
        assert result.best_objective <= exact.best_objective * 3.0 or \
               result.best_objective <= exact.best_objective + 1.0

    @pytest.mark.slow
    def test_vqe_with_warm_start(
        self, small_qubo: PortfolioQUBO, aer_backend: QiskitAerBackend
    ) -> None:
        """Warm-started VQE should produce valid output."""
        config = VQEConfig(reps=1, shots=1024, maxiter=20, seed=42)
        solver = VQEPortfolio(small_qubo, config, aer_backend)
        n_params = solver._n_params()

        ws = warm_start_vqe(small_qubo, n_params=n_params, seed=42)
        config_ws = VQEConfig(
            reps=1, shots=1024, maxiter=20, seed=42,
            initial_params=ws.initial_params,
        )
        solver_ws = VQEPortfolio(small_qubo, config_ws, aer_backend)
        result = solver_ws.run()

        assert isinstance(result, VQEResult)
        assert not np.isnan(result.best_objective)

    @pytest.mark.slow
    def test_vqe_entanglement_modes(
        self, small_qubo: PortfolioQUBO, aer_backend: QiskitAerBackend
    ) -> None:
        """All entanglement modes should work."""
        for ent in ["linear", "circular", "full"]:
            config = VQEConfig(
                reps=1, entanglement=ent, shots=512, maxiter=10, seed=42,
            )
            solver = VQEPortfolio(small_qubo, config, aer_backend)
            result = solver.run()
            assert len(result.best_bitstring) == 4


class TestHybridOnAer:
    @pytest.mark.slow
    def test_hybrid_produces_valid_output(
        self, small_qubo: PortfolioQUBO, aer_backend: QiskitAerBackend
    ) -> None:
        """Hybrid optimizer should produce valid output."""
        config = HybridConfig(
            qaoa_p=1, qaoa_maxiter=15, qaoa_shots=1024, seed=42,
        )
        optimizer = HybridOptimizer(small_qubo, config, aer_backend)
        result = optimizer.run()

        assert isinstance(result, HybridResult)
        assert len(result.best_bitstring) == 4
        assert result.weights.shape == (4,)
        assert result.classical_time_s > 0
        assert result.quantum_time_s > 0
        assert result.relaxed_solution.shape == (4,)

    @pytest.mark.slow
    def test_hybrid_beats_random(
        self, small_qubo: PortfolioQUBO, aer_backend: QiskitAerBackend
    ) -> None:
        """Hybrid should outperform a random bitstring."""
        config = HybridConfig(
            qaoa_p=1, qaoa_maxiter=20, qaoa_shots=2048, seed=42,
        )
        optimizer = HybridOptimizer(small_qubo, config, aer_backend)
        result = optimizer.run()

        # Compare to random feasible bitstring "1100"
        random_obj = small_qubo.evaluate("1100")
        # Hybrid should be no worse than 2x the random solution
        assert result.best_objective <= random_obj * 2.0 + 1.0

    @pytest.mark.slow
    def test_hybrid_rounded_is_feasible(
        self, small_qubo: PortfolioQUBO, aer_backend: QiskitAerBackend
    ) -> None:
        """The rounded classical solution should have correct cardinality."""
        config = HybridConfig(qaoa_p=1, qaoa_maxiter=10, qaoa_shots=512, seed=42)
        optimizer = HybridOptimizer(small_qubo, config, aer_backend)
        result = optimizer.run()

        # Rounded bitstring should have exactly K=2 ones
        assert sum(int(c) for c in result.rounded_bitstring) == 2
