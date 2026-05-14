"""Tests for the D-Wave quantum annealing backend.

All dwave-ocean-sdk imports are mocked since the SDK may not be installed.
"""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Mock the dwave SDK before any qufin import that touches it
# ---------------------------------------------------------------------------


def _make_dwave_mocks() -> dict[str, ModuleType]:
    """Create a minimal mock tree for dwave-ocean-sdk."""
    dimod = MagicMock(name="dimod")
    dimod.BinaryQuadraticModel = MagicMock(name="BinaryQuadraticModel")
    dimod.SimulatedAnnealingSampler = MagicMock(name="SimulatedAnnealingSampler")
    dimod.ConstrainedQuadraticModel = MagicMock(name="ConstrainedQuadraticModel")
    dimod.Binary = MagicMock(name="Binary")
    dimod.Integer = MagicMock(name="Integer")

    dwave = ModuleType("dwave")
    dwave_system = MagicMock(name="dwave.system")
    dwave_system.DWaveSampler = MagicMock(name="DWaveSampler")
    dwave_system.EmbeddingComposite = MagicMock(name="EmbeddingComposite")
    dwave_system.LeapHybridCQMSampler = MagicMock(name="LeapHybridCQMSampler")
    dwave_system.LeapHybridSampler = MagicMock(name="LeapHybridSampler")
    dwave.system = dwave_system

    minorminer = MagicMock(name="minorminer")
    dwave_networkx = MagicMock(name="dwave_networkx")

    return {
        "dimod": dimod,
        "dwave": dwave,
        "dwave.system": dwave_system,
        "minorminer": minorminer,
        "dwave_networkx": dwave_networkx,
    }


_dwave_mocks = _make_dwave_mocks()


@pytest.fixture(autouse=True)
def _patch_dwave():
    """Patch dwave modules for every test in this file."""
    with patch.dict(sys.modules, _dwave_mocks):
        yield


# ---------------------------------------------------------------------------
# Helper to get a fresh import of the module under test
# ---------------------------------------------------------------------------


def _import_dwave_backend():
    """Import (or reimport) the dwave_backend module."""
    # Remove cached module to force reimport with mocks active
    mod_name = "qufin.backends.dwave_backend"
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    from qufin.backends.dwave_backend import (
        AnnealingResult,
        AnnealingTiming,
        BenchmarkMetrics,
        DWaveBackend,
        DWaveConfig,
        DWaveTopology,
        EmbeddingInfo,
        PortfolioBenchmarkConfig,
        SolverType,
        _matrix_to_qubo_dict,
        estimate_dwave_cost,
        generate_portfolio_problem,
        simulated_annealing_solve,
    )
    return {
        "AnnealingResult": AnnealingResult,
        "AnnealingTiming": AnnealingTiming,
        "BenchmarkMetrics": BenchmarkMetrics,
        "DWaveBackend": DWaveBackend,
        "DWaveConfig": DWaveConfig,
        "DWaveTopology": DWaveTopology,
        "EmbeddingInfo": EmbeddingInfo,
        "PortfolioBenchmarkConfig": PortfolioBenchmarkConfig,
        "SolverType": SolverType,
        "_matrix_to_qubo_dict": _matrix_to_qubo_dict,
        "estimate_dwave_cost": estimate_dwave_cost,
        "generate_portfolio_problem": generate_portfolio_problem,
        "simulated_annealing_solve": simulated_annealing_solve,
    }


# ---------------------------------------------------------------------------
# Enums and config tests
# ---------------------------------------------------------------------------


class TestEnumsAndConfig:
    """Tests for DWaveTopology, SolverType, DWaveConfig."""

    def test_topology_values(self):
        m = _import_dwave_backend()
        assert m["DWaveTopology"].PEGASUS.value == "pegasus"
        assert m["DWaveTopology"].ZEPHYR.value == "zephyr"
        assert m["DWaveTopology"].CHIMERA.value == "chimera"

    def test_solver_type_values(self):
        m = _import_dwave_backend()
        assert m["SolverType"].QPU.value == "qpu"
        assert m["SolverType"].HYBRID_CQM.value == "hybrid_cqm"
        assert m["SolverType"].SIMULATED.value == "simulated"

    def test_config_defaults(self):
        m = _import_dwave_backend()
        config = m["DWaveConfig"]()
        assert config.solver_type == m["SolverType"].SIMULATED
        assert config.topology == m["DWaveTopology"].PEGASUS
        assert config.num_reads == 1000
        assert config.annealing_time == 20.0
        assert config.chain_strength is None
        assert config.auto_scale is True

    def test_config_custom(self):
        m = _import_dwave_backend()
        config = m["DWaveConfig"](
            solver_type=m["SolverType"].QPU,
            num_reads=500,
            token="test-token",
        )
        assert config.solver_type == m["SolverType"].QPU
        assert config.num_reads == 500
        assert config.token == "test-token"


# ---------------------------------------------------------------------------
# Result dataclass tests
# ---------------------------------------------------------------------------


class TestResultDataclasses:
    """Tests for AnnealingResult, AnnealingTiming, EmbeddingInfo."""

    def test_annealing_result_defaults(self):
        m = _import_dwave_backend()
        r = m["AnnealingResult"]()
        assert r.best_sample == {}
        assert r.best_energy == 0.0
        assert r.all_samples == []
        assert r.chain_break_fraction == 0.0

    def test_annealing_timing(self):
        m = _import_dwave_backend()
        t = m["AnnealingTiming"](
            total_seconds=1.5,
            embedding_seconds=0.3,
            sampling_seconds=1.0,
        )
        assert t.total_seconds == 1.5
        assert t.embedding_seconds == 0.3

    def test_embedding_info(self):
        m = _import_dwave_backend()
        ei = m["EmbeddingInfo"](
            logical_qubits=10,
            physical_qubits=25,
            max_chain_length=4,
            avg_chain_length=2.5,
        )
        assert ei.logical_qubits == 10
        assert ei.physical_qubits == 25

    def test_benchmark_metrics(self):
        m = _import_dwave_backend()
        bm = m["BenchmarkMetrics"](
            method="dwave_qpu",
            n_assets=15,
            best_energy=-5.3,
            feasibility_rate=0.85,
        )
        assert bm.method == "dwave_qpu"
        assert bm.feasibility_rate == 0.85


# ---------------------------------------------------------------------------
# DWaveBackend core tests
# ---------------------------------------------------------------------------


class TestDWaveBackend:
    """Tests for the DWaveBackend class."""

    def test_backend_id(self):
        m = _import_dwave_backend()
        backend = m["DWaveBackend"]()
        assert backend.backend_id == "dwave_simulated"

    def test_backend_id_qpu(self):
        m = _import_dwave_backend()
        config = m["DWaveConfig"](solver_type=m["SolverType"].QPU)
        backend = m["DWaveBackend"](config)
        assert backend.backend_id == "dwave_qpu"

    def test_is_simulator_true(self):
        m = _import_dwave_backend()
        backend = m["DWaveBackend"]()
        assert backend.is_simulator() is True

    def test_is_simulator_false(self):
        m = _import_dwave_backend()
        config = m["DWaveConfig"](solver_type=m["SolverType"].QPU)
        backend = m["DWaveBackend"](config)
        assert backend.is_simulator() is False

    def test_run_raises(self):
        m = _import_dwave_backend()
        backend = m["DWaveBackend"]()
        with pytest.raises(NotImplementedError, match="annealing backend"):
            backend.run(None)

    def test_statevector_raises(self):
        m = _import_dwave_backend()
        backend = m["DWaveBackend"]()
        with pytest.raises(NotImplementedError, match="annealing backend"):
            backend.statevector(None)

    def test_config_property(self):
        m = _import_dwave_backend()
        config = m["DWaveConfig"](num_reads=42)
        backend = m["DWaveBackend"](config)
        assert backend.config.num_reads == 42

    def test_last_embedding_info_initially_none(self):
        m = _import_dwave_backend()
        backend = m["DWaveBackend"]()
        assert backend.last_embedding_info is None


# ---------------------------------------------------------------------------
# QUBO matrix conversion tests
# ---------------------------------------------------------------------------


class TestMatrixToQuboDict:
    """Tests for _matrix_to_qubo_dict helper."""

    def test_diagonal_matrix(self):
        m = _import_dwave_backend()
        Q = np.diag([1.0, -2.0, 3.0])
        d = m["_matrix_to_qubo_dict"](Q)
        assert d[(0, 0)] == 1.0
        assert d[(1, 1)] == -2.0
        assert d[(2, 2)] == 3.0

    def test_off_diagonal(self):
        m = _import_dwave_backend()
        Q = np.array([[0.0, 0.5], [0.5, 0.0]])
        d = m["_matrix_to_qubo_dict"](Q)
        assert d[(0, 1)] == 1.0  # upper + lower combined

    def test_zero_entries_excluded(self):
        m = _import_dwave_backend()
        Q = np.zeros((3, 3))
        Q[0, 0] = 1.0
        d = m["_matrix_to_qubo_dict"](Q)
        assert len(d) == 1
        assert d[(0, 0)] == 1.0

    def test_empty_matrix(self):
        m = _import_dwave_backend()
        Q = np.zeros((2, 2))
        d = m["_matrix_to_qubo_dict"](Q)
        assert len(d) == 0


# ---------------------------------------------------------------------------
# Simulated annealing tests (pure numpy, no SDK)
# ---------------------------------------------------------------------------


class TestSimulatedAnnealing:
    """Tests for the pure-numpy simulated annealing solver."""

    def test_simple_qubo(self):
        m = _import_dwave_backend()
        # Simple QUBO: minimize x0 + x1 - 3*x0*x1
        # Best: (1,1) -> 1 + 1 - 3 = -1
        Q = np.array([[1.0, -1.5], [-1.5, 1.0]])
        result = m["simulated_annealing_solve"](Q, num_reads=20, num_sweeps=200, seed=42)
        assert result.best_energy <= 0.0
        assert len(result.all_samples) == 20

    def test_all_zero_qubo(self):
        m = _import_dwave_backend()
        Q = np.zeros((3, 3))
        result = m["simulated_annealing_solve"](Q, num_reads=5, num_sweeps=50, seed=0)
        # All solutions have energy 0
        assert result.best_energy == 0.0

    def test_result_shapes(self):
        m = _import_dwave_backend()
        Q = np.eye(4)
        result = m["simulated_annealing_solve"](Q, num_reads=10, num_sweeps=100, seed=1)
        assert len(result.all_samples) == 10
        assert len(result.all_energies) == 10
        assert len(result.num_occurrences) == 10

    def test_timing_positive(self):
        m = _import_dwave_backend()
        Q = np.eye(2)
        result = m["simulated_annealing_solve"](Q, num_reads=5, num_sweeps=50, seed=2)
        assert result.timing.total_seconds > 0

    def test_deterministic_with_seed(self):
        m = _import_dwave_backend()
        Q = np.array([[1.0, -0.5], [-0.5, 1.0]])
        r1 = m["simulated_annealing_solve"](Q, num_reads=5, num_sweeps=100, seed=99)
        r2 = m["simulated_annealing_solve"](Q, num_reads=5, num_sweeps=100, seed=99)
        assert r1.best_energy == r2.best_energy
        np.testing.assert_array_equal(r1.all_energies, r2.all_energies)


# ---------------------------------------------------------------------------
# Portfolio problem generation tests
# ---------------------------------------------------------------------------


class TestPortfolioGeneration:
    """Tests for generate_portfolio_problem."""

    def test_shapes(self):
        m = _import_dwave_backend()
        config = m["PortfolioBenchmarkConfig"](n_assets=15)
        mu, cov = m["generate_portfolio_problem"](config)
        assert mu.shape == (15,)
        assert cov.shape == (15, 15)

    def test_cov_symmetric(self):
        m = _import_dwave_backend()
        config = m["PortfolioBenchmarkConfig"](n_assets=10)
        _mu, cov = m["generate_portfolio_problem"](config)
        np.testing.assert_allclose(cov, cov.T, atol=1e-12)

    def test_cov_positive_semidefinite(self):
        m = _import_dwave_backend()
        config = m["PortfolioBenchmarkConfig"](n_assets=10)
        _mu, cov = m["generate_portfolio_problem"](config)
        eigenvalues = np.linalg.eigvalsh(cov)
        assert np.all(eigenvalues >= -1e-10)

    def test_returns_range(self):
        m = _import_dwave_backend()
        config = m["PortfolioBenchmarkConfig"](n_assets=20)
        mu, _ = m["generate_portfolio_problem"](config)
        assert np.all(mu >= 0.05)
        assert np.all(mu <= 0.15)

    def test_deterministic(self):
        m = _import_dwave_backend()
        config = m["PortfolioBenchmarkConfig"](n_assets=10, seed=123)
        mu1, cov1 = m["generate_portfolio_problem"](config)
        mu2, cov2 = m["generate_portfolio_problem"](config)
        np.testing.assert_array_equal(mu1, mu2)
        np.testing.assert_array_equal(cov1, cov2)


# ---------------------------------------------------------------------------
# Cost estimation tests
# ---------------------------------------------------------------------------


class TestCostEstimation:
    """Tests for estimate_dwave_cost."""

    def test_qpu_cost(self):
        m = _import_dwave_backend()
        cost = m["estimate_dwave_cost"](10, 1000, m["SolverType"].QPU)
        assert cost > 0

    def test_hybrid_cost(self):
        m = _import_dwave_backend()
        cost = m["estimate_dwave_cost"](10, 1000, m["SolverType"].HYBRID_CQM)
        assert cost > 0

    def test_simulated_free(self):
        m = _import_dwave_backend()
        cost = m["estimate_dwave_cost"](10, 1000, m["SolverType"].SIMULATED)
        assert cost == 0.0

    def test_larger_problem_costs_more_hybrid(self):
        m = _import_dwave_backend()
        cost_small = m["estimate_dwave_cost"](10, 1000, m["SolverType"].HYBRID_CQM)
        cost_large = m["estimate_dwave_cost"](100, 1000, m["SolverType"].HYBRID_CQM)
        assert cost_large >= cost_small


# ---------------------------------------------------------------------------
# PortfolioBenchmarkConfig tests
# ---------------------------------------------------------------------------


class TestPortfolioBenchmarkConfig:
    """Tests for the benchmark config dataclass."""

    def test_defaults(self):
        m = _import_dwave_backend()
        config = m["PortfolioBenchmarkConfig"]()
        assert config.n_assets == 15
        assert config.gamma == 1.0
        assert config.cardinality is None

    def test_custom(self):
        m = _import_dwave_backend()
        config = m["PortfolioBenchmarkConfig"](
            n_assets=50, gamma=2.0, cardinality=10
        )
        assert config.n_assets == 50
        assert config.cardinality == 10
