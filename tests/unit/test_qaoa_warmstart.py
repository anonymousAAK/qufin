"""Unit tests for QAOA warm-start from continuous relaxation."""

from __future__ import annotations

import numpy as np
import pytest

from qufin.backends.mock import MockBackend
from qufin.portfolio.optimizers.qaoa import QAOAConfig
from qufin.portfolio.optimizers.qaoa_warmstart import (
    WarmStartQAOA,
    WarmStartQAOAResult,
)
from qufin.portfolio.qubo import PortfolioQUBO


def _mock_4q() -> MockBackend:
    """MockBackend with 4-qubit bitstrings."""
    counts = {"0110": 256, "1010": 256, "0011": 256, "1100": 256}
    return MockBackend(default_counts=counts, seed=42)


def _mock_3q() -> MockBackend:
    """MockBackend with 3-qubit bitstrings."""
    counts = {"010": 341, "101": 341, "110": 342}
    return MockBackend(default_counts=counts, seed=42)


@pytest.fixture
def small_qubo() -> PortfolioQUBO:
    mu = np.array([0.01, 0.02, 0.015, 0.008])
    cov = np.array([
        [0.04, 0.006, 0.002, 0.001],
        [0.006, 0.09, 0.004, 0.003],
        [0.002, 0.004, 0.01, 0.002],
        [0.001, 0.003, 0.002, 0.025],
    ])
    return PortfolioQUBO(mu=mu, cov=cov, cardinality=2, gamma=0.5)


@pytest.fixture
def backend() -> MockBackend:
    return _mock_4q()


@pytest.fixture
def config() -> QAOAConfig:
    return QAOAConfig(p=1, maxiter=5, shots=128, seed=42)


class TestWarmStartQAOAInit:
    def test_creates_instance(self, small_qubo, config, backend) -> None:
        ws = WarmStartQAOA(small_qubo, config, backend)
        assert ws.qubo is small_qubo
        assert ws.config is config
        assert ws.backend is backend

    def test_warm_params_shape(self, small_qubo, config, backend) -> None:
        ws = WarmStartQAOA(small_qubo, config, backend)
        x_rel, gammas, betas, obj, bs = ws._warm_params()
        assert x_rel.shape == (4,)
        assert gammas.shape == (config.p,)
        assert betas.shape == (config.p,)
        assert isinstance(obj, float)
        assert len(bs) == 4

    def test_warm_params_relaxed_bounds(self, small_qubo, config, backend) -> None:
        ws = WarmStartQAOA(small_qubo, config, backend)
        x_rel, _, _, _, _ = ws._warm_params()
        assert np.all(x_rel >= -1e-10)
        assert np.all(x_rel <= 1.0 + 1e-10)


class TestWarmStartQAOARun:
    def test_returns_result_type(self, small_qubo, config, backend) -> None:
        ws = WarmStartQAOA(small_qubo, config, backend)
        result = ws.run()
        assert isinstance(result, WarmStartQAOAResult)

    def test_result_has_relaxed_solution(self, small_qubo, config, backend) -> None:
        ws = WarmStartQAOA(small_qubo, config, backend)
        result = ws.run()
        assert result.relaxed_solution.shape == (4,)
        assert isinstance(result.relaxed_objective, float)

    def test_result_has_bitstring(self, small_qubo, config, backend) -> None:
        ws = WarmStartQAOA(small_qubo, config, backend)
        result = ws.run()
        assert len(result.best_bitstring) == 4
        assert all(c in "01" for c in result.best_bitstring)

    def test_result_has_weights(self, small_qubo, config, backend) -> None:
        ws = WarmStartQAOA(small_qubo, config, backend)
        result = ws.run()
        assert result.weights.shape == (4,)

    def test_wall_time_positive(self, small_qubo, config, backend) -> None:
        ws = WarmStartQAOA(small_qubo, config, backend)
        result = ws.run()
        assert result.wall_time_s > 0

    def test_history_populated(self, small_qubo, config, backend) -> None:
        ws = WarmStartQAOA(small_qubo, config, backend)
        result = ws.run()
        assert len(result.history) > 0

    def test_rounded_bitstring_ws(self, small_qubo, config, backend) -> None:
        ws = WarmStartQAOA(small_qubo, config, backend)
        result = ws.run()
        assert len(result.rounded_bitstring_ws) == 4

    def test_deterministic(self, small_qubo) -> None:
        cfg = QAOAConfig(p=1, maxiter=3, shots=64, seed=42)
        r1 = WarmStartQAOA(small_qubo, cfg, _mock_4q()).run()
        r2 = WarmStartQAOA(small_qubo, cfg, _mock_4q()).run()
        assert r1.best_bitstring == r2.best_bitstring


class TestWarmStartQAOAVariants:
    def test_no_cardinality(self) -> None:
        mu = np.array([0.01, 0.02, 0.015])
        cov = np.eye(3) * 0.04
        qubo = PortfolioQUBO(mu=mu, cov=cov, gamma=1.0)
        cfg = QAOAConfig(p=1, maxiter=3, shots=64, seed=42)
        ws = WarmStartQAOA(qubo, cfg, _mock_3q())
        result = ws.run()
        assert isinstance(result, WarmStartQAOAResult)
        assert result.relaxed_solution.shape == (3,)

    def test_different_p(self, small_qubo) -> None:
        for p in (1, 2):
            cfg = QAOAConfig(p=p, maxiter=3, shots=64, seed=42)
            ws = WarmStartQAOA(small_qubo, cfg, _mock_4q())
            result = ws.run()
            assert result.gammas.shape == (p,)
            assert result.betas.shape == (p,)

    def test_cvar_alpha(self, small_qubo) -> None:
        cfg = QAOAConfig(p=1, maxiter=3, shots=64, seed=42, cvar_alpha=0.5)
        ws = WarmStartQAOA(small_qubo, cfg, _mock_4q())
        result = ws.run()
        assert isinstance(result, WarmStartQAOAResult)

    def test_relaxed_objective_finite(self, small_qubo, config, backend) -> None:
        ws = WarmStartQAOA(small_qubo, config, backend)
        result = ws.run()
        assert np.isfinite(result.relaxed_objective)

    def test_backend_id_set(self, small_qubo, config, backend) -> None:
        ws = WarmStartQAOA(small_qubo, config, backend)
        result = ws.run()
        assert result.backend_id is not None
        assert len(result.backend_id) > 0
