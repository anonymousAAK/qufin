"""Tests for portfolio QUBO formulation with full constraints."""

from __future__ import annotations

import numpy as np
import pytest

from qufin.portfolio.qubo import PortfolioQUBO


class TestPortfolioQUBO:
    def test_matrix_shape(self, sample_mu: np.ndarray, sample_cov: np.ndarray) -> None:
        qubo = PortfolioQUBO(mu=sample_mu, cov=sample_cov)
        Q = qubo.build_matrix()
        n = len(sample_mu)
        assert Q.shape == (n, n)

    def test_evaluate_all_zeros(self, sample_mu: np.ndarray, sample_cov: np.ndarray) -> None:
        qubo = PortfolioQUBO(mu=sample_mu, cov=sample_cov)
        val = qubo.evaluate("00000")
        assert val == 0.0

    def test_evaluate_deterministic(self, sample_mu: np.ndarray, sample_cov: np.ndarray) -> None:
        qubo = PortfolioQUBO(mu=sample_mu, cov=sample_cov)
        v1 = qubo.evaluate("10101")
        v2 = qubo.evaluate("10101")
        assert v1 == v2

    def test_n_qubits_one_hot(self, sample_mu: np.ndarray, sample_cov: np.ndarray) -> None:
        qubo = PortfolioQUBO(mu=sample_mu, cov=sample_cov)
        assert qubo.n_qubits == len(sample_mu)

    def test_n_qubits_binary(self, sample_mu: np.ndarray, sample_cov: np.ndarray) -> None:
        qubo = PortfolioQUBO(mu=sample_mu, cov=sample_cov, encoding="binary", bits_per_asset=3)
        assert qubo.n_qubits == len(sample_mu) * 3

    def test_cov_shape_mismatch_raises(self) -> None:
        mu = np.array([0.01, 0.02])
        cov = np.eye(3)
        with pytest.raises(ValueError, match="Covariance shape"):
            PortfolioQUBO(mu=mu, cov=cov)


class TestCardinalityConstraint:
    def test_cardinality_penalty_prefers_k(
        self, sample_mu: np.ndarray, sample_cov: np.ndarray
    ) -> None:
        K = 2
        qubo = PortfolioQUBO(mu=sample_mu, cov=sample_cov, cardinality=K)
        val_k = qubo.evaluate("10100")  # 2 selected = K
        val_4 = qubo.evaluate("11110")  # 4 selected != K
        assert val_k < val_4  # K-asset solution should be preferred

    def test_cardinality_feasibility_check(
        self, sample_mu: np.ndarray, sample_cov: np.ndarray
    ) -> None:
        qubo = PortfolioQUBO(mu=sample_mu, cov=sample_cov, cardinality=2)
        assert qubo.feasibility_check("10100")["cardinality"] is True
        assert qubo.feasibility_check("11100")["cardinality"] is False


class TestSectorConstraints:
    def test_sector_cap_feasibility(self) -> None:
        mu = np.array([0.01, 0.02, 0.03, 0.04])
        cov = np.eye(4) * 0.01
        sector_map = {0: 0, 1: 0, 2: 1, 3: 1}  # 2 sectors
        sector_caps = {0: 1, 1: 1}  # max 1 per sector
        qubo = PortfolioQUBO(
            mu=mu, cov=cov, sector_map=sector_map, sector_caps=sector_caps
        )
        # "1010" -> 1 from sector 0, 1 from sector 1 (feasible)
        assert qubo.feasibility_check("1010")["sector"] is True
        # "1100" -> 2 from sector 0 (infeasible)
        assert qubo.feasibility_check("1100")["sector"] is False

    def test_sector_penalty_applied(self) -> None:
        mu = np.array([0.01, 0.02, 0.03, 0.04])
        cov = np.eye(4) * 0.01
        sector_map = {0: 0, 1: 0, 2: 1, 3: 1}
        sector_caps = {0: 1, 1: 1}
        qubo = PortfolioQUBO(
            mu=mu, cov=cov, sector_map=sector_map, sector_caps=sector_caps
        )
        Q = qubo.build_matrix()
        assert Q.shape == (4, 4)
        # The penalty should make the matrix different from unconstrained
        qubo_nc = PortfolioQUBO(mu=mu, cov=cov)
        Q_nc = qubo_nc.build_matrix()
        assert not np.allclose(Q, Q_nc)


class TestTurnoverAndTransactionCost:
    def test_turnover_penalty(self, sample_mu: np.ndarray, sample_cov: np.ndarray) -> None:
        prev = np.array([1.0, 0.0, 1.0, 0.0, 0.0])
        qubo = PortfolioQUBO(
            mu=sample_mu, cov=sample_cov,
            turnover_penalty=1.0, previous_weights=prev,
        )
        # Staying in same portfolio should be cheaper than flipping
        val_same = qubo.evaluate("10100")
        val_flip = qubo.evaluate("01011")
        # With turnover penalty, same should be lower
        assert val_same < val_flip

    def test_transaction_cost_penalty(self, sample_mu: np.ndarray, sample_cov: np.ndarray) -> None:
        prev = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
        qubo = PortfolioQUBO(
            mu=sample_mu, cov=sample_cov,
            transaction_cost=2.0, previous_weights=prev,
        )
        Q = qubo.build_matrix()
        qubo_nc = PortfolioQUBO(mu=sample_mu, cov=sample_cov)
        Q_nc = qubo_nc.build_matrix()
        assert not np.allclose(Q, Q_nc)


class TestBinaryEncoding:
    def test_binary_matrix_shape(self) -> None:
        mu = np.array([0.01, 0.02, 0.03])
        cov = np.eye(3) * 0.01
        qubo = PortfolioQUBO(mu=mu, cov=cov, encoding="binary", bits_per_asset=3)
        Q = qubo.build_matrix()
        assert Q.shape == (9, 9)

    def test_binary_evaluate(self) -> None:
        mu = np.array([0.01, 0.02])
        cov = np.eye(2) * 0.01
        qubo = PortfolioQUBO(mu=mu, cov=cov, encoding="binary", bits_per_asset=2)
        val = qubo.evaluate("0000")
        assert val == 0.0  # all zeros is trivially 0 (x=0)


class TestDecodeWeights:
    def test_one_hot_decode(self, sample_mu: np.ndarray, sample_cov: np.ndarray) -> None:
        qubo = PortfolioQUBO(mu=sample_mu, cov=sample_cov)
        w = qubo.decode_weights("10100")
        assert abs(w.sum() - 1.0) < 1e-10
        assert w[0] == pytest.approx(0.5)
        assert w[2] == pytest.approx(0.5)
        assert w[1] == 0.0

    def test_one_hot_all_zeros(self, sample_mu: np.ndarray, sample_cov: np.ndarray) -> None:
        qubo = PortfolioQUBO(mu=sample_mu, cov=sample_cov)
        w = qubo.decode_weights("00000")
        np.testing.assert_array_equal(w, 0.0)

    def test_binary_decode(self) -> None:
        mu = np.array([0.01, 0.02])
        cov = np.eye(2) * 0.01
        qubo = PortfolioQUBO(mu=mu, cov=cov, encoding="binary", bits_per_asset=3)
        # "111111" = all max level -> equal weights
        w = qubo.decode_weights("111111")
        assert abs(w.sum() - 1.0) < 1e-10
        assert abs(w[0] - 0.5) < 1e-10
