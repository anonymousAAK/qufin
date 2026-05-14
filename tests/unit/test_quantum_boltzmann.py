"""Unit tests for qufin.ml.quantum_boltzmann module."""

from __future__ import annotations

import numpy as np
import pytest

from qufin.backends.mock import MockBackend
from qufin.ml.quantum_boltzmann import (
    ClassicalRBM,
    HMMRegimeDetector,
    MarketRegime,
    RegimeBacktestResult,
    RestrictedQuantumBoltzmannMachine,
    RQBMConfig,
    RQBMResult,
    backtest_regime_strategy,
    prepare_market_indicators,
    regime_conditional_allocation,
)

# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------


@pytest.fixture
def backend() -> MockBackend:
    return MockBackend(seed=42)


@pytest.fixture
def config() -> RQBMConfig:
    return RQBMConfig(n_visible=3, n_hidden=2, n_epochs=3, seed=42)


@pytest.fixture
def rqbm(config: RQBMConfig, backend: MockBackend) -> RestrictedQuantumBoltzmannMachine:
    return RestrictedQuantumBoltzmannMachine(config, backend)


@pytest.fixture
def sample_data() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.uniform(0, 1, (10, 3))


# -----------------------------------------------------------------------
# MarketRegime enum
# -----------------------------------------------------------------------


class TestMarketRegime:
    def test_values(self) -> None:
        assert MarketRegime.RISK_ON == 0
        assert MarketRegime.RISK_OFF == 1
        assert MarketRegime.CRISIS == 2
        assert MarketRegime.RECOVERY == 3

    def test_len(self) -> None:
        assert len(MarketRegime) == 4


# -----------------------------------------------------------------------
# RQBMConfig
# -----------------------------------------------------------------------


class TestRQBMConfig:
    def test_defaults(self) -> None:
        cfg = RQBMConfig()
        assert cfg.n_visible == 6
        assert cfg.n_hidden == 4
        assert cfg.n_epochs == 50

    def test_custom(self) -> None:
        cfg = RQBMConfig(n_visible=8, n_hidden=5, temperature=2.0)
        assert cfg.n_visible == 8
        assert cfg.temperature == 2.0


# -----------------------------------------------------------------------
# RestrictedQuantumBoltzmannMachine
# -----------------------------------------------------------------------


class TestRQBM:
    def test_instantiation(self, rqbm: RestrictedQuantumBoltzmannMachine) -> None:
        assert rqbm.config.n_visible == 3
        assert rqbm.config.n_hidden == 2
        assert rqbm.weights.shape == (3, 2)

    def test_sigmoid(self, rqbm: RestrictedQuantumBoltzmannMachine) -> None:
        x = np.array([0.0, 10.0, -10.0])
        s = rqbm._sigmoid(x)
        np.testing.assert_allclose(s[0], 0.5, atol=1e-10)
        assert s[1] > 0.99
        assert s[2] < 0.01

    def test_sample_hidden_single(
        self, rqbm: RestrictedQuantumBoltzmannMachine
    ) -> None:
        v = np.array([0.5, 0.3, 0.8])
        h = rqbm.sample_hidden(v)
        assert h.shape == (2,)
        assert np.all(h >= 0.0)
        assert np.all(h <= 1.0)

    def test_sample_hidden_batch(
        self, rqbm: RestrictedQuantumBoltzmannMachine
    ) -> None:
        v = np.array([[0.5, 0.3, 0.8], [0.1, 0.9, 0.2]])
        h = rqbm.sample_hidden(v)
        assert h.shape == (2, 2)

    def test_sample_visible(
        self, rqbm: RestrictedQuantumBoltzmannMachine
    ) -> None:
        h = np.array([0.5, 0.5])
        v = rqbm.sample_visible(h)
        assert v.shape == (3,)
        assert np.all(v >= 0.0)
        assert np.all(v <= 1.0)

    def test_reconstruction_error_nonnegative(
        self,
        rqbm: RestrictedQuantumBoltzmannMachine,
        sample_data: np.ndarray,
    ) -> None:
        err = rqbm.reconstruction_error(sample_data)
        assert err >= 0.0

    def test_fit_returns_result(
        self,
        rqbm: RestrictedQuantumBoltzmannMachine,
        sample_data: np.ndarray,
    ) -> None:
        result = rqbm.fit(sample_data)
        assert isinstance(result, RQBMResult)
        assert len(result.loss_history) == 3
        assert result.weights.shape == (3, 2)
        assert result.wall_time_s > 0.0

    def test_fit_regime_labels(
        self,
        rqbm: RestrictedQuantumBoltzmannMachine,
        sample_data: np.ndarray,
    ) -> None:
        result = rqbm.fit(sample_data)
        assert result.regime_labels.shape == (10,)
        assert np.all(result.regime_labels >= 0)
        assert np.all(result.regime_labels < 4)

    def test_fit_regime_probabilities_sum_to_one(
        self,
        rqbm: RestrictedQuantumBoltzmannMachine,
        sample_data: np.ndarray,
    ) -> None:
        result = rqbm.fit(sample_data)
        row_sums = result.regime_probabilities.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)

    def test_extract_features(
        self,
        rqbm: RestrictedQuantumBoltzmannMachine,
        sample_data: np.ndarray,
    ) -> None:
        features = rqbm.extract_features(sample_data)
        assert features.shape == (10, 2)

    def test_classify_regimes_empty(
        self, rqbm: RestrictedQuantumBoltzmannMachine
    ) -> None:
        empty = np.empty((0, 2), dtype=np.float64)
        labels, probs = rqbm.classify_regimes(empty)
        assert labels.shape == (0,)
        assert probs.shape == (0, 4)

    def test_build_sampling_circuit(
        self, rqbm: RestrictedQuantumBoltzmannMachine
    ) -> None:
        pytest.importorskip("qiskit")
        v = np.array([0.5, 0.3, 0.8])
        circ = rqbm._build_sampling_circuit(v)
        assert circ.num_qubits == 2

    def test_momentum_updates(self, backend: MockBackend) -> None:
        cfg = RQBMConfig(n_visible=3, n_hidden=2, n_epochs=2, momentum=0.9, seed=42)
        rqbm = RestrictedQuantumBoltzmannMachine(cfg, backend)
        data = np.random.default_rng(0).uniform(0, 1, (5, 3))
        result = rqbm.fit(data)
        assert len(result.loss_history) == 2


# -----------------------------------------------------------------------
# prepare_market_indicators
# -----------------------------------------------------------------------


class TestPrepareMarketIndicators:
    def test_basic(self) -> None:
        vix = np.array([15.0, 25.0, 35.0])
        yld = np.array([1.0, 0.5, -0.5])
        mom = np.array([0.05, -0.02, 0.10])
        result = prepare_market_indicators(vix, yld, mom)
        assert result.shape == (3, 3)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_with_extra(self) -> None:
        n = 5
        vix = np.ones(n) * 20
        yld = np.ones(n) * 1.0
        mom = np.ones(n) * 0.05
        extra = [np.linspace(0, 1, n)]
        result = prepare_market_indicators(vix, yld, mom, extra_indicators=extra)
        assert result.shape == (n, 4)

    def test_constant_column(self) -> None:
        """Constant columns should not cause division by zero."""
        vix = np.ones(3) * 20
        yld = np.array([1.0, 2.0, 3.0])
        mom = np.array([0.0, 0.1, 0.2])
        result = prepare_market_indicators(vix, yld, mom)
        assert not np.any(np.isnan(result))


# -----------------------------------------------------------------------
# regime_conditional_allocation
# -----------------------------------------------------------------------


class TestRegimeAllocation:
    @pytest.mark.parametrize("regime", [0, 1, 2, 3])
    def test_weights_sum_to_one(self, regime: int) -> None:
        alloc = regime_conditional_allocation(regime, n_assets=3)
        np.testing.assert_allclose(alloc.sum(), 1.0, atol=1e-12)

    def test_crisis_mostly_risk_free(self) -> None:
        alloc = regime_conditional_allocation(MarketRegime.CRISIS, n_assets=3)
        assert alloc[-1] > 0.5  # risk-free > 50%

    def test_risk_on_high_equity(self) -> None:
        alloc = regime_conditional_allocation(MarketRegime.RISK_ON, n_assets=3)
        assert alloc[:3].sum() > 0.8

    def test_unknown_regime(self) -> None:
        alloc = regime_conditional_allocation(99, n_assets=3)
        np.testing.assert_allclose(alloc.sum(), 1.0, atol=1e-12)


# -----------------------------------------------------------------------
# backtest_regime_strategy
# -----------------------------------------------------------------------


class TestBacktestRegimeStrategy:
    def test_basic_backtest(self) -> None:
        n_periods = 20
        rng = np.random.default_rng(42)
        regimes = rng.integers(0, 4, size=n_periods).astype(np.int64)
        returns = rng.normal(0.001, 0.02, (n_periods, 3))
        result = backtest_regime_strategy(regimes, returns)
        assert isinstance(result, RegimeBacktestResult)
        assert result.portfolio_returns.shape == (n_periods,)
        assert result.max_drawdown >= 0.0

    def test_sharpe_ratio_finite(self) -> None:
        regimes = np.zeros(10, dtype=np.int64)
        returns = np.random.default_rng(0).normal(0.001, 0.01, (10, 2))
        result = backtest_regime_strategy(regimes, returns)
        assert np.isfinite(result.sharpe_ratio)


# -----------------------------------------------------------------------
# ClassicalRBM baseline
# -----------------------------------------------------------------------


class TestClassicalRBM:
    def test_fit_returns_losses(self) -> None:
        rbm = ClassicalRBM(n_visible=3, n_hidden=2, seed=42)
        data = np.random.default_rng(0).uniform(0, 1, (10, 3))
        losses = rbm.fit(data, n_epochs=5)
        assert len(losses) == 5
        assert all(l >= 0 for l in losses)

    def test_sample_hidden_shape(self) -> None:
        rbm = ClassicalRBM(n_visible=4, n_hidden=3, seed=0)
        v = np.random.default_rng(1).uniform(0, 1, (5, 4))
        h = rbm.sample_hidden(v)
        assert h.shape == (5, 3)


# -----------------------------------------------------------------------
# HMMRegimeDetector baseline
# -----------------------------------------------------------------------


class TestHMMRegimeDetector:
    def test_fit_predict(self) -> None:
        hmm = HMMRegimeDetector(n_regimes=3, n_features=2, seed=42)
        data = np.random.default_rng(0).normal(0, 1, (20, 2))
        hmm.fit(data)
        labels = hmm.predict(data)
        assert labels.shape == (20,)
        assert np.all(labels >= 0)
        assert np.all(labels < 3)

    def test_predict_before_fit_raises(self) -> None:
        hmm = HMMRegimeDetector(n_regimes=2, n_features=2)
        data = np.ones((5, 2))
        with pytest.raises(RuntimeError, match="fit"):
            hmm.predict(data)
