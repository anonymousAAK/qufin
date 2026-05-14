"""Unit tests for sector rotation with quantum regime detection."""

from __future__ import annotations

import numpy as np
import pytest

from qufin.backends.mock import MockBackend
from qufin.portfolio.sector_rotation import (
    DEFAULT_SECTORS,
    BacktestResult,
    Regime,
    RegimeDetector,
    RegimeDetectorConfig,
    SectorRotator,
    SectorWeightProfile,
    backtest_sector_rotation,
)

# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _synthetic_macro_data(
    n: int = 60, seed: int = 0
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """Generate synthetic macro features and regime labels.

    Returns (vix, yield_slope, pmi, credit_spread, labels).
    """
    rng = np.random.default_rng(seed)

    # Risk-on block
    n_on = n // 3
    # Risk-off block
    n_off = n // 3
    # Crisis block
    n_cr = n - n_on - n_off

    vix = np.concatenate([
        rng.uniform(10, 18, n_on),   # risk_on
        rng.uniform(21, 28, n_off),  # risk_off
        rng.uniform(32, 60, n_cr),   # crisis
    ])
    pmi = np.concatenate([
        rng.uniform(52, 60, n_on),
        rng.uniform(42, 49, n_off),
        rng.uniform(35, 45, n_cr),
    ])
    yield_slope = rng.uniform(-50, 200, n)
    credit_spread = rng.uniform(50, 400, n)

    labels = RegimeDetector.label_regimes(vix, pmi)
    return vix, yield_slope, pmi, credit_spread, labels


# -----------------------------------------------------------------------
# Regime detection
# -----------------------------------------------------------------------

class TestRegimeDetector:
    """Tests for the VQC-based regime detector."""

    @pytest.fixture
    def backend(self) -> MockBackend:
        return MockBackend(seed=42)

    def test_label_regimes_values(self) -> None:
        vix = np.array([12.0, 25.0, 40.0])
        pmi = np.array([55.0, 45.0, 38.0])
        labels = RegimeDetector.label_regimes(vix, pmi)
        assert labels[0] == Regime.RISK_ON
        assert labels[1] == Regime.RISK_OFF
        assert labels[2] == Regime.CRISIS

    def test_label_regimes_shape(self) -> None:
        n = 50
        vix = np.random.default_rng(0).uniform(10, 50, n)
        pmi = np.random.default_rng(1).uniform(40, 60, n)
        labels = RegimeDetector.label_regimes(vix, pmi)
        assert labels.shape == (n,)
        assert set(labels.tolist()).issubset({0, 1, 2})

    def test_build_features_shape(self) -> None:
        n = 20
        rng = np.random.default_rng(0)
        X = RegimeDetector.build_features(
            rng.uniform(size=n),
            rng.uniform(size=n),
            rng.uniform(size=n),
            rng.uniform(size=n),
        )
        assert X.shape == (n, 4)

    def test_fit_predict_runs(self, backend: MockBackend) -> None:
        """Smoke test: fit + predict completes without error."""
        vix, ys, pmi, cs, labels = _synthetic_macro_data(n=30, seed=7)
        X = RegimeDetector.build_features(vix, ys, pmi, cs)

        cfg = RegimeDetectorConfig(n_qubits=4, n_layers=1, n_epochs=5, seed=0)
        detector = RegimeDetector(cfg, backend)
        detector.fit(X, labels)

        preds = detector.predict(X)
        assert preds.shape == labels.shape
        assert set(preds.tolist()).issubset({0, 1, 2})

    def test_predict_before_fit_raises(self, backend: MockBackend) -> None:
        cfg = RegimeDetectorConfig()
        detector = RegimeDetector(cfg, backend)
        X = np.random.default_rng(0).uniform(size=(5, 4))
        with pytest.raises(RuntimeError, match="not been fitted"):
            detector.predict(X)

    def test_predict_regime_name(self, backend: MockBackend) -> None:
        vix, ys, pmi, cs, labels = _synthetic_macro_data(n=18, seed=3)
        X = RegimeDetector.build_features(vix, ys, pmi, cs)

        cfg = RegimeDetectorConfig(n_qubits=4, n_layers=1, n_epochs=3, seed=1)
        detector = RegimeDetector(cfg, backend)
        detector.fit(X, labels)

        names = detector.predict_regime_name(X)
        assert len(names) == len(labels)
        assert all(n in ("risk_on", "risk_off", "crisis") for n in names)


# -----------------------------------------------------------------------
# Sector weight allocation
# -----------------------------------------------------------------------

class TestSectorRotator:
    """Tests for deterministic sector weight allocation."""

    def test_allocate_sums_to_one(self) -> None:
        rotator = SectorRotator()
        for regime in (Regime.RISK_ON, Regime.RISK_OFF, Regime.CRISIS):
            w = rotator.allocate(regime)
            assert abs(sum(w.values()) - 1.0) < 1e-9

    def test_allocate_all_sectors_present(self) -> None:
        rotator = SectorRotator()
        for regime in (Regime.RISK_ON, Regime.RISK_OFF, Regime.CRISIS):
            w = rotator.allocate(regime)
            assert set(w.keys()) == set(DEFAULT_SECTORS)

    def test_allocate_array_shape(self) -> None:
        rotator = SectorRotator()
        arr = rotator.allocate_array(Regime.RISK_ON)
        assert arr.shape == (len(DEFAULT_SECTORS),)
        assert abs(arr.sum() - 1.0) < 1e-9

    def test_risk_on_favours_tech(self) -> None:
        rotator = SectorRotator()
        w = rotator.allocate(Regime.RISK_ON)
        assert w["Technology"] > w["Utilities"]

    def test_crisis_favours_defensives(self) -> None:
        rotator = SectorRotator()
        w = rotator.allocate(Regime.CRISIS)
        assert w["Utilities"] > w["Technology"]
        assert w["Consumer Staples"] > w["Consumer Discretionary"]

    def test_custom_sectors_and_profiles(self) -> None:
        sectors = ["A", "B", "C"]
        profiles = SectorWeightProfile(
            risk_on={"A": 0.6, "B": 0.3, "C": 0.1},
            risk_off={"A": 0.2, "B": 0.3, "C": 0.5},
            crisis={"A": 0.1, "B": 0.2, "C": 0.7},
        )
        rotator = SectorRotator(sectors=sectors, profiles=profiles)
        w = rotator.allocate(Regime.CRISIS)
        assert w["C"] == pytest.approx(0.7)
        assert abs(sum(w.values()) - 1.0) < 1e-9

    def test_allocate_timeseries_shape(self) -> None:
        rotator = SectorRotator()
        regimes = np.array([0, 1, 2, 0, 1], dtype=np.int64)
        wts = rotator.allocate_timeseries(regimes)
        assert wts.shape == (5, len(DEFAULT_SECTORS))
        np.testing.assert_allclose(wts.sum(axis=1), 1.0, atol=1e-9)

    def test_missing_sector_gets_residual(self) -> None:
        sectors = ["X", "Y", "Z"]
        profiles = SectorWeightProfile(risk_on={"X": 0.5})
        rotator = SectorRotator(sectors=sectors, profiles=profiles)
        w = rotator.allocate(Regime.RISK_ON)
        assert w["X"] == pytest.approx(0.5)
        assert w["Y"] == pytest.approx(0.25)
        assert w["Z"] == pytest.approx(0.25)


# -----------------------------------------------------------------------
# Backtest
# -----------------------------------------------------------------------

class TestBacktestSectorRotation:
    """Tests for the backtest function."""

    @pytest.fixture
    def synthetic_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Sector returns (T=100, 11 sectors) and regime sequence."""
        rng = np.random.default_rng(42)
        T = 100
        n_sectors = len(DEFAULT_SECTORS)
        returns = rng.normal(0.0005, 0.01, (T, n_sectors))
        regimes = np.array([0] * 40 + [1] * 30 + [2] * 30, dtype=np.int64)
        return returns, regimes

    def test_backtest_returns_dataclass(
        self, synthetic_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        returns, regimes = synthetic_data
        result = backtest_sector_rotation(returns, regimes)
        assert isinstance(result, BacktestResult)

    def test_backtest_strategy_returns_length(
        self, synthetic_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        returns, regimes = synthetic_data
        result = backtest_sector_rotation(returns, regimes)
        assert len(result.strategy_returns) == returns.shape[0]
        assert len(result.buy_hold_returns) == returns.shape[0]

    def test_backtest_metrics_finite(
        self, synthetic_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        returns, regimes = synthetic_data
        result = backtest_sector_rotation(returns, regimes)
        assert np.isfinite(result.total_return)
        assert np.isfinite(result.annualised_return)
        assert np.isfinite(result.annualised_vol)
        assert np.isfinite(result.sharpe_ratio)
        assert 0.0 <= result.max_drawdown <= 1.0

    def test_backtest_as_dict(
        self, synthetic_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        returns, regimes = synthetic_data
        result = backtest_sector_rotation(returns, regimes)
        d = result.as_dict()
        assert set(d.keys()) == {
            "total_return",
            "annualised_return",
            "annualised_vol",
            "sharpe_ratio",
            "max_drawdown",
            "buy_hold_return",
        }

    def test_backtest_buy_hold_equals_equal_weight(
        self, synthetic_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        returns, regimes = synthetic_data
        result = backtest_sector_rotation(returns, regimes)
        expected_bh = returns @ np.full(returns.shape[1], 1.0 / returns.shape[1])
        np.testing.assert_allclose(result.buy_hold_returns, expected_bh, atol=1e-12)

    def test_backtest_custom_profiles(self) -> None:
        rng = np.random.default_rng(99)
        sectors = ["A", "B"]
        T = 50
        ret = rng.normal(0.001, 0.01, (T, 2))
        regimes = np.zeros(T, dtype=np.int64)
        profiles = SectorWeightProfile(
            risk_on={"A": 0.8, "B": 0.2},
            risk_off={"A": 0.5, "B": 0.5},
            crisis={"A": 0.2, "B": 0.8},
        )
        result = backtest_sector_rotation(
            ret, regimes, sectors=sectors, profiles=profiles
        )
        assert np.isfinite(result.total_return)
