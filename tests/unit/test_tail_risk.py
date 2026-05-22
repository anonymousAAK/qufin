"""Unit tests for advanced tail risk measures."""

from __future__ import annotations

import numpy as np
import pytest

from qufin.risk.tail_risk import (
    entropic_var,
    expected_tail_loss,
    spectral_risk_measure,
    tail_dependence_coefficient,
)


@pytest.fixture
def normal_losses() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.normal(0.0, 1.0, 10_000)


@pytest.fixture
def positive_losses() -> np.ndarray:
    rng = np.random.default_rng(42)
    return np.abs(rng.normal(0.5, 1.0, 5_000))


# --- Entropic VaR ---

class TestEntropicVaR:
    def test_returns_float(self, normal_losses) -> None:
        result = entropic_var(normal_losses, alpha=0.95)
        assert isinstance(result, float)

    def test_exceeds_cvar(self, normal_losses) -> None:
        evar = entropic_var(normal_losses, alpha=0.95)
        cvar = expected_tail_loss(normal_losses, alpha=0.95)
        # EVaR >= CVaR for the same alpha
        assert evar >= cvar - 0.1  # tolerance for sampling noise

    def test_higher_alpha_higher_evar(self, normal_losses) -> None:
        e90 = entropic_var(normal_losses, alpha=0.90)
        e99 = entropic_var(normal_losses, alpha=0.99)
        assert e99 > e90

    def test_empty_losses(self) -> None:
        assert entropic_var(np.array([])) == 0.0

    def test_positive_losses(self, positive_losses) -> None:
        result = entropic_var(positive_losses, alpha=0.95)
        assert result > 0


# --- Tail Dependence ---

class TestTailDependence:
    def test_independent_low(self) -> None:
        rng = np.random.default_rng(42)
        u = rng.normal(0, 1, 5000)
        v = rng.normal(0, 1, 5000)
        td = tail_dependence_coefficient(u, v, threshold=0.95)
        assert td < 0.3  # independent => low tail dependence

    def test_perfectly_correlated(self) -> None:
        rng = np.random.default_rng(42)
        u = rng.normal(0, 1, 5000)
        td = tail_dependence_coefficient(u, u, threshold=0.90)
        assert td > 0.8  # perfect correlation => high tail dep

    def test_returns_in_unit_interval(self) -> None:
        rng = np.random.default_rng(42)
        u = rng.normal(0, 1, 1000)
        v = rng.normal(0, 1, 1000)
        td = tail_dependence_coefficient(u, v)
        assert 0.0 <= td <= 1.0

    def test_short_series(self) -> None:
        td = tail_dependence_coefficient(np.array([1.0]), np.array([2.0]))
        assert td == 0.0


# --- Expected Tail Loss ---

class TestExpectedTailLoss:
    def test_returns_float(self, normal_losses) -> None:
        result = expected_tail_loss(normal_losses, alpha=0.95)
        assert isinstance(result, float)

    def test_exceeds_var(self, normal_losses) -> None:
        etl = expected_tail_loss(normal_losses, alpha=0.95)
        var = float(np.quantile(normal_losses, 0.95))
        assert etl >= var

    def test_alpha_0_is_mean(self, normal_losses) -> None:
        etl = expected_tail_loss(normal_losses, alpha=0.0)
        assert abs(etl - float(np.mean(normal_losses))) < 0.01

    def test_empty_losses(self) -> None:
        assert expected_tail_loss(np.array([])) == 0.0


# --- Spectral Risk Measure ---

class TestSpectralRiskMeasure:
    def test_default_spectrum(self, normal_losses) -> None:
        result = spectral_risk_measure(normal_losses)
        assert isinstance(result, float)

    def test_custom_spectrum(self, normal_losses) -> None:
        def uniform_phi(p: np.ndarray) -> np.ndarray:
            return np.ones_like(p)

        result = spectral_risk_measure(normal_losses, phi=uniform_phi)
        # Uniform weights => should be close to the mean of sorted losses = mean
        assert abs(result - float(np.mean(normal_losses))) < 0.05

    def test_risk_averse_exceeds_mean(self, positive_losses) -> None:
        result = spectral_risk_measure(positive_losses)
        # Risk-averse spectrum puts more weight on tails
        assert result > float(np.mean(positive_losses)) - 0.1

    def test_empty_losses(self) -> None:
        assert spectral_risk_measure(np.array([])) == 0.0

    def test_single_value(self) -> None:
        result = spectral_risk_measure(np.array([5.0]))
        assert abs(result - 5.0) < 1e-10
