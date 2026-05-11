"""Unit tests for qufin.hedging modules (delta and deep_hedging)."""

from __future__ import annotations

import pytest

from qufin.hedging.deep_hedging import DeepHedger, DeepHedgingConfig
from qufin.hedging.delta import DeltaHedger, HedgeResult, bs_delta


class TestBsDelta:
    """Tests for Black-Scholes delta function."""

    def test_call_delta_in_0_1(self) -> None:
        d = bs_delta(100.0, 100.0, 0.05, 0.2, 1.0, is_call=True)
        assert 0.0 <= d <= 1.0

    def test_put_delta_in_neg1_0(self) -> None:
        d = bs_delta(100.0, 100.0, 0.05, 0.2, 1.0, is_call=False)
        assert -1.0 <= d <= 0.0

    def test_deep_itm_call_near_one(self) -> None:
        d = bs_delta(200.0, 100.0, 0.05, 0.2, 1.0, is_call=True)
        assert d > 0.9

    def test_deep_otm_call_near_zero(self) -> None:
        d = bs_delta(50.0, 100.0, 0.05, 0.2, 1.0, is_call=True)
        assert d < 0.1

    def test_put_call_parity(self) -> None:
        """call delta - put delta == 1."""
        dc = bs_delta(100.0, 100.0, 0.05, 0.2, 1.0, is_call=True)
        dp = bs_delta(100.0, 100.0, 0.05, 0.2, 1.0, is_call=False)
        assert dc - dp == pytest.approx(1.0, abs=1e-10)


class TestDeltaHedger:
    """Tests for DeltaHedger simulation."""

    def test_hedge_returns_hedge_result(self) -> None:
        hedger = DeltaHedger()
        result = hedger.hedge(
            spot=100.0, strike=100.0, r=0.05, sigma=0.2, T=1.0,
            n_rebalances=10, seed=42,
        )
        assert isinstance(result, HedgeResult)

    def test_hedge_result_has_fields(self) -> None:
        hedger = DeltaHedger()
        result = hedger.hedge(100.0, 100.0, 0.05, 0.2, 1.0, n_rebalances=10, seed=42)
        assert hasattr(result, "pnl")
        assert hasattr(result, "hedging_error")
        assert hasattr(result, "option_price")
        assert len(result.deltas) == 11  # n_rebalances + 1
        assert len(result.spot_path) == 11

    def test_deterministic_with_seed(self) -> None:
        hedger = DeltaHedger()
        r1 = hedger.hedge(100.0, 100.0, 0.05, 0.2, 1.0, 10, seed=7)
        r2 = hedger.hedge(100.0, 100.0, 0.05, 0.2, 1.0, 10, seed=7)
        assert r1.pnl == r2.pnl


class TestDeepHedgingConfig:
    """Tests for DeepHedgingConfig dataclass."""

    def test_default_creation(self) -> None:
        cfg = DeepHedgingConfig()
        assert cfg.n_layers > 0
        assert cfg.hidden_dim > 0
        assert cfg.n_epochs > 0
        assert cfg.lr > 0

    def test_custom_creation(self) -> None:
        cfg = DeepHedgingConfig(n_layers=3, hidden_dim=64, n_epochs=10)
        assert cfg.n_layers == 3
        assert cfg.hidden_dim == 64
        assert cfg.n_epochs == 10


class TestDeepHedger:
    """Tests for DeepHedger instantiation."""

    def test_can_be_instantiated_default(self) -> None:
        hedger = DeepHedger()
        assert hedger is not None
        assert hedger.cfg is not None

    def test_can_be_instantiated_with_config(self) -> None:
        cfg = DeepHedgingConfig(n_epochs=2, n_paths=64, n_steps=5)
        hedger = DeepHedger(cfg, s0=100.0, strike=100.0, seed=0)
        assert hedger.s0 == 100.0
        assert hedger.strike == 100.0
        assert len(hedger.params) > 0
