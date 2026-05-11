"""Unit tests for qufin.risk.stress module."""

from __future__ import annotations

import numpy as np
import pytest

from qufin.risk.stress import (
    SCENARIO_LIBRARY,
    StressScenario,
    apply_stress,
    stress_test_suite,
)


class TestScenarioLibrary:
    """Tests for the pre-defined scenario library."""

    def test_has_four_scenarios(self) -> None:
        assert len(SCENARIO_LIBRARY) == 4

    def test_scenario_names(self) -> None:
        expected = {"Black Monday 1987", "GFC 2008", "COVID 2020", "Rates 2022"}
        assert set(SCENARIO_LIBRARY.keys()) == expected


class TestStressScenarioFields:
    """Tests for StressScenario dataclass fields."""

    def test_equity_shock_field(self) -> None:
        s = SCENARIO_LIBRARY["Black Monday 1987"]
        assert isinstance(s.equity_shock, float)
        assert s.equity_shock < 0

    def test_vol_shock_field(self) -> None:
        s = SCENARIO_LIBRARY["COVID 2020"]
        assert isinstance(s.vol_shock, float)
        assert s.vol_shock > 0

    def test_rates_shock_field(self) -> None:
        s = SCENARIO_LIBRARY["Rates 2022"]
        assert isinstance(s.rates_shock, float)

    def test_spread_shock_field(self) -> None:
        s = SCENARIO_LIBRARY["GFC 2008"]
        assert isinstance(s.spread_shock, float)
        assert s.spread_shock > 0

    def test_name_and_date_fields(self) -> None:
        s = SCENARIO_LIBRARY["GFC 2008"]
        assert s.name == "GFC 2008"
        assert isinstance(s.date, str)

    def test_custom_scenario_creation(self) -> None:
        sc = StressScenario(
            name="Custom", date="2025-01-01",
            equity_shock=-0.10, rates_shock=-25.0, vol_shock=0.5,
        )
        assert sc.name == "Custom"
        assert sc.spread_shock == 0.0  # default


class TestApplyStress:
    """Tests for the apply_stress function."""

    def test_negative_pnl_for_crash_equity(self) -> None:
        scenario = SCENARIO_LIBRARY["Black Monday 1987"]
        weights = np.array([1.0, 0.0, 0.0, 0.0])
        result = apply_stress(1_000_000.0, weights, scenario)
        assert result["total_pnl"] < 0

    def test_negative_pnl_for_gfc(self) -> None:
        scenario = SCENARIO_LIBRARY["GFC 2008"]
        weights = np.array([0.5, 0.0, 0.0, 0.5])
        result = apply_stress(1_000_000.0, weights, scenario)
        assert result["total_pnl"] < 0

    def test_zero_exposure_gives_zero_pnl(self) -> None:
        scenario = SCENARIO_LIBRARY["COVID 2020"]
        weights = np.zeros(4)
        result = apply_stress(1_000_000.0, weights, scenario)
        assert result["total_pnl"] == pytest.approx(0.0)

    def test_result_keys(self) -> None:
        scenario = SCENARIO_LIBRARY["Black Monday 1987"]
        weights = np.array([0.25, 0.25, 0.25, 0.25])
        result = apply_stress(100_000.0, weights, scenario)
        for key in ("scenario", "equity_pnl", "rates_pnl", "vol_pnl",
                     "spread_pnl", "total_pnl", "pct_loss"):
            assert key in result

    def test_bad_weights_shape_raises(self) -> None:
        scenario = SCENARIO_LIBRARY["Rates 2022"]
        with pytest.raises(ValueError):
            apply_stress(100_000.0, np.array([1.0, 0.0]), scenario)


class TestStressTestSuite:
    """Tests for stress_test_suite running all scenarios."""

    def test_runs_all_scenarios(self) -> None:
        weights = np.array([0.6, 0.2, 0.1, 0.1])
        results = stress_test_suite(1_000_000.0, weights)
        assert len(results) == 4
        for name in SCENARIO_LIBRARY:
            assert name in results

    def test_custom_subset(self) -> None:
        weights = np.array([1.0, 0.0, 0.0, 0.0])
        subset = [SCENARIO_LIBRARY["Black Monday 1987"]]
        results = stress_test_suite(500_000.0, weights, scenarios=subset)
        assert len(results) == 1
        assert "Black Monday 1987" in results
