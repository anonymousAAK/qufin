"""Tests for qufin.data.quality — data quality framework."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from qufin.data.quality import (
    DataLineage,
    DividendEvent,
    SplitEvent,
    adjust_for_dividends,
    adjust_for_splits,
    compute_quality_score,
    compute_quality_scores,
    detect_gaps,
    detect_outliers,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_prices(
    start: str = "2024-01-02",
    periods: int = 20,
    freq: str = "B",
    base: float = 100.0,
    seed: int = 42,
) -> pd.Series:
    """Create a synthetic daily price series."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=periods)
    returns = rng.normal(0.0005, 0.01, size=periods)
    prices = base * np.exp(np.cumsum(returns))
    return pd.Series(prices, index=idx, name="TEST")


# ---------------------------------------------------------------------------
# DataLineage tests
# ---------------------------------------------------------------------------

class TestDataLineage:
    def test_create_lineage(self):
        lineage = DataLineage(ticker="AAPL", source="yahoo")
        assert lineage.ticker == "AAPL"
        assert lineage.source == "yahoo"
        assert len(lineage.transformations) == 0

    def test_add_step(self):
        lineage = DataLineage(ticker="MSFT", source="bloomberg")
        lineage.add_step("split_adjust", "Applied 4:1 split", {"ratio": 4.0})
        assert len(lineage.transformations) == 1
        step = lineage.transformations[0]
        assert step.name == "split_adjust"
        assert step.parameters["ratio"] == 4.0

    def test_multiple_steps(self):
        lineage = DataLineage(ticker="GOOG", source="refinitiv")
        lineage.add_step("split_adjust", "split")
        lineage.add_step("dividend_adjust", "div")
        lineage.add_step("outlier_review", "flagged")
        assert len(lineage.transformations) == 3
        assert [s.name for s in lineage.transformations] == [
            "split_adjust", "dividend_adjust", "outlier_review"
        ]


# ---------------------------------------------------------------------------
# Gap detection tests
# ---------------------------------------------------------------------------

class TestDetectGaps:
    def test_no_gaps(self):
        prices = _make_prices(periods=10)
        report = detect_gaps(prices)
        assert report.gap_fraction == 0.0
        assert report.missing_dates == []
        assert report.total_expected == report.total_present

    def test_with_gaps(self):
        prices = _make_prices(periods=10)
        # Drop the 3rd and 5th business days
        prices = prices.drop(prices.index[[2, 4]])
        report = detect_gaps(prices)
        assert len(report.missing_dates) == 2
        assert report.total_present == 8
        assert report.gap_fraction == pytest.approx(2 / 10)

    def test_empty_series(self):
        prices = pd.Series([], dtype=float, index=pd.DatetimeIndex([]))
        report = detect_gaps(prices)
        assert report.total_expected == 0
        assert report.gap_fraction == 0.0

    def test_single_day(self):
        idx = pd.DatetimeIndex([pd.Timestamp("2024-01-02")])
        prices = pd.Series([100.0], index=idx)
        report = detect_gaps(prices)
        assert report.total_expected == 1
        assert report.gap_fraction == 0.0

    def test_custom_calendar(self):
        # Create 7-day calendar, but prices only cover 5 of those days
        cal = pd.bdate_range("2024-01-02", periods=7)
        prices = pd.Series(100.0, index=cal)
        # Drop 2 days to simulate gaps
        prices = prices.drop(cal[[2, 4]])
        report = detect_gaps(prices, trading_calendar=cal)
        assert len(report.missing_dates) == 2

    def test_dataframe_input(self):
        prices = _make_prices(periods=10)
        df = prices.to_frame("AAPL")
        report = detect_gaps(df)
        assert report.gap_fraction == 0.0


# ---------------------------------------------------------------------------
# Outlier detection tests
# ---------------------------------------------------------------------------

class TestDetectOutliers:
    def test_no_outliers_calm_data(self):
        # Very calm price series — no 5-sigma events
        idx = pd.bdate_range("2024-01-02", periods=100)
        prices = pd.Series(100.0 + np.arange(100) * 0.01, index=idx)
        report = detect_outliers(prices)
        assert len(report.outlier_dates) == 0
        assert report.sigma_threshold == 5.0

    def test_injected_outlier(self):
        idx = pd.bdate_range("2024-01-02", periods=100)
        rng = np.random.default_rng(0)
        prices_arr = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 100)))
        prices = pd.Series(prices_arr, index=idx)
        # Inject a massive spike
        prices.iloc[50] = prices.iloc[49] * 1.50  # +50% daily
        report = detect_outliers(prices, sigma_threshold=5.0)
        assert len(report.outlier_dates) >= 1

    def test_custom_threshold(self):
        idx = pd.bdate_range("2024-01-02", periods=50)
        rng = np.random.default_rng(7)
        prices = pd.Series(
            100 * np.exp(np.cumsum(rng.normal(0, 0.02, 50))), index=idx
        )
        report_tight = detect_outliers(prices, sigma_threshold=1.0)
        report_loose = detect_outliers(prices, sigma_threshold=10.0)
        assert len(report_tight.outlier_dates) >= len(report_loose.outlier_dates)

    def test_empty_series(self):
        prices = pd.Series([], dtype=float, index=pd.DatetimeIndex([]))
        report = detect_outliers(prices)
        assert report.mean == 0.0
        assert report.std == 0.0

    def test_single_price(self):
        prices = pd.Series([100.0], index=pd.DatetimeIndex(["2024-01-02"]))
        report = detect_outliers(prices)
        assert len(report.outlier_dates) == 0

    def test_constant_prices(self):
        idx = pd.bdate_range("2024-01-02", periods=20)
        prices = pd.Series(100.0, index=idx)
        report = detect_outliers(prices)
        assert report.std == 0.0
        assert len(report.outlier_dates) == 0


# ---------------------------------------------------------------------------
# Corporate action tests
# ---------------------------------------------------------------------------

class TestAdjustForSplits:
    def test_two_for_one_split(self):
        idx = pd.bdate_range("2024-01-02", periods=10)
        prices = pd.Series(200.0, index=idx)
        split = SplitEvent(date=dt.date(2024, 1, 9), ratio=2.0)
        adjusted = adjust_for_splits(prices, [split])
        # Prices before Jan 9 should be halved
        before = adjusted[adjusted.index < pd.Timestamp("2024-01-09")]
        after = adjusted[adjusted.index >= pd.Timestamp("2024-01-09")]
        assert all(before == 100.0)
        assert all(after == 200.0)

    def test_no_splits(self):
        prices = _make_prices()
        adjusted = adjust_for_splits(prices, [])
        pd.testing.assert_series_equal(adjusted, prices)

    def test_empty_prices(self):
        prices = pd.Series([], dtype=float, index=pd.DatetimeIndex([]))
        adjusted = adjust_for_splits(prices, [SplitEvent(dt.date(2024, 1, 5), 2.0)])
        assert adjusted.empty

    def test_lineage_recorded(self):
        prices = _make_prices(periods=5)
        lineage = DataLineage(ticker="TEST", source="test")
        split = SplitEvent(date=dt.date(2024, 1, 4), ratio=3.0)
        adjust_for_splits(prices, [split], lineage=lineage)
        assert len(lineage.transformations) == 1
        assert lineage.transformations[0].name == "split_adjust"


class TestAdjustForDividends:
    def test_dividend_adjustment(self):
        idx = pd.bdate_range("2024-01-02", periods=10)
        prices = pd.Series(100.0, index=idx)
        div = DividendEvent(date=dt.date(2024, 1, 9), amount=2.0)
        adjusted = adjust_for_dividends(prices, [div])
        before = adjusted[adjusted.index < pd.Timestamp("2024-01-09")]
        after = adjusted[adjusted.index >= pd.Timestamp("2024-01-09")]
        expected_factor = (100.0 - 2.0) / 100.0
        np.testing.assert_allclose(before.values, 100.0 * expected_factor)
        np.testing.assert_allclose(after.values, 100.0)

    def test_no_dividends(self):
        prices = _make_prices()
        adjusted = adjust_for_dividends(prices, [])
        pd.testing.assert_series_equal(adjusted, prices)

    def test_empty_prices(self):
        prices = pd.Series([], dtype=float, index=pd.DatetimeIndex([]))
        adjusted = adjust_for_dividends(
            prices, [DividendEvent(dt.date(2024, 1, 5), 1.0)]
        )
        assert adjusted.empty

    def test_lineage_recorded(self):
        prices = _make_prices(periods=5)
        lineage = DataLineage(ticker="TEST", source="test")
        div = DividendEvent(date=dt.date(2024, 1, 4), amount=1.0)
        adjust_for_dividends(prices, [div], lineage=lineage)
        assert len(lineage.transformations) == 1
        assert lineage.transformations[0].name == "dividend_adjust"


# ---------------------------------------------------------------------------
# Quality score tests
# ---------------------------------------------------------------------------

class TestQualityScore:
    def test_perfect_score(self):
        prices = _make_prices(periods=20)
        ref = prices.index[-1].date()
        score = compute_quality_score(prices, "TEST", reference_date=ref)
        assert score.completeness == 1.0
        assert score.freshness == pytest.approx(1.0)
        assert score.consistency == pytest.approx(1.0, abs=0.05)
        assert 0.0 <= score.overall <= 1.0

    def test_stale_data(self):
        prices = _make_prices(periods=20)
        # Reference date 90 days after last data point
        ref = prices.index[-1].date() + dt.timedelta(days=90)
        score = compute_quality_score(
            prices, "STALE", reference_date=ref, freshness_halflife_days=30
        )
        # 90 days = 3 half-lives → freshness ~ 0.125
        assert score.freshness < 0.2

    def test_empty_prices(self):
        prices = pd.Series([], dtype=float, index=pd.DatetimeIndex([]))
        score = compute_quality_score(prices, "EMPTY")
        assert score.completeness == 1.0  # 0/0 → no gaps
        assert score.freshness == 0.0
        assert score.consistency == 1.0

    def test_custom_weights(self):
        prices = _make_prices(periods=20)
        ref = prices.index[-1].date()
        score = compute_quality_score(
            prices, "W", reference_date=ref, weights=(1.0, 0.0, 0.0)
        )
        assert score.overall == pytest.approx(score.completeness)

    def test_compute_quality_scores_multi(self):
        p1 = _make_prices(periods=20, seed=1)
        p2 = _make_prices(periods=20, seed=2)
        df = pd.DataFrame({"A": p1, "B": p2})
        ref = df.index[-1].date()
        scores = compute_quality_scores(df, reference_date=ref)
        assert set(scores.keys()) == {"A", "B"}
        for s in scores.values():
            assert 0.0 <= s.overall <= 1.0
