"""Unit tests for counterparty credit risk."""

from __future__ import annotations

import numpy as np
import pytest

from qufin.risk.counterparty import (
    CounterpartyExposure,
    compute_cva,
    compute_ead_sa_ccr,
    portfolio_cva,
)


@pytest.fixture
def sample_exposure() -> CounterpartyExposure:
    return CounterpartyExposure(
        name="Bank_A",
        notional=1_000_000,
        pd=0.02,
        lgd=0.6,
        exposure_profile=np.array([50_000, 80_000, 120_000, 90_000, 60_000]),
    )


class TestCounterpartyExposure:
    def test_epe(self, sample_exposure: CounterpartyExposure) -> None:
        assert sample_exposure.epe == pytest.approx(80_000)

    def test_peak(self, sample_exposure: CounterpartyExposure) -> None:
        assert sample_exposure.peak_exposure == 120_000


class TestCVA:
    def test_positive_cva(self, sample_exposure: CounterpartyExposure) -> None:
        cva = compute_cva(sample_exposure)
        assert cva > 0

    def test_higher_pd_higher_cva(self) -> None:
        profile = np.array([100_000, 100_000, 100_000])
        low_pd = CounterpartyExposure("A", 1e6, 0.01, 0.6, profile)
        high_pd = CounterpartyExposure("B", 1e6, 0.05, 0.6, profile)
        assert compute_cva(high_pd) > compute_cva(low_pd)

    def test_zero_exposure(self) -> None:
        exp = CounterpartyExposure("C", 1e6, 0.02, 0.6, np.zeros(5))
        assert compute_cva(exp) == 0.0


class TestEAD:
    def test_in_the_money(self) -> None:
        ead = compute_ead_sa_ccr(notional=1e6, mtm=50_000, add_on_factor=0.01)
        assert ead > 0

    def test_out_of_money(self) -> None:
        ead = compute_ead_sa_ccr(notional=1e6, mtm=-50_000, add_on_factor=0.01)
        assert ead > 0  # PFE still contributes

    def test_with_collateral(self) -> None:
        ead_no_coll = compute_ead_sa_ccr(notional=1e6, mtm=50_000)
        ead_with_coll = compute_ead_sa_ccr(notional=1e6, mtm=50_000, collateral=30_000)
        assert ead_with_coll < ead_no_coll


class TestPortfolioCVA:
    def test_portfolio(self) -> None:
        profile = np.array([100_000, 80_000, 60_000])
        exposures = [
            CounterpartyExposure("A", 1e6, 0.02, 0.6, profile),
            CounterpartyExposure("B", 2e6, 0.03, 0.5, profile * 1.5),
        ]
        result = portfolio_cva(exposures)
        assert "A" in result
        assert "B" in result
        assert "total" in result
        assert result["total"] == pytest.approx(result["A"] + result["B"])
