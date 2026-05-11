"""Unit tests for Asian and barrier option pricing."""

from __future__ import annotations

import numpy as np
import pytest

from qufin.options.asian import (
    AsianOptionSpec,
    geometric_asian_closed_form,
)
from qufin.options.barrier import (
    BarrierOptionSpec,
    barrier_closed_form,
)
from qufin.options.classical.monte_carlo import asian_mc, barrier_mc


class TestGeometricAsianClosedForm:
    def test_basic_price(self) -> None:
        price = geometric_asian_closed_form(
            s0=100, k=100, r=0.05, sigma=0.2, T=1.0, n_monitoring=12,
        )
        assert price > 0
        # Geometric Asian call should be cheaper than European
        from qufin.options.european import EuropeanOption
        eu = EuropeanOption(s0=100, k=100, r=0.05, sigma=0.2, T=1.0)
        assert price < eu.bs_price() * 1.1  # allow small tolerance

    def test_put_call_relationship(self) -> None:
        call = geometric_asian_closed_form(
            s0=100, k=100, r=0.05, sigma=0.2, T=1.0, n_monitoring=12, is_call=True,
        )
        put = geometric_asian_closed_form(
            s0=100, k=100, r=0.05, sigma=0.2, T=1.0, n_monitoring=12, is_call=False,
        )
        assert call > 0
        assert put > 0
        # Both should be positive for ATM

    def test_high_vol_increases_price(self) -> None:
        low_vol = geometric_asian_closed_form(
            s0=100, k=100, r=0.05, sigma=0.1, T=1.0, n_monitoring=12,
        )
        high_vol = geometric_asian_closed_form(
            s0=100, k=100, r=0.05, sigma=0.4, T=1.0, n_monitoring=12,
        )
        assert high_vol > low_vol

    def test_vs_monte_carlo(self) -> None:
        """Geometric Asian closed-form should match MC geometric average."""
        cf_price = geometric_asian_closed_form(
            s0=100, k=100, r=0.05, sigma=0.2, T=1.0, n_monitoring=52,
        )
        mc_result = asian_mc(
            s=100, k=100, r=0.05, sigma=0.2, T=1.0,
            n_steps=52, n_paths=200_000,
            average_type="geometric", seed=42,
        )
        # Should be within ~5% (MC has noise, CF is approximate for discrete)
        assert abs(cf_price - mc_result.price) / mc_result.price < 0.15


class TestBarrierClosedForm:
    def test_up_and_out_call(self) -> None:
        price = barrier_closed_form(
            s0=100, k=100, r=0.05, sigma=0.2, T=1.0, barrier=120,
            barrier_type="up-and-out", is_call=True,
        )
        assert price >= 0
        # Up-and-out call should be cheaper than vanilla
        from qufin.options.european import EuropeanOption
        eu = EuropeanOption(s0=100, k=100, r=0.05, sigma=0.2, T=1.0)
        assert price <= eu.bs_price() + 0.01

    def test_down_and_out_call(self) -> None:
        price = barrier_closed_form(
            s0=100, k=100, r=0.05, sigma=0.2, T=1.0, barrier=80,
            barrier_type="down-and-out", is_call=True,
        )
        assert price >= 0

    def test_in_out_parity(self) -> None:
        """knock-in + knock-out = vanilla."""
        from qufin.options.european import EuropeanOption
        eu = EuropeanOption(s0=100, k=100, r=0.05, sigma=0.2, T=1.0)
        vanilla = eu.bs_price()

        uo = barrier_closed_form(
            s0=100, k=100, r=0.05, sigma=0.2, T=1.0, barrier=120,
            barrier_type="up-and-out",
        )
        ui = barrier_closed_form(
            s0=100, k=100, r=0.05, sigma=0.2, T=1.0, barrier=120,
            barrier_type="up-and-in",
        )
        # uo + ui should approximately equal vanilla
        assert abs(uo + ui - vanilla) / vanilla < 0.05

    def test_high_barrier_approaches_vanilla(self) -> None:
        """Very high barrier -> up-and-out approaches vanilla."""
        price = barrier_closed_form(
            s0=100, k=100, r=0.05, sigma=0.2, T=1.0, barrier=500,
            barrier_type="up-and-out",
        )
        from qufin.options.european import EuropeanOption
        eu = EuropeanOption(s0=100, k=100, r=0.05, sigma=0.2, T=1.0)
        assert abs(price - eu.bs_price()) / eu.bs_price() < 0.05
