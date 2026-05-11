"""Unit tests for exotic derivatives modules."""

from __future__ import annotations

import numpy as np
import pytest

from qufin.options.bermudan import BermudanOptionSpec, bermudan_binomial
from qufin.derivatives.bermudan_lsm import lsm_price
from qufin.derivatives.path_dependent import (
    LookbackOptionSpec,
    lookback_mc,
    cliquet_mc,
)
from qufin.derivatives.autocallable import (
    AutocallableSpec,
    autocallable_mc,
    resource_estimate_chakrabarti,
)


# -----------------------------------------------------------------------
# Bermudan binomial tree
# -----------------------------------------------------------------------

class TestBermudanBinomial:
    """Tests for bermudan_binomial pricing."""

    def test_call_price_positive(self) -> None:
        spec = BermudanOptionSpec(
            s0=100.0, k=100.0, r=0.05, sigma=0.2, T=1.0,
            exercise_dates=[0.25, 0.5, 0.75, 1.0],
            is_call=True, n_steps=50,
        )
        result = bermudan_binomial(spec)
        assert result.price > 0

    def test_put_price_positive(self) -> None:
        spec = BermudanOptionSpec(
            s0=100.0, k=100.0, r=0.05, sigma=0.2, T=1.0,
            exercise_dates=[0.5, 1.0],
            is_call=False, n_steps=50,
        )
        result = bermudan_binomial(spec)
        assert result.price > 0

    def test_bermudan_geq_european(self) -> None:
        """Bermudan price >= European price (early exercise premium)."""
        # European = Bermudan with exercise only at maturity
        european_spec = BermudanOptionSpec(
            s0=100.0, k=105.0, r=0.05, sigma=0.2, T=1.0,
            exercise_dates=[1.0],  # only at maturity
            is_call=False, n_steps=100,
        )
        bermudan_spec = BermudanOptionSpec(
            s0=100.0, k=105.0, r=0.05, sigma=0.2, T=1.0,
            exercise_dates=[0.25, 0.5, 0.75, 1.0],
            is_call=False, n_steps=100,
        )
        euro_price = bermudan_binomial(european_spec).price
        berm_price = bermudan_binomial(bermudan_spec).price
        assert berm_price >= euro_price - 1e-10


# -----------------------------------------------------------------------
# LSM Monte Carlo
# -----------------------------------------------------------------------

class TestLSMPrice:
    """Tests for Longstaff-Schwartz Monte Carlo pricer."""

    def test_put_price_positive(self) -> None:
        result = lsm_price(
            s0=100.0, k=100.0, r=0.05, sigma=0.2, T=1.0,
            n_steps=20, n_paths=5_000, is_call=False, seed=42,
        )
        assert result["price"] > 0

    def test_call_price_positive(self) -> None:
        result = lsm_price(
            s0=100.0, k=100.0, r=0.05, sigma=0.2, T=1.0,
            n_steps=20, n_paths=5_000, is_call=True, seed=42,
        )
        assert result["price"] > 0

    def test_result_has_keys(self) -> None:
        result = lsm_price(
            s0=100.0, k=100.0, r=0.05, sigma=0.2, T=1.0,
            n_steps=10, n_paths=2_000, seed=42,
        )
        assert "price" in result
        assert "std_err" in result
        assert "optimal_exercise_times" in result


# -----------------------------------------------------------------------
# Lookback Monte Carlo
# -----------------------------------------------------------------------

class TestLookbackMC:
    """Tests for lookback_mc pricing."""

    def test_call_price_positive(self) -> None:
        spec = LookbackOptionSpec(
            s0=100.0, r=0.05, sigma=0.2, T=1.0,
            is_call=True, n_steps=50,
        )
        price = lookback_mc(spec, n_paths=10_000, seed=42)
        assert price > 0

    def test_put_price_positive(self) -> None:
        spec = LookbackOptionSpec(
            s0=100.0, r=0.05, sigma=0.2, T=1.0,
            is_call=False, n_steps=50,
        )
        price = lookback_mc(spec, n_paths=10_000, seed=42)
        assert price > 0


# -----------------------------------------------------------------------
# Cliquet Monte Carlo
# -----------------------------------------------------------------------

class TestCliquetMC:
    """Tests for cliquet_mc pricing."""

    def test_price_positive(self) -> None:
        price = cliquet_mc(
            s0=100.0, r=0.05, sigma=0.2, T=1.0,
            n_periods=4, cap=0.05, floor=-0.05,
            n_paths=10_000, seed=42,
        )
        assert price > 0

    def test_deterministic_with_seed(self) -> None:
        args = dict(s0=100.0, r=0.05, sigma=0.2, T=1.0,
                    n_periods=4, n_paths=5_000, seed=99)
        p1 = cliquet_mc(**args)
        p2 = cliquet_mc(**args)
        assert p1 == p2


# -----------------------------------------------------------------------
# Autocallable Monte Carlo
# -----------------------------------------------------------------------

class TestAutocallableMC:
    """Tests for autocallable_mc pricing."""

    def test_price_positive(self) -> None:
        spec = AutocallableSpec(
            s0=100.0, k=100.0, barrier=110.0, coupon=0.05,
            observation_dates=[0.25, 0.5, 0.75, 1.0],
            r=0.05, sigma=0.2, T=1.0,
        )
        result = autocallable_mc(spec, n_paths=10_000, seed=42)
        assert result["price"] > 0

    def test_result_keys(self) -> None:
        spec = AutocallableSpec(
            s0=100.0, k=100.0, barrier=120.0, coupon=0.03,
            observation_dates=[0.5, 1.0],
            r=0.05, sigma=0.2, T=1.0,
        )
        result = autocallable_mc(spec, n_paths=5_000, seed=42)
        assert "price" in result
        assert "std_err" in result
        assert "autocall_prob" in result


# -----------------------------------------------------------------------
# Quantum resource estimate
# -----------------------------------------------------------------------

class TestResourceEstimateChakrabarti:
    """Tests for resource_estimate_chakrabarti."""

    def test_returns_required_keys(self) -> None:
        res = resource_estimate_chakrabarti(n_qubits=8, n_timesteps=10)
        assert "T_count" in res
        assert "T_depth" in res
        assert "logical_qubits" in res

    def test_values_positive(self) -> None:
        res = resource_estimate_chakrabarti(n_qubits=4, n_timesteps=5)
        assert res["T_count"] > 0
        assert res["T_depth"] > 0
        assert res["logical_qubits"] > 0

    def test_scaling_with_qubits(self) -> None:
        small = resource_estimate_chakrabarti(n_qubits=4, n_timesteps=10)
        large = resource_estimate_chakrabarti(n_qubits=8, n_timesteps=10)
        assert large["T_count"] > small["T_count"]
        assert large["logical_qubits"] > small["logical_qubits"]
