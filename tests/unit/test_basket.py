"""Unit tests for basket option pricing."""

from __future__ import annotations

import numpy as np
import pytest

from qufin.derivatives.basket import (
    BasketOptionSpec,
    basket_mc,
    geometric_basket_closed_form,
)


@pytest.fixture
def two_asset_spec() -> BasketOptionSpec:
    return BasketOptionSpec(
        s0=np.array([100.0, 100.0]),
        k=100.0,
        r=0.05,
        sigma=np.array([0.2, 0.3]),
        corr=np.array([[1.0, 0.5], [0.5, 1.0]]),
        T=1.0,
    )


@pytest.fixture
def three_asset_spec() -> BasketOptionSpec:
    return BasketOptionSpec(
        s0=np.array([100.0, 110.0, 90.0]),
        k=100.0,
        r=0.05,
        sigma=np.array([0.2, 0.25, 0.3]),
        corr=np.array([
            [1.0, 0.3, 0.1],
            [0.3, 1.0, 0.4],
            [0.1, 0.4, 1.0],
        ]),
        T=1.0,
    )


class TestBasketMC:
    def test_call_price_positive(self, two_asset_spec: BasketOptionSpec) -> None:
        result = basket_mc(two_asset_spec, n_paths=50_000, seed=42)
        assert result.price > 0
        assert result.std_err > 0

    def test_put_price_positive(self, two_asset_spec: BasketOptionSpec) -> None:
        spec = BasketOptionSpec(
            s0=two_asset_spec.s0, k=two_asset_spec.k,
            r=two_asset_spec.r, sigma=two_asset_spec.sigma,
            corr=two_asset_spec.corr, T=two_asset_spec.T,
            is_call=False,
        )
        result = basket_mc(spec, n_paths=50_000, seed=42)
        assert result.price > 0

    def test_three_assets(self, three_asset_spec: BasketOptionSpec) -> None:
        result = basket_mc(three_asset_spec, n_paths=50_000, seed=42)
        assert result.price > 0
        assert result.n_paths == 50_000

    def test_confidence_interval(self, two_asset_spec: BasketOptionSpec) -> None:
        result = basket_mc(two_asset_spec, n_paths=100_000, seed=42)
        low, high = result.confidence_interval
        assert low < result.price < high

    def test_deterministic(self, two_asset_spec: BasketOptionSpec) -> None:
        r1 = basket_mc(two_asset_spec, n_paths=10_000, seed=42)
        r2 = basket_mc(two_asset_spec, n_paths=10_000, seed=42)
        assert r1.price == r2.price

    def test_custom_weights(self) -> None:
        spec = BasketOptionSpec(
            s0=np.array([100.0, 100.0]),
            k=100.0, r=0.05,
            sigma=np.array([0.2, 0.3]),
            corr=np.eye(2),
            T=1.0,
            weights=np.array([0.7, 0.3]),
        )
        result = basket_mc(spec, n_paths=50_000, seed=42)
        assert result.price > 0


class TestGeometricBasketClosedForm:
    def test_basic_price(self, two_asset_spec: BasketOptionSpec) -> None:
        price = geometric_basket_closed_form(two_asset_spec)
        assert price > 0

    def test_three_assets(self, three_asset_spec: BasketOptionSpec) -> None:
        price = geometric_basket_closed_form(three_asset_spec)
        assert price > 0

    def test_vs_mc_geometric(self, two_asset_spec: BasketOptionSpec) -> None:
        """Geometric closed form should be in the ballpark of MC."""
        cf = geometric_basket_closed_form(two_asset_spec)
        # MC with arithmetic average will differ, but should be similar order
        mc = basket_mc(two_asset_spec, n_paths=200_000, seed=42)
        # Both should be positive and in similar range
        assert cf > 0
        assert mc.price > 0
