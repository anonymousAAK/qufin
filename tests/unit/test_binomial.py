"""Tests for binomial tree option pricing."""

from __future__ import annotations

from qufin.options.classical.binomial import crr_tree
from qufin.options.classical.black_scholes import call_price, put_price


class TestBinomialEuropean:
    def test_call_converges_to_bs(self) -> None:
        s, k, r, sigma, T = 100, 100, 0.05, 0.2, 1.0
        bs = call_price(s, k, r, sigma, T)
        tree = crr_tree(s, k, r, sigma, T, n_steps=500, option_type="call")
        assert abs(tree.price - bs) < 0.05

    def test_put_converges_to_bs(self) -> None:
        s, k, r, sigma, T = 100, 105, 0.05, 0.2, 1.0
        bs = put_price(s, k, r, sigma, T)
        tree = crr_tree(s, k, r, sigma, T, n_steps=500, option_type="put")
        assert abs(tree.price - bs) < 0.05

    def test_no_early_exercise_european(self) -> None:
        result = crr_tree(100, 100, 0.05, 0.2, 1.0, exercise="european")
        assert result.early_exercise is False


class TestBinomialAmerican:
    def test_american_put_geq_european(self) -> None:
        s, k, r, sigma, T = 100, 100, 0.05, 0.2, 1.0
        eu = crr_tree(s, k, r, sigma, T, n_steps=200, option_type="put", exercise="european")
        am = crr_tree(s, k, r, sigma, T, n_steps=200, option_type="put", exercise="american")
        assert am.price >= eu.price - 1e-10

    def test_american_call_no_dividend_equals_european(self) -> None:
        # Without dividends, American call = European call
        s, k, r, sigma, T = 100, 100, 0.05, 0.2, 1.0
        eu = crr_tree(s, k, r, sigma, T, n_steps=200, option_type="call", exercise="european")
        am = crr_tree(s, k, r, sigma, T, n_steps=200, option_type="call", exercise="american")
        assert abs(am.price - eu.price) < 0.01

    def test_deep_itm_put_early_exercise(self) -> None:
        # Deep ITM American put should show early exercise
        result = crr_tree(50, 100, 0.05, 0.2, 1.0, n_steps=100,
                         option_type="put", exercise="american")
        assert result.early_exercise is True
