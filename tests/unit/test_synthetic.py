"""Tests for synthetic data generators."""

from __future__ import annotations

import numpy as np
import pytest

from qufin.data.synthetic import gbm_paths, heston_paths, merton_jump_paths


class TestGBM:
    def test_shape(self) -> None:
        paths = gbm_paths(s0=100, mu=0.05, sigma=0.2, T=1.0, n_steps=252, n_paths=100, seed=42)
        assert paths.shape == (100, 253)

    def test_initial_price(self) -> None:
        paths = gbm_paths(s0=100, mu=0.05, sigma=0.2, T=1.0, n_steps=10, n_paths=50, seed=42)
        np.testing.assert_array_equal(paths[:, 0], 100.0)

    def test_positive_prices(self) -> None:
        paths = gbm_paths(s0=100, mu=0.05, sigma=0.2, T=1.0, n_steps=252, n_paths=1000, seed=42)
        assert np.all(paths > 0)

    def test_reproducibility(self) -> None:
        p1 = gbm_paths(s0=100, mu=0.05, sigma=0.2, T=1.0, n_steps=10, n_paths=5, seed=42)
        p2 = gbm_paths(s0=100, mu=0.05, sigma=0.2, T=1.0, n_steps=10, n_paths=5, seed=42)
        np.testing.assert_array_equal(p1, p2)


class TestHeston:
    def test_shape(self) -> None:
        prices, variances = heston_paths(
            s0=100, v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7,
            mu=0.05, T=1.0, n_steps=252, n_paths=100, seed=42,
        )
        assert prices.shape == (100, 253)
        assert variances.shape == (100, 253)

    def test_positive_prices(self) -> None:
        prices, _ = heston_paths(
            s0=100, v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7,
            mu=0.05, T=1.0, n_steps=252, n_paths=500, seed=42,
        )
        assert np.all(prices > 0)

    def test_non_negative_variance(self) -> None:
        _, variances = heston_paths(
            s0=100, v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7,
            mu=0.05, T=1.0, n_steps=252, n_paths=500, seed=42,
        )
        assert np.all(variances >= 0)


class TestMertonJump:
    def test_shape(self) -> None:
        paths = merton_jump_paths(
            s0=100, mu=0.05, sigma=0.2, lam=1.0, jump_mean=-0.05, jump_std=0.1,
            T=1.0, n_steps=252, n_paths=100, seed=42,
        )
        assert paths.shape == (100, 253)

    def test_positive_prices(self) -> None:
        paths = merton_jump_paths(
            s0=100, mu=0.05, sigma=0.2, lam=1.0, jump_mean=-0.05, jump_std=0.1,
            T=1.0, n_steps=252, n_paths=500, seed=42,
        )
        assert np.all(paths > 0)
