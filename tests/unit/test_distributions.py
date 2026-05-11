"""Unit tests for distribution loading."""

from __future__ import annotations

import numpy as np

from qufin.options.distributions import (
    DistributionSpec,
    log_normal_distribution,
    normal_distribution,
    uniform_distribution,
)


class TestLogNormalDistribution:
    def test_basic_properties(self) -> None:
        dist = log_normal_distribution(n_qubits=3, s0=100, mu=0.05, sigma=0.2, T=1.0)
        assert dist.n_qubits == 3
        assert dist.n_states == 8
        assert dist.probabilities.shape == (8,)
        assert dist.values.shape == (8,)
        assert abs(dist.probabilities.sum() - 1.0) < 1e-10

    def test_positive_prices(self) -> None:
        dist = log_normal_distribution(n_qubits=4, s0=100, sigma=0.3)
        assert np.all(dist.values > 0)

    def test_probabilities_nonneg(self) -> None:
        dist = log_normal_distribution(n_qubits=3)
        assert np.all(dist.probabilities >= 0)

    def test_amplitudes_normalized(self) -> None:
        dist = log_normal_distribution(n_qubits=3)
        amps = dist.amplitudes()
        assert abs(np.sum(amps**2) - 1.0) < 1e-10

    def test_different_params(self) -> None:
        d1 = log_normal_distribution(n_qubits=3, sigma=0.1)
        d2 = log_normal_distribution(n_qubits=3, sigma=0.5)
        # Higher vol -> wider distribution
        assert d2.high - d2.low > d1.high - d1.low


class TestNormalDistribution:
    def test_basic_properties(self) -> None:
        dist = normal_distribution(n_qubits=3, mean=0, std=1)
        assert dist.n_states == 8
        assert abs(dist.probabilities.sum() - 1.0) < 1e-10

    def test_symmetric(self) -> None:
        dist = normal_distribution(n_qubits=4, mean=0, std=1)
        probs = dist.probabilities
        # Should be approximately symmetric
        n = len(probs)
        for i in range(n // 2):
            assert abs(probs[i] - probs[n - 1 - i]) < 0.02

    def test_peak_at_center(self) -> None:
        dist = normal_distribution(n_qubits=4, mean=0, std=1)
        mid = len(dist.probabilities) // 2
        # Peak should be near the center
        peak_idx = np.argmax(dist.probabilities)
        assert abs(peak_idx - mid) <= 1


class TestUniformDistribution:
    def test_uniform(self) -> None:
        dist = uniform_distribution(n_qubits=3, low=0, high=1)
        assert dist.n_states == 8
        expected = 1.0 / 8
        np.testing.assert_allclose(dist.probabilities, expected, atol=1e-10)

    def test_bounds(self) -> None:
        dist = uniform_distribution(n_qubits=2, low=5, high=10)
        assert dist.low == 5
        assert dist.high == 10
        assert dist.values[0] == 5
        assert dist.values[-1] == 10


class TestDistributionSpec:
    def test_amplitudes(self) -> None:
        probs = np.array([0.25, 0.25, 0.25, 0.25])
        dist = DistributionSpec(
            n_qubits=2, low=0, high=1,
            probabilities=probs, values=np.linspace(0, 1, 4),
        )
        amps = dist.amplitudes()
        np.testing.assert_allclose(amps, 0.5 * np.ones(4), atol=1e-10)
