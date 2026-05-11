"""Unit tests for CVaR objective and ascending CVaR."""

from __future__ import annotations

import numpy as np
import pytest

from qufin.risk.cvar import (
    AscendingCVaR,
    CVaRObjective,
    cvar_from_samples,
    portfolio_cvar,
)


class TestCVaRObjective:
    def test_alpha_1_equals_mean(self) -> None:
        """alpha=1 should recover the standard mean."""
        obj = CVaRObjective(alpha=1.0)
        costs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert obj.evaluate(costs) == pytest.approx(3.0)

    def test_alpha_small_focuses_on_best(self) -> None:
        """Small alpha should focus on lowest cost samples."""
        obj = CVaRObjective(alpha=0.2)
        costs = np.array([10.0, 1.0, 5.0, 3.0, 8.0])
        # alpha=0.2 -> top 1 sample (ceil(0.2 * 5) = 1) -> min = 1.0
        assert obj.evaluate(costs) == pytest.approx(1.0)

    def test_alpha_half(self) -> None:
        obj = CVaRObjective(alpha=0.5)
        costs = np.array([1.0, 2.0, 3.0, 4.0])
        # top 2 of 4: mean(1, 2) = 1.5
        assert obj.evaluate(costs) == pytest.approx(1.5)

    def test_with_counts(self) -> None:
        obj = CVaRObjective(alpha=0.5)
        costs = np.array([1.0, 3.0])
        counts = np.array([2, 2])
        # Expanded: [1, 1, 3, 3], alpha=0.5 -> mean(1, 1) = 1.0
        assert obj.evaluate(costs, counts) == pytest.approx(1.0)

    def test_invalid_alpha(self) -> None:
        with pytest.raises(ValueError):
            CVaRObjective(alpha=0.0)
        with pytest.raises(ValueError):
            CVaRObjective(alpha=1.5)

    def test_gradient_weights(self) -> None:
        obj = CVaRObjective(alpha=0.5)
        costs = np.array([1.0, 2.0, 3.0, 4.0])
        weights = obj.gradient_weight(costs)
        # Best 2 of 4 get weight, others get 0
        assert weights[0] > 0  # cost=1
        assert weights[1] > 0  # cost=2
        assert weights[2] == 0  # cost=3
        assert weights[3] == 0  # cost=4

    def test_empty_costs(self) -> None:
        obj = CVaRObjective(alpha=0.5)
        assert obj.evaluate(np.array([])) == 0.0


class TestAscendingCVaR:
    def test_linear_schedule(self) -> None:
        sched = AscendingCVaR(alpha_start=0.1, alpha_end=1.0, n_steps=11, schedule="linear")
        alphas = []
        for _ in range(11):
            alphas.append(sched.get_alpha())
            sched.step()
        assert alphas[0] == pytest.approx(0.1)
        assert alphas[-1] == pytest.approx(1.0)
        # Should be monotonically increasing
        assert all(alphas[i] <= alphas[i + 1] for i in range(len(alphas) - 1))

    def test_cosine_schedule(self) -> None:
        sched = AscendingCVaR(alpha_start=0.1, alpha_end=1.0, n_steps=10, schedule="cosine")
        alpha_first = sched.get_alpha()
        sched._step = 9
        alpha_last = sched.get_alpha()
        assert alpha_first == pytest.approx(0.1)
        assert alpha_last == pytest.approx(1.0)

    def test_exponential_schedule(self) -> None:
        sched = AscendingCVaR(alpha_start=0.1, alpha_end=1.0, n_steps=10, schedule="exponential")
        alpha_first = sched.get_alpha()
        assert alpha_first == pytest.approx(0.1)

    def test_get_objective(self) -> None:
        sched = AscendingCVaR(alpha_start=0.2, alpha_end=0.8, n_steps=5)
        obj = sched.get_objective()
        assert isinstance(obj, CVaRObjective)
        assert obj.alpha == pytest.approx(0.2)

    def test_reset(self) -> None:
        sched = AscendingCVaR(alpha_start=0.1, alpha_end=1.0, n_steps=10)
        sched.step()
        sched.step()
        sched.reset()
        assert sched.get_alpha() == pytest.approx(0.1)


class TestCVaRFromSamples:
    def test_basic(self) -> None:
        costs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        result = cvar_from_samples(costs, alpha=0.1)
        # 10% of 10 samples = 1 sample -> min = 1.0
        assert result == pytest.approx(1.0)


class TestPortfolioCVaR:
    def test_basic(self) -> None:
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.02, (500, 3))
        weights = np.array([0.5, 0.3, 0.2])
        cvar = portfolio_cvar(returns, weights, alpha=0.05)
        assert cvar > 0  # positive loss
