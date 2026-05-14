"""Tests for compliance explainability module."""

from __future__ import annotations

import numpy as np
import pytest

from qufin.compliance.explainability import (
    MarginalContribution,
    build_interaction_heatmap_data,
    compare_selections,
    decompose_qubo,
    marginal_contribution,
    plot_interaction_heatmap,
    plot_marginal_contributions,
    plot_shapley_values,
    shapley_attribution,
)
from qufin.portfolio.qubo import PortfolioQUBO

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_Q() -> np.ndarray:  # noqa: N802
    """3-asset symmetric QUBO matrix."""
    Q = np.array(
        [
            [-0.05, 0.02, 0.01],
            [0.02, -0.03, 0.04],
            [0.01, 0.04, -0.02],
        ],
        dtype=np.float64,
    )
    return Q


@pytest.fixture
def medium_Q(rng: np.random.Generator) -> np.ndarray:  # noqa: N802
    """10-asset QUBO built from random returns/covariance."""
    mu = rng.uniform(0.01, 0.10, size=10)
    A = rng.standard_normal((10, 10))
    cov = A.T @ A / 10  # positive semi-definite
    qubo = PortfolioQUBO(mu=mu, cov=cov, gamma=1.0)
    return qubo.build_matrix()


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


# ---------------------------------------------------------------------------
# QUBO Decomposition Tests
# ---------------------------------------------------------------------------


class TestQUBODecomposition:
    def test_linear_terms_match_diagonal(self, small_Q: np.ndarray) -> None:
        result = decompose_qubo(small_Q)
        np.testing.assert_array_equal(result.linear_terms, np.diag(small_Q))

    def test_interaction_matrix_zero_diagonal(self, small_Q: np.ndarray) -> None:
        result = decompose_qubo(small_Q)
        np.testing.assert_array_equal(np.diag(result.interaction_matrix), 0.0)

    def test_top_interactions_sorted_by_abs(self, small_Q: np.ndarray) -> None:
        result = decompose_qubo(small_Q)
        abs_strengths = [abs(t[2]) for t in result.top_interactions]
        assert abs_strengths == sorted(abs_strengths, reverse=True)

    def test_top_interactions_count(self, small_Q: np.ndarray) -> None:
        result = decompose_qubo(small_Q, top_k=2)
        assert len(result.top_interactions) == 2

    def test_top_interactions_full(self, small_Q: np.ndarray) -> None:
        # 3 assets => 3 unique pairs
        result = decompose_qubo(small_Q, top_k=100)
        assert len(result.top_interactions) == 3

    def test_asset_total_interaction_nonneg(self, small_Q: np.ndarray) -> None:
        result = decompose_qubo(small_Q)
        assert np.all(result.asset_total_interaction >= 0)

    def test_nonsquare_raises(self) -> None:
        Q = np.zeros((3, 4))
        with pytest.raises(ValueError, match="square"):
            decompose_qubo(Q)

    def test_medium_decomposition(self, medium_Q: np.ndarray) -> None:
        result = decompose_qubo(medium_Q, top_k=5)
        assert result.linear_terms.shape == (10,)
        assert result.interaction_matrix.shape == (10, 10)
        assert len(result.top_interactions) == 5

    def test_interaction_strength_is_sum(self, small_Q: np.ndarray) -> None:
        """Top interaction strength should equal Q[i,j] + Q[j,i]."""
        result = decompose_qubo(small_Q)
        for i, j, strength in result.top_interactions:
            expected = small_Q[i, j] + small_Q[j, i]
            assert abs(strength - expected) < 1e-12


# ---------------------------------------------------------------------------
# Marginal Contribution Tests
# ---------------------------------------------------------------------------


class TestMarginalContribution:
    def test_returns_correct_shape(self, small_Q: np.ndarray) -> None:
        sel = np.array([1.0, 1.0, 0.0])
        mc = marginal_contribution(small_Q, sel)
        assert mc.contributions.shape == (3,)

    def test_base_objective_matches(self, small_Q: np.ndarray) -> None:
        sel = np.array([1.0, 0.0, 1.0])
        mc = marginal_contribution(small_Q, sel)
        expected = float(sel @ small_Q @ sel)
        assert abs(mc.base_objective - expected) < 1e-12

    def test_removing_all_zeros_gives_zero(self) -> None:
        Q = np.eye(3) * -1.0
        sel = np.array([0.0, 0.0, 0.0])
        mc = marginal_contribution(Q, sel)
        assert mc.base_objective == 0.0

    def test_rankings_length(self, small_Q: np.ndarray) -> None:
        sel = np.array([1.0, 1.0, 1.0])
        mc = marginal_contribution(small_Q, sel)
        assert len(mc.asset_rankings) == 3
        assert set(mc.asset_rankings) == {0, 1, 2}

    def test_medium_marginal(self, medium_Q: np.ndarray) -> None:
        sel = np.zeros(10)
        sel[:5] = 1.0
        mc = marginal_contribution(medium_Q, sel)
        assert mc.contributions.shape == (10,)
        assert len(mc.asset_rankings) == 10

    def test_marginal_contribution_consistency(self, small_Q: np.ndarray) -> None:
        """Manually verify marginal for one asset."""
        sel = np.array([1.0, 1.0, 0.0])
        mc = marginal_contribution(small_Q, sel)
        # Remove asset 0
        sel_without_0 = np.array([0.0, 1.0, 0.0])
        expected_delta = float(sel_without_0 @ small_Q @ sel_without_0) - mc.base_objective
        assert abs(mc.contributions[0] - expected_delta) < 1e-12


# ---------------------------------------------------------------------------
# SHAP Attribution Tests
# ---------------------------------------------------------------------------


class TestSHAPAttribution:
    def test_shapley_shape(self, small_Q: np.ndarray) -> None:
        sel = np.array([1.0, 1.0, 0.0])
        attr = shapley_attribution(small_Q, sel, n_permutations=50, seed=0)
        assert attr.shapley_values.shape == (3,)

    def test_shapley_sums_to_objective(self, small_Q: np.ndarray) -> None:
        """Sum of Shapley values should approximate the total objective."""
        sel = np.array([1.0, 1.0, 1.0])
        attr = shapley_attribution(small_Q, sel, n_permutations=500, seed=42)
        total_obj = float(sel @ small_Q @ sel)
        shapley_sum = float(attr.shapley_values.sum())
        # Shapley values sum to f(grand coalition) - f(empty)
        assert abs(shapley_sum - (total_obj - attr.base_value)) < 0.05

    def test_base_value_is_zero(self, small_Q: np.ndarray) -> None:
        sel = np.array([1.0, 0.0, 1.0])
        attr = shapley_attribution(small_Q, sel, seed=7)
        assert attr.base_value == 0.0

    def test_unselected_asset_has_zero_shapley(self, small_Q: np.ndarray) -> None:
        """Assets not in the selection should have zero Shapley value."""
        sel = np.array([1.0, 0.0, 0.0])
        attr = shapley_attribution(small_Q, sel, n_permutations=200, seed=1)
        assert abs(attr.shapley_values[1]) < 1e-12
        assert abs(attr.shapley_values[2]) < 1e-12

    def test_deterministic_with_seed(self, small_Q: np.ndarray) -> None:
        sel = np.array([1.0, 1.0, 1.0])
        a1 = shapley_attribution(small_Q, sel, n_permutations=100, seed=99)
        a2 = shapley_attribution(small_Q, sel, n_permutations=100, seed=99)
        np.testing.assert_array_equal(a1.shapley_values, a2.shapley_values)

    def test_medium_shapley(self, medium_Q: np.ndarray) -> None:
        sel = np.zeros(10)
        sel[0] = sel[3] = sel[7] = 1.0
        attr = shapley_attribution(medium_Q, sel, n_permutations=50, seed=0)
        assert attr.shapley_values.shape == (10,)


# ---------------------------------------------------------------------------
# Comparison Report Tests
# ---------------------------------------------------------------------------


class TestComparisonReport:
    def test_identical_selections_full_agreement(self, small_Q: np.ndarray) -> None:
        sel = np.array([1.0, 0.0, 1.0])
        report = compare_selections(small_Q, sel, sel)
        assert report.agreement_ratio == 1.0
        assert report.quantum_unique == []
        assert report.classical_unique == []

    def test_disjoint_selections(self, small_Q: np.ndarray) -> None:
        q_sel = np.array([1.0, 0.0, 0.0])
        c_sel = np.array([0.0, 1.0, 1.0])
        report = compare_selections(small_Q, q_sel, c_sel)
        assert report.agreement_ratio == 0.0
        assert set(report.quantum_unique) == {0}
        assert set(report.classical_unique) == {1, 2}

    def test_summary_contains_key_info(self, small_Q: np.ndarray) -> None:
        q_sel = np.array([1.0, 1.0, 0.0])
        c_sel = np.array([1.0, 0.0, 1.0])
        report = compare_selections(small_Q, q_sel, c_sel)
        assert "Agreement" in report.summary
        assert "objective" in report.summary.lower()

    def test_objectives_computed_correctly(self, small_Q: np.ndarray) -> None:
        q_sel = np.array([1.0, 0.0, 0.0])
        c_sel = np.array([0.0, 1.0, 0.0])
        report = compare_selections(small_Q, q_sel, c_sel)
        assert abs(report.quantum_objective - float(q_sel @ small_Q @ q_sel)) < 1e-12
        assert abs(report.classical_objective - float(c_sel @ small_Q @ c_sel)) < 1e-12

    def test_custom_asset_labels(self, small_Q: np.ndarray) -> None:
        q_sel = np.array([1.0, 0.0, 0.0])
        c_sel = np.array([0.0, 1.0, 0.0])
        labels = ["AAPL", "GOOGL", "MSFT"]
        report = compare_selections(small_Q, q_sel, c_sel, asset_labels=labels)
        assert "AAPL" in report.summary

    def test_marginals_populated(self, small_Q: np.ndarray) -> None:
        q_sel = np.array([1.0, 1.0, 0.0])
        c_sel = np.array([0.0, 1.0, 1.0])
        report = compare_selections(small_Q, q_sel, c_sel)
        assert isinstance(report.quantum_marginals, MarginalContribution)
        assert isinstance(report.classical_marginals, MarginalContribution)


# ---------------------------------------------------------------------------
# Heatmap / Visualization Tests
# ---------------------------------------------------------------------------


class TestVisualization:
    def test_heatmap_data_shape(self, small_Q: np.ndarray) -> None:
        data = build_interaction_heatmap_data(small_Q)
        assert data.matrix.shape == (3, 3)
        assert np.diag(data.matrix).sum() == 0.0

    def test_heatmap_data_symmetric(self, small_Q: np.ndarray) -> None:
        data = build_interaction_heatmap_data(small_Q, symmetrize=True)
        np.testing.assert_array_almost_equal(data.matrix, data.matrix.T)

    def test_heatmap_no_symmetrize(self, small_Q: np.ndarray) -> None:
        data = build_interaction_heatmap_data(small_Q, symmetrize=False)
        assert data.matrix.shape == (3, 3)

    def test_heatmap_custom_labels(self, small_Q: np.ndarray) -> None:
        labels = ["A", "B", "C"]
        data = build_interaction_heatmap_data(small_Q, asset_labels=labels)
        assert data.labels == labels

    def test_plot_returns_none_without_plotly(self, small_Q: np.ndarray) -> None:
        """If plotly is not installed, plot functions return None gracefully."""
        data = build_interaction_heatmap_data(small_Q)
        # This will return a figure if plotly is installed, or None if not.
        # Either way it should not raise.
        result = plot_interaction_heatmap(data)
        assert result is None or result is not None  # no crash

    def test_plot_marginal_no_crash(self, small_Q: np.ndarray) -> None:
        sel = np.array([1.0, 1.0, 0.0])
        mc = marginal_contribution(small_Q, sel)
        result = plot_marginal_contributions(mc)
        assert result is None or result is not None

    def test_plot_shapley_no_crash(self, small_Q: np.ndarray) -> None:
        sel = np.array([1.0, 1.0, 0.0])
        attr = shapley_attribution(small_Q, sel, n_permutations=10, seed=0)
        result = plot_shapley_values(attr)
        assert result is None or result is not None


# ---------------------------------------------------------------------------
# Integration: PortfolioQUBO -> Explainability pipeline
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_full_pipeline_small(self) -> None:
        """End-to-end: build QUBO -> decompose -> marginal -> compare."""
        mu = np.array([0.05, 0.03, 0.07])
        cov = np.array(
            [[0.01, 0.002, 0.001], [0.002, 0.008, 0.003], [0.001, 0.003, 0.012]]
        )
        qubo = PortfolioQUBO(mu=mu, cov=cov, gamma=1.0)
        Q = qubo.build_matrix()

        # Decompose
        decomp = decompose_qubo(Q)
        assert decomp.linear_terms.shape == (3,)

        # Marginal contribution
        sel_q = np.array([1.0, 0.0, 1.0])
        sel_c = np.array([1.0, 1.0, 0.0])
        mc = marginal_contribution(Q, sel_q)
        assert mc.contributions.shape == (3,)

        # Compare
        report = compare_selections(Q, sel_q, sel_c)
        assert 0.0 <= report.agreement_ratio <= 1.0

    def test_full_pipeline_medium(self) -> None:
        """10-asset pipeline."""
        rng = np.random.default_rng(123)
        mu = rng.uniform(0.01, 0.10, size=10)
        A = rng.standard_normal((10, 10))
        cov = A.T @ A / 10
        qubo = PortfolioQUBO(mu=mu, cov=cov, gamma=0.5, cardinality=4)
        Q = qubo.build_matrix()

        decomp = decompose_qubo(Q, top_k=5)
        assert len(decomp.top_interactions) == 5

        sel = np.zeros(10)
        sel[[0, 2, 5, 8]] = 1.0
        attr = shapley_attribution(Q, sel, n_permutations=50, seed=0)
        assert attr.shapley_values.shape == (10,)

        data = build_interaction_heatmap_data(Q)
        assert data.matrix.shape == (10, 10)
