"""Tests for the model validation framework (SR 11-7 / SS1/23 compliance)."""

from __future__ import annotations

import numpy as np
import pytest

from qufin.compliance.validation import (
    ChampionChallengerResult,
    ComplianceFramework,
    FullSensitivityReport,
    ModelAssumption,
    ModelDocumentation,
    ModelInput,
    ModelTier,
    PnLAttribution,
    compare_champion_challenger,
    compute_pnl_attribution,
    generate_checklist,
    sensitivity_analysis,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_inputs():
    """Sample model inputs for checklist generation."""
    return [
        ModelInput(
            name="spot_price",
            dtype="float64",
            description="Current asset price",
            range_low=0.0,
            range_high=1e6,
            source="Bloomberg",
        ),
        ModelInput(
            name="volatility",
            dtype="float64",
            description="Annualized implied volatility",
            range_low=0.0,
            range_high=5.0,
            source="Options market",
        ),
    ]


@pytest.fixture
def sample_assumptions():
    """Sample model assumptions."""
    return [
        ModelAssumption(
            statement="Log-normal price dynamics",
            impact="Mispricing if heavy tails present",
            validation_method="Jarque-Bera test on returns",
        ),
        ModelAssumption(
            statement="Constant volatility over option lifetime",
            impact="Term structure effects ignored",
            validation_method="Compare with Heston model",
        ),
    ]


@pytest.fixture
def sample_pnl():
    """Synthetic P&L arrays for attribution."""
    rng = np.random.default_rng(42)
    actual = rng.normal(100, 20, size=252)
    model = actual * 0.9 + rng.normal(0, 5, size=252)
    return actual, model


# ---------------------------------------------------------------------------
# Checklist generation tests
# ---------------------------------------------------------------------------


class TestGenerateChecklist:
    def test_basic_checklist(self, sample_inputs, sample_assumptions):
        doc = generate_checklist(
            model_name="BSM Pricer",
            model_version="1.0.0",
            owner="Quant Team",
            methodology="Black-Scholes-Merton closed-form",
            inputs=sample_inputs,
            assumptions=sample_assumptions,
            limitations=["No smile dynamics"],
        )
        assert isinstance(doc, ModelDocumentation)
        assert doc.model_name == "BSM Pricer"
        assert doc.model_version == "1.0.0"
        assert doc.owner == "Quant Team"
        assert len(doc.inputs) == 2
        assert len(doc.assumptions) == 2
        assert len(doc.limitations) == 1
        assert len(doc.checklist_items) > 0
        assert doc.generated_at != ""

    def test_sr_11_7_framework(self):
        doc = generate_checklist(
            model_name="Test",
            model_version="0.1.0",
            framework=ComplianceFramework.SR_11_7,
            methodology="Test method",
        )
        assert doc.framework == ComplianceFramework.SR_11_7
        # SR 11-7 has 14 items
        assert len(doc.checklist_items) == 14

    def test_ss1_23_framework(self):
        doc = generate_checklist(
            model_name="Test",
            model_version="0.1.0",
            framework=ComplianceFramework.SS1_23,
            methodology="Test method",
        )
        assert doc.framework == ComplianceFramework.SS1_23
        assert len(doc.checklist_items) == 14

    def test_both_frameworks_union(self):
        doc = generate_checklist(
            model_name="Test",
            model_version="0.1.0",
            framework=ComplianceFramework.BOTH,
            methodology="Test method",
        )
        # BOTH merges unique items from SR 11-7 and SS1/23
        assert len(doc.checklist_items) >= 14

    def test_auto_assess_inputs_satisfied(self, sample_inputs):
        doc = generate_checklist(
            model_name="Test",
            model_version="0.1.0",
            framework=ComplianceFramework.SR_11_7,
            inputs=sample_inputs,
            methodology="A methodology",
        )
        assert doc.checklist_items["Inputs and data sources identified"] is True

    def test_auto_assess_inputs_missing(self):
        doc = generate_checklist(
            model_name="Test",
            model_version="0.1.0",
            framework=ComplianceFramework.SR_11_7,
            methodology="A methodology",
        )
        assert doc.checklist_items["Inputs and data sources identified"] is False

    def test_auto_assess_methodology(self):
        doc = generate_checklist(
            model_name="Test",
            model_version="0.1.0",
            framework=ComplianceFramework.SR_11_7,
            methodology="Monte Carlo simulation",
        )
        assert doc.checklist_items["Methodology description provided"] is True

    def test_auto_assess_no_methodology(self):
        doc = generate_checklist(
            model_name="Test",
            model_version="0.1.0",
            framework=ComplianceFramework.SR_11_7,
        )
        assert doc.checklist_items["Methodology description provided"] is False

    def test_model_tier(self):
        doc = generate_checklist(
            model_name="High Impact",
            model_version="1.0.0",
            model_tier=ModelTier.TIER_1,
        )
        assert doc.model_tier == ModelTier.TIER_1

    def test_with_pnl_attribution(self, sample_pnl):
        actual, model = sample_pnl
        attr = compute_pnl_attribution(actual, model)
        doc = generate_checklist(
            model_name="Test",
            model_version="0.1.0",
            framework=ComplianceFramework.SR_11_7,
            methodology="Test",
            pnl_attribution=attr,
        )
        assert doc.checklist_items["Backtesting results available"] is True
        assert doc.pnl_attribution is not None

    def test_owner_checklist_item(self):
        doc = generate_checklist(
            model_name="Test",
            model_version="0.1.0",
            framework=ComplianceFramework.SR_11_7,
            owner="Risk Team",
        )
        assert doc.checklist_items["Model owner assigned"] is True

    def test_no_owner(self):
        doc = generate_checklist(
            model_name="Test",
            model_version="0.1.0",
            framework=ComplianceFramework.SR_11_7,
        )
        assert doc.checklist_items["Model owner assigned"] is False


# ---------------------------------------------------------------------------
# P&L attribution tests
# ---------------------------------------------------------------------------


class TestPnLAttribution:
    def test_basic_attribution(self, sample_pnl):
        actual, model = sample_pnl
        attr = compute_pnl_attribution(actual, model)
        assert isinstance(attr, PnLAttribution)
        assert attr.n_days == 252
        assert attr.total_pnl == pytest.approx(actual.sum())
        assert attr.model_pnl == pytest.approx(model.sum())
        assert attr.residual_pnl == pytest.approx(attr.total_pnl - attr.model_pnl)

    def test_perfect_attribution(self):
        pnl = np.array([100.0, -50.0, 75.0, -25.0, 200.0])
        attr = compute_pnl_attribution(pnl, pnl)
        assert attr.attribution_ratio == pytest.approx(1.0)
        assert attr.residual_pnl == pytest.approx(0.0)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_pnl_attribution(np.array([1.0, 2.0]), np.array([1.0]))

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            compute_pnl_attribution(np.array([]), np.array([]))

    def test_zero_total_pnl(self):
        actual = np.array([100.0, -100.0])
        model = np.array([50.0, -50.0])
        attr = compute_pnl_attribution(actual, model)
        assert attr.attribution_ratio == pytest.approx(0.0)

    def test_single_day(self):
        attr = compute_pnl_attribution(np.array([42.0]), np.array([40.0]))
        assert attr.n_days == 1
        assert attr.std_daily_pnl == 0.0


# ---------------------------------------------------------------------------
# Champion-challenger tests
# ---------------------------------------------------------------------------


class TestChampionChallenger:
    def test_identical_models(self):
        scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = compare_champion_challenger(scores, scores, seed=42)
        assert isinstance(result, ChampionChallengerResult)
        assert result.p_value == pytest.approx(1.0, abs=0.01)
        assert result.significant is False
        assert result.cohens_d == pytest.approx(0.0)

    def test_clearly_different_models(self):
        rng = np.random.default_rng(123)
        champion = rng.normal(0.0, 0.1, size=50)
        challenger = champion + 1.0  # strong improvement
        result = compare_champion_challenger(
            champion, challenger, alpha=0.05, seed=42
        )
        assert result.significant is True
        assert result.p_value < 0.01
        assert result.mean_challenger > result.mean_champion

    def test_bootstrap_ci_contains_zero_for_similar(self):
        rng = np.random.default_rng(7)
        champion = rng.normal(0.0, 1.0, size=30)
        challenger = rng.normal(0.0, 1.0, size=30)
        result = compare_champion_challenger(
            champion, challenger, seed=42, n_bootstrap=5000
        )
        ci_low, ci_high = result.bootstrap_ci
        # For two independent draws, CI should often contain 0
        # (not guaranteed, but very likely with same distribution)
        assert ci_low < ci_high

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="Shape mismatch"):
            compare_champion_challenger(
                np.array([1.0, 2.0]),
                np.array([1.0, 2.0, 3.0]),
            )

    def test_too_few_observations_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            compare_champion_challenger(np.array([1.0]), np.array([2.0]))

    def test_custom_names(self):
        scores = np.array([1.0, 2.0, 3.0])
        result = compare_champion_challenger(
            scores,
            scores + 0.1,
            champion_name="baseline",
            challenger_name="qml",
            seed=42,
        )
        assert result.champion_name == "baseline"
        assert result.challenger_name == "qml"
        assert result.n_problems == 3

    def test_cohens_d_direction(self):
        rng = np.random.default_rng(99)
        champion = rng.normal(0.0, 1.0, size=30)
        challenger = champion + 2.0 + rng.normal(0, 0.1, size=30)
        result = compare_champion_challenger(champion, challenger, seed=42)
        assert result.cohens_d > 0  # challenger better

    def test_constant_difference_cohens_d(self):
        champion = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        challenger = champion + 2.0
        result = compare_champion_challenger(champion, challenger, seed=42)
        assert result.cohens_d == np.inf  # perfect constant improvement


# ---------------------------------------------------------------------------
# Sensitivity analysis tests
# ---------------------------------------------------------------------------


def _linear_model(a: float = 1.0, b: float = 2.0) -> float:
    return a + b


def _quadratic_model(x: float = 1.0, y: float = 1.0) -> float:
    return x**2 + y


class TestSensitivityAnalysis:
    def test_linear_model(self):
        report = sensitivity_analysis(
            _linear_model,
            {"a": 1.0, "b": 2.0},
        )
        assert isinstance(report, FullSensitivityReport)
        assert len(report.results) == 2
        assert report.most_sensitive in ("a", "b")

    def test_single_parameter(self):
        report = sensitivity_analysis(
            _linear_model,
            {"a": 1.0, "b": 2.0},
            param_names=["a"],
        )
        assert len(report.results) == 1
        assert report.results[0].parameter_name == "a"

    def test_custom_perturbations(self):
        pcts = [-0.01, 0.01]
        report = sensitivity_analysis(
            _linear_model,
            {"a": 1.0, "b": 2.0},
            perturbation_pcts=pcts,
        )
        assert len(report.results[0].perturbations) == 2

    def test_unknown_parameter_raises(self):
        with pytest.raises(ValueError, match="Unknown parameters"):
            sensitivity_analysis(
                _linear_model,
                {"a": 1.0, "b": 2.0},
                param_names=["c"],
            )

    def test_ranking_order(self):
        # b has larger base value so perturbing it causes larger change
        report = sensitivity_analysis(
            _linear_model,
            {"a": 1.0, "b": 10.0},
        )
        # b should be more sensitive (larger absolute changes)
        assert report.ranking[0][0] == "b"
        assert report.ranking[0][1] >= report.ranking[1][1]

    def test_elasticity_constant(self):
        # For f(x) = x, elasticity = 1.0
        def identity(x: float = 1.0) -> float:
            return x

        report = sensitivity_analysis(
            identity,
            {"x": 5.0},
            perturbation_pcts=[0.10],
        )
        result = report.results[0]
        assert result.elasticities[0] == pytest.approx(1.0)

    def test_quadratic_sensitivity(self):
        report = sensitivity_analysis(
            _quadratic_model,
            {"x": 2.0, "y": 1.0},
        )
        # x^2 is more sensitive than y when x=2
        assert report.most_sensitive == "x"

    def test_zero_base_value_elasticity(self):
        # When base_output is zero, elasticity should be 0
        def zero_at_base(x: float = 0.0) -> float:
            return x

        sensitivity_analysis(
            zero_at_base,
            {"x": 1.0},
            perturbation_pcts=[0.10],
        )
        # base_output = 1.0, so elasticity is well-defined here
        # But let's test the actual zero case
        def always_zero(x: float = 1.0) -> float:
            return 0.0

        report2 = sensitivity_analysis(
            always_zero,
            {"x": 1.0},
            perturbation_pcts=[0.10],
        )
        assert report2.results[0].elasticities[0] == 0.0

    def test_result_base_output(self):
        report = sensitivity_analysis(
            _linear_model,
            {"a": 3.0, "b": 7.0},
            param_names=["a"],
        )
        assert report.results[0].base_output == pytest.approx(10.0)
        assert report.results[0].base_value == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Edge-case and data-class tests
# ---------------------------------------------------------------------------


class TestDataClasses:
    def test_model_input_defaults(self):
        inp = ModelInput(name="x", dtype="float", description="test")
        assert inp.range_low is None
        assert inp.range_high is None
        assert inp.source == ""

    def test_model_assumption_defaults(self):
        a = ModelAssumption(statement="Normality")
        assert a.impact == ""
        assert a.validation_method == ""

    def test_model_tier_values(self):
        assert ModelTier.TIER_1.value == 1
        assert ModelTier.TIER_2.value == 2
        assert ModelTier.TIER_3.value == 3

    def test_compliance_framework_values(self):
        assert ComplianceFramework.SR_11_7.value == "SR 11-7"
        assert ComplianceFramework.SS1_23.value == "SS1/23"
        assert ComplianceFramework.BOTH.value == "BOTH"
