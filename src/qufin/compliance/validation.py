"""Model validation framework aligned with SR 11-7 (Fed) and SS1/23 (PRA).

Provides three core capabilities for quantitative model governance:

1. **Checklist generation** -- auto-generates model documentation covering
   inputs, methodology, assumptions, and limitations per SR 11-7 / SS1/23.
2. **Champion-challenger** -- compares a quantum (challenger) model against a
   classical (champion) model on the same problem set using paired t-tests,
   bootstrap confidence intervals, and effect-size metrics.
3. **Sensitivity analysis** -- systematically perturbs key parameters and
   measures output change to assess model stability.

References
----------
Federal Reserve SR 11-7, "Guidance on Model Risk Management" (2011).
PRA SS1/23, "Model risk management principles for banks" (2023).
"""

from __future__ import annotations

import copy
import itertools
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

import numpy as np
from numpy.typing import NDArray
from scipy import stats

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class ModelTier(Enum):
    """Model materiality tier per SR 11-7 classification.

    Tier 1 models have highest potential impact and require full independent
    validation; Tier 3 models require only periodic review.
    """

    TIER_1 = 1
    TIER_2 = 2
    TIER_3 = 3


class ComplianceFramework(Enum):
    """Regulatory framework for checklist generation."""

    SR_11_7 = "SR 11-7"
    SS1_23 = "SS1/23"
    BOTH = "BOTH"


# ---------------------------------------------------------------------------
# Data classes -- checklist and documentation
# ---------------------------------------------------------------------------


@dataclass
class ModelInput:
    """Description of a single model input.

    Attributes
    ----------
    name : str
        Input variable name.
    dtype : str
        Data type (e.g. ``"float64"``, ``"NDArray"``).
    description : str
        Free-text description of the input.
    range_low : float | None
        Lower bound of acceptable values, if applicable.
    range_high : float | None
        Upper bound of acceptable values, if applicable.
    source : str
        Data source (e.g. ``"Bloomberg"``, ``"synthetic"``).
    """

    name: str
    dtype: str
    description: str
    range_low: float | None = None
    range_high: float | None = None
    source: str = ""


@dataclass
class ModelAssumption:
    """A documented model assumption.

    Attributes
    ----------
    statement : str
        The assumption in plain English.
    impact : str
        What happens if the assumption is violated.
    validation_method : str
        How the assumption can be tested.
    """

    statement: str
    impact: str = ""
    validation_method: str = ""


@dataclass
class ModelDocumentation:
    """Auto-generated model documentation for SR 11-7 / SS1/23.

    This is the core artefact produced by :func:`generate_checklist`.

    Attributes
    ----------
    model_name : str
        Canonical model name.
    model_version : str
        Semantic version string.
    model_tier : ModelTier
        Materiality tier.
    framework : ComplianceFramework
        Governing regulatory framework.
    owner : str
        Model owner / responsible party.
    methodology : str
        Description of the modelling methodology.
    inputs : list[ModelInput]
        Documented inputs.
    assumptions : list[ModelAssumption]
        Key assumptions.
    limitations : list[str]
        Known limitations.
    generated_at : str
        ISO-8601 timestamp of generation.
    checklist_items : dict[str, bool]
        Checklist items and their completion status.
    pnl_attribution : PnLAttribution | None
        Backtesting P&L attribution report, if available.
    """

    model_name: str
    model_version: str
    model_tier: ModelTier
    framework: ComplianceFramework
    owner: str
    methodology: str
    inputs: list[ModelInput] = field(default_factory=list)
    assumptions: list[ModelAssumption] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    generated_at: str = ""
    checklist_items: dict[str, bool] = field(default_factory=dict)
    pnl_attribution: PnLAttribution | None = None


# ---------------------------------------------------------------------------
# Data classes -- P&L attribution
# ---------------------------------------------------------------------------


@dataclass
class PnLAttribution:
    """Backtesting P&L attribution report.

    Attributes
    ----------
    total_pnl : float
        Total profit/loss over the backtest period.
    model_pnl : float
        P&L explained by the model.
    residual_pnl : float
        Unexplained P&L (total - model).
    attribution_ratio : float
        Fraction of total P&L explained by the model.
    daily_pnl : NDArray
        Daily P&L time series.
    daily_model_pnl : NDArray
        Daily model-explained P&L.
    n_days : int
        Number of trading days.
    mean_daily_pnl : float
        Mean daily P&L.
    std_daily_pnl : float
        Standard deviation of daily P&L.
    """

    total_pnl: float
    model_pnl: float
    residual_pnl: float
    attribution_ratio: float
    daily_pnl: NDArray[np.float64]
    daily_model_pnl: NDArray[np.float64]
    n_days: int
    mean_daily_pnl: float
    std_daily_pnl: float


# ---------------------------------------------------------------------------
# Data classes -- champion-challenger
# ---------------------------------------------------------------------------


@dataclass
class ChampionChallengerResult:
    """Result of a champion-challenger comparison.

    Attributes
    ----------
    champion_name : str
        Name of the champion (baseline) model.
    challenger_name : str
        Name of the challenger model.
    champion_scores : NDArray
        Per-problem scores from the champion.
    challenger_scores : NDArray
        Per-problem scores from the challenger.
    n_problems : int
        Number of problems compared.
    mean_champion : float
        Mean score for the champion.
    mean_challenger : float
        Mean score for the challenger.
    t_statistic : float
        Paired t-test statistic.
    p_value : float
        Two-sided p-value from the paired t-test.
    significant : bool
        True if difference is statistically significant at ``alpha``.
    alpha : float
        Significance level used.
    bootstrap_ci : tuple[float, float]
        Bootstrap confidence interval for mean difference.
    cohens_d : float
        Cohen's d effect size for the paired difference.
    """

    champion_name: str
    challenger_name: str
    champion_scores: NDArray[np.float64]
    challenger_scores: NDArray[np.float64]
    n_problems: int
    mean_champion: float
    mean_challenger: float
    t_statistic: float
    p_value: float
    significant: bool
    alpha: float
    bootstrap_ci: tuple[float, float]
    cohens_d: float


# ---------------------------------------------------------------------------
# Data classes -- sensitivity analysis
# ---------------------------------------------------------------------------


@dataclass
class SensitivityResult:
    """Result of a sensitivity analysis for one parameter.

    Attributes
    ----------
    parameter_name : str
        Name of the perturbed parameter.
    base_value : float
        Original parameter value.
    perturbations : NDArray
        Array of perturbation values used.
    outputs : NDArray
        Model output at each perturbation.
    base_output : float
        Model output at the base value.
    elasticities : NDArray
        Point elasticities: ``(dY/Y) / (dX/X)`` at each perturbation.
    max_abs_change : float
        Maximum absolute output change across perturbations.
    mean_abs_change : float
        Mean absolute output change.
    """

    parameter_name: str
    base_value: float
    perturbations: NDArray[np.float64]
    outputs: NDArray[np.float64]
    base_output: float
    elasticities: NDArray[np.float64]
    max_abs_change: float
    mean_abs_change: float


@dataclass
class FullSensitivityReport:
    """Aggregated sensitivity report across multiple parameters.

    Attributes
    ----------
    results : list[SensitivityResult]
        Per-parameter results.
    most_sensitive : str
        Name of the parameter with the highest mean absolute change.
    ranking : list[tuple[str, float]]
        Parameters ranked by mean absolute change (descending).
    """

    results: list[SensitivityResult]
    most_sensitive: str
    ranking: list[tuple[str, float]]


# ---------------------------------------------------------------------------
# Checklist items per framework
# ---------------------------------------------------------------------------

_SR_11_7_ITEMS: list[str] = [
    "Model purpose and scope documented",
    "Inputs and data sources identified",
    "Methodology description provided",
    "Assumptions documented with impact analysis",
    "Limitations acknowledged",
    "Performance metrics defined",
    "Backtesting results available",
    "Outcomes analysis performed",
    "Sensitivity analysis completed",
    "Independent validation performed",
    "Model owner assigned",
    "Version control in place",
    "Change management process defined",
    "Ongoing monitoring plan established",
]

_SS1_23_ITEMS: list[str] = [
    "Model purpose and scope documented",
    "Inputs and data sources identified",
    "Methodology description provided",
    "Assumptions documented with impact analysis",
    "Limitations acknowledged",
    "Model tiering completed",
    "Model inventory entry created",
    "Pre-implementation validation performed",
    "Performance benchmarking conducted",
    "Sensitivity analysis completed",
    "Model risk appetite statement",
    "Board-level reporting mechanism",
    "Periodic review schedule defined",
    "Third-party model assessment (if applicable)",
]


# ---------------------------------------------------------------------------
# Checklist generation
# ---------------------------------------------------------------------------


def generate_checklist(
    *,
    model_name: str,
    model_version: str,
    model_tier: ModelTier = ModelTier.TIER_2,
    framework: ComplianceFramework = ComplianceFramework.BOTH,
    owner: str = "",
    methodology: str = "",
    inputs: list[ModelInput] | None = None,
    assumptions: list[ModelAssumption] | None = None,
    limitations: list[str] | None = None,
    pnl_attribution: PnLAttribution | None = None,
) -> ModelDocumentation:
    """Generate an SR 11-7 / SS1/23 model documentation checklist.

    Automatically assesses which checklist items are satisfied based on
    the supplied documentation artefacts.

    Parameters
    ----------
    model_name : str
        Canonical model name.
    model_version : str
        Semantic version string.
    model_tier : ModelTier
        Materiality tier (default ``TIER_2``).
    framework : ComplianceFramework
        Which regulatory framework(s) to cover.
    owner : str
        Model owner / responsible party.
    methodology : str
        Description of the modelling methodology.
    inputs : list[ModelInput] | None
        Documented model inputs.
    assumptions : list[ModelAssumption] | None
        Model assumptions.
    limitations : list[str] | None
        Known model limitations.
    pnl_attribution : PnLAttribution | None
        Optional P&L attribution from backtesting.

    Returns
    -------
    ModelDocumentation
        Complete documentation object with auto-assessed checklist.
    """
    inputs = inputs or []
    assumptions = assumptions or []
    limitations = limitations or []

    # Merge checklist items from the selected framework(s)
    if framework == ComplianceFramework.SR_11_7:
        items = list(_SR_11_7_ITEMS)
    elif framework == ComplianceFramework.SS1_23:
        items = list(_SS1_23_ITEMS)
    else:
        # BOTH -- union, preserving order
        seen: set[str] = set()
        items = []
        for item in itertools.chain(_SR_11_7_ITEMS, _SS1_23_ITEMS):
            if item not in seen:
                seen.add(item)
                items.append(item)

    # Auto-assess checklist
    checklist: dict[str, bool] = {}
    for item in items:
        checklist[item] = _auto_assess(
            item,
            model_name=model_name,
            methodology=methodology,
            owner=owner,
            inputs=inputs,
            assumptions=assumptions,
            limitations=limitations,
            pnl_attribution=pnl_attribution,
            model_tier=model_tier,
        )

    timestamp = datetime.now(tz=timezone.utc).isoformat()

    return ModelDocumentation(
        model_name=model_name,
        model_version=model_version,
        model_tier=model_tier,
        framework=framework,
        owner=owner,
        methodology=methodology,
        inputs=inputs,
        assumptions=assumptions,
        limitations=limitations,
        generated_at=timestamp,
        checklist_items=checklist,
        pnl_attribution=pnl_attribution,
    )


def _auto_assess(
    item: str,
    *,
    model_name: str,
    methodology: str,
    owner: str,
    inputs: list[ModelInput],
    assumptions: list[ModelAssumption],
    limitations: list[str],
    pnl_attribution: PnLAttribution | None,
    model_tier: ModelTier,
) -> bool:
    """Heuristically assess whether a checklist item is satisfied."""
    lower = item.lower()

    if "purpose" in lower or "scope" in lower:
        return bool(model_name and methodology)
    if "inputs" in lower or "data sources" in lower:
        return len(inputs) > 0
    if "methodology" in lower:
        return bool(methodology)
    if "assumptions" in lower:
        return len(assumptions) > 0
    if "limitations" in lower:
        return len(limitations) > 0
    if "backtesting" in lower:
        return pnl_attribution is not None
    if "owner" in lower:
        return bool(owner)
    if "tiering" in lower:
        return model_tier is not None
    # Items requiring external processes -- default to False
    return "version" in lower


# ---------------------------------------------------------------------------
# P&L attribution
# ---------------------------------------------------------------------------


def compute_pnl_attribution(
    actual_pnl: NDArray[np.float64],
    model_pnl: NDArray[np.float64],
) -> PnLAttribution:
    """Compute backtesting P&L attribution.

    Decomposes total P&L into model-explained and residual components.

    Parameters
    ----------
    actual_pnl : NDArray
        Realised daily P&L series.
    model_pnl : NDArray
        Daily P&L predicted / attributed by the model.

    Returns
    -------
    PnLAttribution
        Attribution report.

    Raises
    ------
    ValueError
        If arrays have different lengths or are empty.
    """
    actual_pnl = np.asarray(actual_pnl, dtype=np.float64)
    model_pnl = np.asarray(model_pnl, dtype=np.float64)

    if actual_pnl.shape != model_pnl.shape:
        msg = (
            f"Shape mismatch: actual_pnl {actual_pnl.shape} "
            f"vs model_pnl {model_pnl.shape}"
        )
        raise ValueError(msg)
    if actual_pnl.size == 0:
        raise ValueError("P&L arrays must not be empty")

    total = float(actual_pnl.sum())
    model_total = float(model_pnl.sum())
    residual = total - model_total
    ratio = model_total / total if total != 0.0 else 0.0

    return PnLAttribution(
        total_pnl=total,
        model_pnl=model_total,
        residual_pnl=residual,
        attribution_ratio=ratio,
        daily_pnl=actual_pnl,
        daily_model_pnl=model_pnl,
        n_days=len(actual_pnl),
        mean_daily_pnl=float(actual_pnl.mean()),
        std_daily_pnl=float(actual_pnl.std(ddof=1)) if len(actual_pnl) > 1 else 0.0,
    )


# ---------------------------------------------------------------------------
# Champion-challenger comparison
# ---------------------------------------------------------------------------


def compare_champion_challenger(
    champion_scores: NDArray[np.float64],
    challenger_scores: NDArray[np.float64],
    *,
    champion_name: str = "classical",
    challenger_name: str = "quantum",
    alpha: float = 0.05,
    n_bootstrap: int = 10_000,
    seed: int | None = None,
) -> ChampionChallengerResult:
    """Compare champion and challenger models on the same problem set.

    Uses a paired t-test for statistical significance and bootstrap
    resampling for a confidence interval on the mean difference
    (challenger - champion).

    Parameters
    ----------
    champion_scores : NDArray
        Scores from the champion model (one per problem).
    challenger_scores : NDArray
        Scores from the challenger model (one per problem).
    champion_name : str
        Label for the champion.
    challenger_name : str
        Label for the challenger.
    alpha : float
        Significance level (default 0.05).
    n_bootstrap : int
        Number of bootstrap resamples (default 10,000).
    seed : int | None
        Random seed for bootstrap reproducibility.

    Returns
    -------
    ChampionChallengerResult
        Full comparison statistics.

    Raises
    ------
    ValueError
        If arrays differ in length, are empty, or have fewer than 2 elements.
    """
    champion_scores = np.asarray(champion_scores, dtype=np.float64)
    challenger_scores = np.asarray(challenger_scores, dtype=np.float64)

    if champion_scores.shape != challenger_scores.shape:
        msg = (
            f"Shape mismatch: champion {champion_scores.shape} "
            f"vs challenger {challenger_scores.shape}"
        )
        raise ValueError(msg)
    n = len(champion_scores)
    if n < 2:
        raise ValueError("Need at least 2 paired observations")

    # Paired t-test (handle zero-variance edge case)
    diffs = challenger_scores - champion_scores
    if np.all(diffs == diffs[0]):
        # Constant differences -- t-test is degenerate
        if diffs[0] == 0.0:
            t_stat, p_val = 0.0, 1.0
        else:
            t_stat, p_val = np.inf if diffs[0] > 0 else -np.inf, 0.0
    else:
        t_stat, p_val = stats.ttest_rel(challenger_scores, champion_scores)
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = diffs[idx].mean()
    ci_low = float(np.percentile(boot_means, 100 * alpha / 2))
    ci_high = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))

    # Cohen's d for paired differences
    d_mean = float(diffs.mean())
    d_std = float(diffs.std(ddof=1)) if n > 1 else 0.0
    if d_std > 0:
        cohens_d = d_mean / d_std
    elif d_mean != 0.0:
        cohens_d = np.inf if d_mean > 0 else -np.inf
    else:
        cohens_d = 0.0

    return ChampionChallengerResult(
        champion_name=champion_name,
        challenger_name=challenger_name,
        champion_scores=champion_scores,
        challenger_scores=challenger_scores,
        n_problems=n,
        mean_champion=float(champion_scores.mean()),
        mean_challenger=float(challenger_scores.mean()),
        t_statistic=float(t_stat),
        p_value=float(p_val),
        significant=float(p_val) < alpha,
        alpha=alpha,
        bootstrap_ci=(ci_low, ci_high),
        cohens_d=cohens_d,
    )


# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------


def sensitivity_analysis(
    model_fn: Callable[..., float],
    base_params: dict[str, float],
    perturbation_pcts: NDArray[np.float64] | list[float] | None = None,
    *,
    param_names: list[str] | None = None,
) -> FullSensitivityReport:
    """Run sensitivity analysis by perturbing each parameter independently.

    For each selected parameter, the function evaluates the model at
    ``base_value * (1 + pct)`` for every percentage in *perturbation_pcts*,
    holding all other parameters at their base values.

    Parameters
    ----------
    model_fn : Callable[..., float]
        Model function that accepts keyword arguments from *base_params*
        and returns a scalar output.
    base_params : dict[str, float]
        Baseline parameter values.
    perturbation_pcts : array-like | None
        Fractional perturbations to apply (e.g. ``[-0.10, -0.05, 0.05, 0.10]``
        for +/-5% and +/-10%).  Defaults to
        ``[-0.20, -0.10, -0.05, 0.05, 0.10, 0.20]``.
    param_names : list[str] | None
        Subset of parameter names to perturb.  Defaults to all keys in
        *base_params*.

    Returns
    -------
    FullSensitivityReport
        Aggregated report with per-parameter results and ranking.

    Raises
    ------
    ValueError
        If *param_names* contains a key not in *base_params*.
    """
    if perturbation_pcts is None:
        perturbation_pcts = [-0.20, -0.10, -0.05, 0.05, 0.10, 0.20]
    pcts = np.asarray(perturbation_pcts, dtype=np.float64)

    if param_names is None:
        param_names = list(base_params.keys())
    else:
        unknown = set(param_names) - set(base_params.keys())
        if unknown:
            raise ValueError(f"Unknown parameters: {unknown}")

    # Compute base output
    base_output = float(model_fn(**base_params))

    results: list[SensitivityResult] = []
    for pname in param_names:
        base_val = base_params[pname]
        outputs = np.empty(len(pcts))
        elasticities = np.empty(len(pcts))

        for j, pct in enumerate(pcts):
            perturbed = copy.copy(base_params)
            perturbed[pname] = base_val * (1.0 + pct)
            outputs[j] = float(model_fn(**perturbed))

            # Elasticity: (dY/Y) / (dX/X)
            dy = outputs[j] - base_output
            if base_output != 0.0 and pct != 0.0:
                elasticities[j] = (dy / base_output) / pct
            else:
                elasticities[j] = 0.0

        abs_changes = np.abs(outputs - base_output)
        results.append(
            SensitivityResult(
                parameter_name=pname,
                base_value=base_val,
                perturbations=pcts,
                outputs=outputs,
                base_output=base_output,
                elasticities=elasticities,
                max_abs_change=float(abs_changes.max()),
                mean_abs_change=float(abs_changes.mean()),
            )
        )

    # Rank by mean absolute change (descending)
    ranking = sorted(
        [(r.parameter_name, r.mean_abs_change) for r in results],
        key=lambda x: x[1],
        reverse=True,
    )
    most_sensitive = ranking[0][0] if ranking else ""

    return FullSensitivityReport(
        results=results,
        most_sensitive=most_sensitive,
        ranking=ranking,
    )
