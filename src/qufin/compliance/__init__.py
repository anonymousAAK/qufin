"""Regulatory compliance and model validation tooling."""

from __future__ import annotations

from qufin.compliance.audit import (
    AuditEntry,
    AuditStore,
    QueryFilter,
    compute_input_hash,
)
from qufin.compliance.explainability import (
    ComparisonReport,
    InteractionHeatmapData,
    MarginalContribution,
    QUBODecomposition,
    SHAPAttribution,
    build_interaction_heatmap_data,
    compare_selections,
    decompose_qubo,
    marginal_contribution,
    plot_interaction_heatmap,
    plot_marginal_contributions,
    plot_shapley_values,
    shapley_attribution,
)
from qufin.compliance.validation import (
    ChampionChallengerResult,
    FullSensitivityReport,
    ModelDocumentation,
    SensitivityResult,
    compare_champion_challenger,
    compute_pnl_attribution,
    generate_checklist,
    sensitivity_analysis,
)

__all__ = [
    "AuditEntry",
    "AuditStore",
    "ChampionChallengerResult",
    "ComparisonReport",
    "FullSensitivityReport",
    "InteractionHeatmapData",
    "MarginalContribution",
    "ModelDocumentation",
    "QUBODecomposition",
    "QueryFilter",
    "SHAPAttribution",
    "SensitivityResult",
    "build_interaction_heatmap_data",
    "compare_champion_challenger",
    "compare_selections",
    "compute_input_hash",
    "compute_pnl_attribution",
    "decompose_qubo",
    "generate_checklist",
    "marginal_contribution",
    "plot_interaction_heatmap",
    "plot_marginal_contributions",
    "plot_shapley_values",
    "sensitivity_analysis",
    "shapley_attribution",
]
