# Compliance & Audit API

## Audit Trail

::: qufin.compliance.audit
    options:
      members:
        - AuditEntry
        - AuditStore
        - QueryFilter
        - compute_input_hash

## Model Validation

::: qufin.compliance.validation
    options:
      members:
        - generate_checklist
        - compare_champion_challenger
        - sensitivity_analysis
        - compute_pnl_attribution
        - ModelDocumentation
        - ChampionChallengerResult
        - FullSensitivityReport

## Explainability

::: qufin.compliance.explainability
    options:
      members:
        - decompose_qubo
        - marginal_contribution
        - shapley_attribution
        - compare_selections
        - build_interaction_heatmap_data
        - QUBODecomposition
        - MarginalContribution
        - SHAPAttribution
        - ComparisonReport
