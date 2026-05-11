# Risk API

## Value at Risk

::: qufin.risk.classical_var
    options:
      members:
        - VaRResult
        - historical_var
        - parametric_var
        - monte_carlo_var
        - portfolio_var

## Conditional VaR

::: qufin.risk.cvar
    options:
      members:
        - CVaRObjective
        - AscendingCVaR
        - cvar_from_samples
        - portfolio_cvar

## Counterparty Risk

::: qufin.risk.counterparty
    options:
      members:
        - CounterpartyExposure
        - compute_cva
        - compute_ead_sa_ccr
        - portfolio_cva

## Stress Testing

::: qufin.risk.stress
    options:
      members:
        - StressScenario
        - apply_stress
        - stress_test_suite
        - SCENARIO_LIBRARY
