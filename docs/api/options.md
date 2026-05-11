# Options API

## European Options

::: qufin.options.european
    options:
      members:
        - EuropeanOption

## Asian Options

::: qufin.options.asian
    options:
      members:
        - AsianOptionSpec
        - geometric_asian_closed_form
        - build_asian_estimation_problem

## Barrier Options

::: qufin.options.barrier
    options:
      members:
        - BarrierOptionSpec
        - barrier_closed_form
        - build_barrier_estimation_problem

## Bermudan Options

::: qufin.options.bermudan
    options:
      members:
        - BermudanOptionSpec
        - BermudanResult
        - bermudan_binomial

## Amplitude Estimation

### Estimation Problem

::: qufin.options.amplitude_estimation.estimation_problem
    options:
      members:
        - EstimationProblem

### Canonical QAE

::: qufin.options.amplitude_estimation.canonical
    options:
      members:
        - CanonicalAmplitudeEstimation
        - CanonicalQAEConfig
        - CanonicalQAEResult

### Iterative QAE (IQAE)

::: qufin.options.amplitude_estimation.iqae
    options:
      members:
        - IterativeAmplitudeEstimation
        - IQAEConfig
        - IQAEResult

## Classical Monte Carlo

::: qufin.options.classical.monte_carlo
    options:
      members:
        - european_mc
        - asian_mc
        - barrier_mc

## Heston Model

::: qufin.options.heston
