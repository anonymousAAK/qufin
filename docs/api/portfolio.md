# Portfolio API

## Classical Optimizers

### Mean-Variance

::: qufin.portfolio.classical.mean_variance
    options:
      members:
        - mean_variance
        - Objective
        - MVResult

### Black-Litterman

::: qufin.portfolio.classical.black_litterman
    options:
      members:
        - black_litterman
        - BLResult

### Risk Parity

::: qufin.portfolio.classical.risk_parity
    options:
      members:
        - risk_parity
        - RiskParityResult

### HRP

::: qufin.portfolio.classical.hrp
    options:
      members:
        - hrp
        - HRPResult

## QUBO Formulation

::: qufin.portfolio.qubo
    options:
      members:
        - PortfolioQUBO

## Quantum Optimizers

### QAOA

::: qufin.portfolio.optimizers.qaoa
    options:
      members:
        - QAOAPortfolio
        - QAOAConfig
        - QAOAResult

### VQE

::: qufin.portfolio.optimizers.vqe
    options:
      members:
        - VQEPortfolio
        - VQEConfig
        - VQEResult

### Exhaustive

::: qufin.portfolio.optimizers.exhaustive
    options:
      members:
        - exhaustive_solve
        - ExhaustiveResult

## Mixers

::: qufin.portfolio.mixers
    options:
      members:
        - XMixer
        - XYRingMixer
        - XYFullMixer
        - GroverMixer
        - DickeInitialState
        - get_mixer

## Encodings

::: qufin.portfolio.encodings
    options:
      members:
        - one_hot_encoding
        - binary_encoding
        - unary_encoding
        - decode_one_hot
        - decode_binary
        - qubit_cost_table
