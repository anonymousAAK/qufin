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

### Multi-Period

::: qufin.portfolio.optimizers.multi_period
    options:
      members:
        - MultiPeriodConfig
        - MultiPeriodResult
        - multi_period_optimize
        - multi_period_backtest
        - compute_turnover

### ADMM

::: qufin.portfolio.optimizers.admm
    options:
      members:
        - ADMMOptimizer
        - ADMMConfig
        - ADMMResult

### Hybrid

::: qufin.portfolio.optimizers.hybrid
    options:
      members:
        - HybridOptimizer
        - HybridConfig
        - HybridResult

### Robust (CVaR QUBO)

::: qufin.portfolio.optimizers.robust
    options:
      members:
        - RobustPortfolioOptimizer
        - EllipsoidalUncertaintySet
        - robust_classical

### Szegedy Quantum Walk

::: qufin.portfolio.optimizers.quantum_walk
    options:
      members:
        - SzegedyWalkOptimizer
        - SzegedyWalkConfig
        - SzegedyWalkResult
        - classical_random_walk

### Factor Models

::: qufin.portfolio.classical.factor_models
    options:
      members:
        - build_factor_model
        - estimate_factor_exposures
        - factor_model_cov
        - factor_expected_returns
        - risk_decomposition
        - FactorModelResult
        - FactorExposureResult

### Sector Rotation

::: qufin.portfolio.sector_rotation
    options:
      members:
        - SectorRotator
        - RegimeDetector
        - backtest_sector_rotation

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
