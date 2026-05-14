# Backends API

## Core

::: qufin.backends.base
    options:
      members:
        - Backend
        - CircuitResult

## Mock Backend

::: qufin.backends.mock.MockBackend

## Qiskit Aer Backend

::: qufin.backends.qiskit_backend.QiskitAerBackend

## Noise Models

::: qufin.backends.noise_models
    options:
      members:
        - NoiseProfile
        - NoisyAerBackend
        - build_noise_model
        - sweep_noise
        - IDEAL
        - IBM_EAGLE_R3
        - IBM_HERON_R2
        - NOISY_NEAR_TERM

## Error Mitigation

::: qufin.backends.error_mitigation
    options:
      members:
        - zne_extrapolate
        - calibrate_readout
        - mitigate_readout
        - trex_mitigate
        - MitigationResult
        - pec_mitigate
        - PECConfig
        - cdr_mitigate
        - CDRConfig

## PennyLane Backend

::: qufin.backends.pennylane_backend
    options:
      members:
        - PennyLaneBackend
        - GradientResult

## Amazon Braket Backend

::: qufin.backends.braket_backend
    options:
      members:
        - BraketBackend
        - IonQTarget
        - RigettiTarget
        - CostEstimate
        - estimate_cost
        - analyze_swap_overhead

## Cirq Backend

::: qufin.backends.cirq_backend
    options:
      members:
        - CirqBackend
        - GoogleHardwareConfig
        - NoiseCharacterization

## CUDA-Q Backend

::: qufin.backends.cudaq_backend
    options:
      members:
        - CudaQBackend

## Backend Auto-Selection

::: qufin.backends.auto_select
    options:
      members:
        - CircuitAnalysis
        - BackendRegistry
        - analyze_circuit
        - auto_select_backend
        - get_available_backends

## Finance Transpiler

::: qufin.backends.transpiler
    options:
      members:
        - FinanceTranspiler
        - TranspilationResult
        - qubo_interaction_graph
        - find_commuting_groups
        - initial_layout_from_qubo

## Dynamical Decoupling

::: qufin.backends.dynamical_decoupling
    options:
      members:
        - DDSequence
        - DDConfig
        - insert_dd_sequences
        - xy4_sequence
        - cpmg_sequence
        - uhrig_sequence
        - estimate_t2_extension
        - dd_with_zne

## M3 Measurement Mitigation

::: qufin.backends.m3_mitigation
    options:
      members:
        - M3Config
        - M3Mitigator
        - tensored_calibration
        - iterative_correction

## Noise-Aware Optimizer

::: qufin.backends.noise_aware_optimizer
    options:
      members:
        - NoiseAwareConfig
        - NoiseAwareOptimizer
        - NoiseChannel
        - DepolarizingModel
        - circuit_noise_budget
        - compare_noise_aware_vs_agnostic
