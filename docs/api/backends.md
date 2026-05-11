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
