# Backends

qufin uses a pluggable backend system. All quantum algorithms accept a `Backend` object, allowing you to switch between simulators and real hardware with a single line change.

## Available Backends

| Backend | Module | Use Case |
|---------|--------|----------|
| `MockBackend` | `qufin.backends.mock` | Unit testing, deterministic |
| `QiskitAerBackend` | `qufin.backends.qiskit_backend` | Local simulation |
| `NoisyAerBackend` | `qufin.backends.noise_models` | Noise-aware simulation |
| `IBMRuntimeBackend` | `qufin.backends.ibm_runtime` | IBM Quantum hardware |

## Quick Usage

```python
from qufin.backends.qiskit_backend import QiskitAerBackend

backend = QiskitAerBackend(seed=42)
result = backend.run(circuit, shots=4096)
print(result.counts)           # {'00': 2048, '11': 2048}
print(result.most_frequent)    # '00'
print(result.probabilities)    # {'00': 0.5, '11': 0.5}
```

## Noise Models

Simulate realistic hardware noise:

```python
from qufin.backends.noise_models import NoisyAerBackend, IBM_HERON_R2, NOISY_NEAR_TERM

# Use a device-calibrated noise profile
backend = NoisyAerBackend(profile=IBM_HERON_R2, seed=42)

# Or use a generic noisy profile
backend = NoisyAerBackend(profile=NOISY_NEAR_TERM)
```

### Device Profiles

| Profile | 1Q Error | 2Q Error | Readout Error | T1 (us) |
|---------|----------|----------|---------------|---------|
| `IDEAL` | 0 | 0 | 0 | inf |
| `IBM_EAGLE_R3` | 2.4e-4 | 7.5e-3 | 1.1e-2 | 290 |
| `IBM_HERON_R2` | 1.5e-4 | 3.5e-3 | 6e-3 | 350 |
| `NOISY_NEAR_TERM` | 1e-3 | 1e-2 | 3e-2 | 50 |

### Custom Noise Profile

```python
from qufin.backends.noise_models import NoiseProfile, NoisyAerBackend

profile = NoiseProfile(
    single_gate_error=5e-4,
    two_gate_error=5e-3,
    readout_error=1e-2,
    t1_us=200.0,
    t2_us=100.0,
    name="my_device",
)
backend = NoisyAerBackend(profile=profile)
```

### Noise Sweep

Analyze algorithm sensitivity to noise:

```python
from qufin.backends.noise_models import sweep_noise

results = sweep_noise(
    circuit,
    error_rates=[0, 0.001, 0.005, 0.01, 0.05],
    shots=4096,
)
for r in results:
    print(f"Error rate: {r['error_rate']}, Entropy: {r['entropy']:.2f}")
```

## Error Mitigation

### Zero-Noise Extrapolation (ZNE)

```python
from qufin.backends.error_mitigation import zne_extrapolate

mitigated = zne_extrapolate(
    circuit, backend,
    scale_factors=[1, 3, 5],
    shots=8192,
)
print(f"Mitigated counts: {mitigated.mitigated_counts}")
```

### Readout Error Mitigation

```python
from qufin.backends.error_mitigation import calibrate_readout, mitigate_readout

# Calibrate (signature: calibrate_readout(n_qubits, backend, shots)).
cal_matrix = calibrate_readout(4, backend, shots=8192)

# Apply correction; returns a MitigationResult (use .mitigated_counts).
corrected = mitigate_readout(raw_counts, cal_matrix, shots=8192)
```

### TREX (Twirled Readout Error eXtinction)

```python
from qufin.backends.error_mitigation import trex_mitigate

# circuit must NOT include measurements (added internally).
result = trex_mitigate(circuit, backend, n_twirls=16, shots_per_twirl=1024)
```

## Backend Interface

All backends implement:

```python
class Backend:
    def run(self, circuit, shots=1024) -> CircuitResult: ...
    def statevector(self, circuit) -> NDArray: ...
    def backend_id(self) -> str: ...
    def is_simulator(self) -> bool: ...
```

`CircuitResult` provides:

- `.counts` — Dict of bitstring -> count
- `.probabilities` — Dict of bitstring -> probability
- `.most_frequent` — Most common bitstring
- `.shots` — Total number of shots
- `.backend_id` — Backend identifier

## Advanced Backend Features (v0.3.0)

### Auto-Selecting a Backend

The `auto_select_backend` function analyzes your circuit and picks the best
available backend based on qubit count, gate depth, and installed packages.

```python
from qufin.backends.auto_select import auto_select_backend

backend = auto_select_backend(circuit)
result = backend.run(circuit, shots=4096)
```

### Using the Finance Transpiler to Optimize QAOA Circuits

`FinanceTranspiler` exploits the structure of financial QUBO problems to
produce better qubit layouts and commuting-gate groupings.

```python
from qufin.backends.transpiler import FinanceTranspiler

transpiler = FinanceTranspiler(optimization_level=3, seed=42)

# QUBO-structure-aware optimization; returns (optimized_circuit, report).
optimized_circuit, report = transpiler.optimize_qaoa_circuit(circuit, qubo_matrix=Q)
print(f"Depth:  {report.original_depth} -> {report.optimized_depth}")
print(f"CNOTs:  {report.original_cx_count} -> {report.optimized_cx_count}")

# A CNOT-focused pass is also available:
optimized_circuit, report = transpiler.reduce_cnot_count(circuit)
```

The transpiler applies Qiskit's `optimization_level=3` passes (gate cancellation,
commutation, template matching). CNOT reduction depends on circuit structure:
dense QAOA cost layers (`cx`-`rz`-`cx` on distinct control/target pairs) contain
little cancellable redundancy, so they often see no reduction; circuits with
redundant entanglers benefit more.

### Applying Dynamical Decoupling

Insert DD sequences into idle periods to suppress decoherence on real hardware.

```python
from qufin.backends.dynamical_decoupling import (
    insert_dd_sequences, DDConfig, DDSequence,
)

# circuit must NOT include measurements.
config = DDConfig(sequence_type=DDSequence.XY4)
protected_circuit = insert_dd_sequences(circuit, config)
```

### Using M3 Measurement Mitigation

M3 (Matrix-free Measurement Mitigation) scales to large qubit counts by
using tensored calibration instead of full calibration matrices.

```python
from qufin.backends.m3_mitigation import M3Mitigator, M3Config

mitigator = M3Mitigator(config=M3Config(n_calibration_shots=8192))
mitigator.calibrate(backend, qubit_list=[0, 1, 2, 3])
corrected_counts = mitigator.apply(raw_counts)
```

### Noise-Aware Optimization

Incorporate estimated noise channels directly into the optimizer cost
function so the variational loop learns to avoid noisy gates.

```python
from qufin.backends.noise_aware_optimizer import NoiseAwareOptimizer, NoiseAwareConfig

config = NoiseAwareConfig(noise_weight=0.1, max_iterations=200)
optimizer = NoiseAwareOptimizer(backend, config)
opt_result = optimizer.minimize(cost_fn, initial_params)
```

### Cost Estimation for Cloud QPU

Estimate the dollar cost of running a circuit on Amazon Braket or IonQ
before submitting the job.

```python
from qufin.backends.braket_backend import estimate_cost, IONQ_ARIA

estimate = estimate_cost(
    target=IONQ_ARIA,
    shots=10_000,
)
print(f"Estimated cost: ${estimate.total_usd:.2f}")
```
