"""Configurable noise models for quantum circuit simulation.

Provides realistic noise channels that can be composed and applied to
the Qiskit Aer simulator. Supports per-gate error rates, thermal
relaxation (T1/T2), readout errors, and device-calibrated profiles.

References
----------
Nielsen & Chuang, Ch. 8 — Quantum noise and quantum operations.
Georgopoulos et al., "Modeling and simulating the noisy behavior of
  near-term quantum computers" (2021).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from qufin.backends.base import Backend


@dataclass
class NoiseProfile:
    """Configurable noise profile for quantum simulation.

    Parameters
    ----------
    single_gate_error : float
        Depolarizing error rate for single-qubit gates (e.g., 1e-4 to 1e-2).
    two_gate_error : float
        Depolarizing error rate for two-qubit gates (typically ~10x single).
    readout_error : float
        Probability of bit-flip on measurement (symmetric for 0->1 and 1->0).
    t1_us : float
        T1 (amplitude damping) time in microseconds.
    t2_us : float
        T2 (dephasing) time in microseconds. Must satisfy T2 <= 2*T1.
    single_gate_time_us : float
        Duration of a single-qubit gate in microseconds.
    two_gate_time_us : float
        Duration of a two-qubit gate in microseconds.
    name : str
        Human-readable name for this profile.
    """

    single_gate_error: float = 1e-4
    two_gate_error: float = 1e-3
    readout_error: float = 1e-2
    t1_us: float = 100.0
    t2_us: float = 80.0
    single_gate_time_us: float = 0.035
    two_gate_time_us: float = 0.3
    name: str = "custom"


# ---------------------------------------------------------------------------
# Preset device profiles (approximate, for benchmarking)
# ---------------------------------------------------------------------------

IBM_EAGLE_R3 = NoiseProfile(
    single_gate_error=2.4e-4,
    two_gate_error=7.5e-3,
    readout_error=1.1e-2,
    t1_us=290.0,
    t2_us=150.0,
    single_gate_time_us=0.06,
    two_gate_time_us=0.66,
    name="ibm_eagle_r3",
)

IBM_HERON_R2 = NoiseProfile(
    single_gate_error=1.5e-4,
    two_gate_error=3.5e-3,
    readout_error=6e-3,
    t1_us=350.0,
    t2_us=200.0,
    single_gate_time_us=0.035,
    two_gate_time_us=0.08,
    name="ibm_heron_r2",
)

IDEAL = NoiseProfile(
    single_gate_error=0.0,
    two_gate_error=0.0,
    readout_error=0.0,
    t1_us=1e9,
    t2_us=1e9,
    name="ideal",
)

NOISY_NEAR_TERM = NoiseProfile(
    single_gate_error=1e-3,
    two_gate_error=1e-2,
    readout_error=3e-2,
    t1_us=50.0,
    t2_us=30.0,
    single_gate_time_us=0.05,
    two_gate_time_us=0.5,
    name="noisy_near_term",
)

DEVICE_PROFILES: dict[str, NoiseProfile] = {
    "ideal": IDEAL,
    "ibm_eagle_r3": IBM_EAGLE_R3,
    "ibm_heron_r2": IBM_HERON_R2,
    "noisy_near_term": NOISY_NEAR_TERM,
}


def build_noise_model(profile: NoiseProfile) -> Any:
    """Build a Qiskit Aer NoiseModel from a NoiseProfile.

    Constructs depolarizing errors on all gates, thermal relaxation,
    and readout bit-flip errors.

    Returns
    -------
    qiskit_aer.noise.NoiseModel
    """
    from qiskit_aer.noise import (
        NoiseModel,
        ReadoutError,
        depolarizing_error,
        thermal_relaxation_error,
    )

    noise_model = NoiseModel()

    # 1. Depolarizing errors on gates
    if profile.single_gate_error > 0:
        err_1q = depolarizing_error(profile.single_gate_error, 1)
        noise_model.add_all_qubit_quantum_error(
            err_1q, ["u1", "u2", "u3", "rx", "ry", "rz", "x", "y", "z",
                      "h", "s", "sdg", "t", "tdg", "sx", "sxdg", "id"]
        )

    if profile.two_gate_error > 0:
        err_2q = depolarizing_error(profile.two_gate_error, 2)
        noise_model.add_all_qubit_quantum_error(
            err_2q, ["cx", "cz", "cy", "ecr", "rzx", "rzz", "swap"]
        )

    # 2. Thermal relaxation (T1/T2)
    if profile.t1_us < 1e8:  # skip for ideal
        err_1q_thermal = thermal_relaxation_error(
            profile.t1_us, profile.t2_us, profile.single_gate_time_us
        )
        noise_model.add_all_qubit_quantum_error(
            err_1q_thermal,
            ["u1", "u2", "u3", "rx", "ry", "rz", "x", "y", "z",
             "h", "s", "sdg", "t", "tdg", "sx", "sxdg"],
        )

        err_2q_thermal = thermal_relaxation_error(
            profile.t1_us, profile.t2_us, profile.two_gate_time_us
        )
        err_2q_thermal_pair = err_2q_thermal.expand(err_2q_thermal)
        noise_model.add_all_qubit_quantum_error(
            err_2q_thermal_pair,
            ["cx", "cz", "ecr", "rzx"],
        )

    # 3. Readout error (symmetric bit-flip)
    if profile.readout_error > 0:
        p = profile.readout_error
        readout_err = ReadoutError([[1 - p, p], [p, 1 - p]])
        noise_model.add_all_qubit_readout_error(readout_err)

    return noise_model


class NoisyAerBackend(Backend):
    """Qiskit Aer backend with a configurable noise model.

    Drop-in replacement for QiskitAerBackend that adds realistic noise.

    Parameters
    ----------
    profile : NoiseProfile
        Noise parameters. Use a preset (e.g., IBM_HERON_R2) or custom.
    seed : int | None
        Random seed for reproducibility.
    """

    def __init__(
        self, profile: NoiseProfile = NOISY_NEAR_TERM, seed: int | None = 42
    ) -> None:
        from qiskit_aer import AerSimulator

        self._profile = profile
        self._seed = seed
        self._noise_model = build_noise_model(profile)
        self._sim = AerSimulator(
            noise_model=self._noise_model,
            seed_simulator=seed,
        )

    @property
    def backend_id(self) -> str:
        return f"noisy-aer-{self._profile.name}"

    @property
    def noise_profile(self) -> NoiseProfile:
        return self._profile

    def is_simulator(self) -> bool:
        return True

    def run(self, circuit: Any, shots: int = 1024):
        """Execute a circuit with noise."""
        from qiskit import transpile

        from qufin.backends.base import CircuitResult

        transpiled = transpile(circuit, self._sim)
        job = self._sim.run(transpiled, shots=shots)
        result = job.result()
        counts = result.get_counts()
        flat_counts = {k.replace(" ", ""): v for k, v in counts.items()}
        return CircuitResult(
            counts=flat_counts,
            shots=shots,
            backend_id=self.backend_id,
            metadata={"noise_profile": self._profile.name},
        )

    def statevector(self, circuit: Any):
        """Not meaningful under noise — use density matrix instead."""
        raise NotImplementedError(
            "Statevector is not meaningful under noise. "
            "Use the 'run' method with shots, or use an ideal backend."
        )


def sweep_noise(
    circuit: Any,
    error_rates: list[float],
    shots: int = 4096,
    seed: int | None = 42,
) -> list[dict[str, Any]]:
    """Run a circuit at multiple noise levels and collect statistics.

    Useful for Zero-Noise Extrapolation (ZNE) or for understanding
    noise sensitivity of an algorithm.

    Parameters
    ----------
    circuit : QuantumCircuit
        Circuit to execute.
    error_rates : list[float]
        Two-qubit gate error rates to sweep (e.g., [0, 0.001, 0.005, 0.01]).
    shots : int
        Shots per execution.
    seed : int | None
        Random seed.

    Returns
    -------
    List of dicts with keys: error_rate, counts, most_frequent, entropy.
    """
    results = []
    for rate in error_rates:
        profile = NoiseProfile(
            single_gate_error=rate / 10,
            two_gate_error=rate,
            readout_error=rate * 3,
            name=f"sweep_{rate:.4f}",
        )
        if rate == 0:
            profile = IDEAL

        backend = NoisyAerBackend(profile=profile, seed=seed)

        # Add measurements if the circuit doesn't have them
        from qiskit.circuit import QuantumCircuit
        n = circuit.num_qubits
        meas_circ = QuantumCircuit(n, n)
        meas_circ.compose(circuit, inplace=True)
        meas_circ.measure(range(n), range(n))

        result = backend.run(meas_circ, shots=shots)

        # Shannon entropy of the distribution
        probs = np.array(list(result.probabilities.values()))
        probs = probs[probs > 0]
        entropy = float(-np.sum(probs * np.log2(probs)))

        results.append({
            "error_rate": rate,
            "counts": result.counts,
            "most_frequent": result.most_frequent,
            "entropy": entropy,
            "n_unique": len(result.counts),
        })

    return results
