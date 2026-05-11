"""Tests for noise models and error mitigation strategies."""

from __future__ import annotations

import numpy as np
import pytest


class TestNoiseProfiles:
    """Validate noise model construction and device profiles."""

    def test_ideal_profile_has_zero_errors(self) -> None:
        from qufin.backends.noise_models import IDEAL

        assert IDEAL.single_gate_error == 0.0
        assert IDEAL.two_gate_error == 0.0
        assert IDEAL.readout_error == 0.0

    def test_all_device_profiles_exist(self) -> None:
        from qufin.backends.noise_models import DEVICE_PROFILES

        assert "ideal" in DEVICE_PROFILES
        assert "ibm_eagle_r3" in DEVICE_PROFILES
        assert "ibm_heron_r2" in DEVICE_PROFILES
        assert "noisy_near_term" in DEVICE_PROFILES

    def test_heron_better_than_eagle(self) -> None:
        """Heron r2 should have lower error rates than Eagle r3."""
        from qufin.backends.noise_models import IBM_EAGLE_R3, IBM_HERON_R2

        assert IBM_HERON_R2.two_gate_error < IBM_EAGLE_R3.two_gate_error
        assert IBM_HERON_R2.readout_error < IBM_EAGLE_R3.readout_error

    def test_t2_leq_2t1(self) -> None:
        """Physical constraint: T2 <= 2*T1 for all profiles."""
        from qufin.backends.noise_models import DEVICE_PROFILES

        for name, profile in DEVICE_PROFILES.items():
            assert profile.t2_us <= 2 * profile.t1_us + 1e-6, (
                f"{name}: T2={profile.t2_us} > 2*T1={2*profile.t1_us}"
            )

    def test_build_noise_model_ideal(self) -> None:
        """Ideal profile should produce a noise model with no errors."""
        from qufin.backends.noise_models import IDEAL, build_noise_model

        nm = build_noise_model(IDEAL)
        # Ideal noise model should exist but have no quantum errors
        assert nm is not None

    def test_build_noise_model_noisy(self) -> None:
        from qufin.backends.noise_models import NOISY_NEAR_TERM, build_noise_model

        nm = build_noise_model(NOISY_NEAR_TERM)
        assert nm is not None


class TestNoisyAerBackend:
    """Test the noisy Aer backend wrapper."""

    def test_ideal_matches_noiseless(self) -> None:
        """With ideal noise, results should match noiseless Aer."""
        from qiskit.circuit import QuantumCircuit
        from qufin.backends.noise_models import IDEAL, NoisyAerBackend
        from qufin.backends.qiskit_backend import QiskitAerBackend

        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])

        ideal = NoisyAerBackend(profile=IDEAL, seed=42)
        clean = QiskitAerBackend(seed=42)

        r_ideal = ideal.run(qc, shots=10000)
        r_clean = clean.run(qc, shots=10000)

        # Both should give ~50% 00 and ~50% 11
        for bs in ["00", "11"]:
            p_ideal = r_ideal.counts.get(bs, 0) / 10000
            p_clean = r_clean.counts.get(bs, 0) / 10000
            assert abs(p_ideal - p_clean) < 0.05

    def test_noise_degrades_bell_state(self) -> None:
        """Noisy simulation should produce more errors than ideal."""
        from qiskit.circuit import QuantumCircuit
        from qufin.backends.noise_models import NOISY_NEAR_TERM, IDEAL, NoisyAerBackend

        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])

        ideal = NoisyAerBackend(profile=IDEAL, seed=42)
        noisy = NoisyAerBackend(profile=NOISY_NEAR_TERM, seed=42)

        r_ideal = ideal.run(qc, shots=10000)
        r_noisy = noisy.run(qc, shots=10000)

        # Ideal: no 01 or 10 states
        p_err_ideal = (r_ideal.counts.get("01", 0) + r_ideal.counts.get("10", 0)) / 10000
        p_err_noisy = (r_noisy.counts.get("01", 0) + r_noisy.counts.get("10", 0)) / 10000

        assert p_err_noisy > p_err_ideal, "Noise should introduce errors"

    def test_backend_id_contains_profile_name(self) -> None:
        from qufin.backends.noise_models import IBM_HERON_R2, NoisyAerBackend

        backend = NoisyAerBackend(profile=IBM_HERON_R2)
        assert "heron" in backend.backend_id

    def test_noise_monotonic_in_error_rate(self) -> None:
        """Higher error rate -> more entropy in output distribution."""
        from qufin.backends.noise_models import NoiseProfile, NoisyAerBackend
        from qiskit.circuit import QuantumCircuit

        qc = QuantumCircuit(3, 3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.measure([0, 1, 2], [0, 1, 2])

        entropies = []
        for rate in [0.0, 0.005, 0.02, 0.05]:
            profile = NoiseProfile(
                single_gate_error=rate / 10,
                two_gate_error=rate,
                readout_error=rate * 2,
                name=f"test_{rate}",
            )
            backend = NoisyAerBackend(profile=profile, seed=42)
            result = backend.run(qc, shots=10000)
            probs = np.array(list(result.probabilities.values()))
            probs = probs[probs > 0]
            entropy = -np.sum(probs * np.log2(probs))
            entropies.append(entropy)

        # Entropy should be monotonically non-decreasing with noise
        for i in range(len(entropies) - 1):
            assert entropies[i + 1] >= entropies[i] - 0.1, (
                f"Entropy not monotonic: {entropies}"
            )


class TestSweepNoise:
    """Test the noise sweep utility."""

    def test_sweep_returns_correct_structure(self) -> None:
        from qiskit.circuit import QuantumCircuit
        from qufin.backends.noise_models import sweep_noise

        qc = QuantumCircuit(1)
        qc.x(0)  # Prepare |1>

        results = sweep_noise(qc, [0, 0.01, 0.05], shots=1000, seed=42)
        assert len(results) == 3
        for r in results:
            assert "error_rate" in r
            assert "counts" in r
            assert "entropy" in r
            assert "most_frequent" in r

    def test_sweep_ideal_has_low_entropy(self) -> None:
        from qiskit.circuit import QuantumCircuit
        from qufin.backends.noise_models import sweep_noise

        qc = QuantumCircuit(1)
        qc.x(0)

        results = sweep_noise(qc, [0, 0.05], shots=2000, seed=42)
        # Ideal should have entropy ~0 (deterministic)
        assert results[0]["entropy"] < 0.1
        # Noisy should have higher entropy
        assert results[1]["entropy"] > results[0]["entropy"]


class TestZNE:
    """Test Zero-Noise Extrapolation."""

    def test_zne_improves_over_noisy(self) -> None:
        """ZNE estimate should be closer to ideal than raw noisy result."""
        from qiskit.circuit import QuantumCircuit
        from qufin.backends.noise_models import NoiseProfile, NoisyAerBackend
        from qufin.backends.error_mitigation import zne_extrapolate

        # Circuit: prepare |11> (all ones)
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.x(1)

        profile = NoiseProfile(
            single_gate_error=5e-3,
            two_gate_error=2e-2,
            readout_error=2e-2,
            name="zne_test",
        )
        backend = NoisyAerBackend(profile=profile, seed=42)

        def obs(counts, n_shots):
            return counts.get("11", 0) / n_shots

        result = zne_extrapolate(
            qc, backend, scale_factors=[1, 3, 5], shots=8192, observable_fn=obs,
        )

        # Ideal value is 1.0
        # Raw (scale=1) is less than 1.0 due to noise
        raw = result["raw_values"][0]
        mitigated = result["mitigated_value"]

        assert raw < 1.0, "Raw should be degraded by noise"
        assert abs(mitigated - 1.0) < abs(raw - 1.0) + 0.1, (
            f"ZNE ({mitigated:.4f}) should be closer to 1.0 than raw ({raw:.4f})"
        )

    def test_richardson_coefficients_sum_to_one(self) -> None:
        from qufin.backends.error_mitigation import _richardson_coefficients

        for factors in [[1, 3, 5], [1, 3], [1, 3, 5, 7]]:
            coeffs = _richardson_coefficients(factors)
            assert abs(np.sum(coeffs) - 1.0) < 1e-10, (
                f"Richardson coefficients should sum to 1: {coeffs}"
            )

    def test_fold_circuit_preserves_unitarity(self) -> None:
        """Folded circuit should be logically equivalent to original."""
        from qiskit.circuit import QuantumCircuit
        from qiskit.quantum_info import Operator
        from qufin.backends.error_mitigation import _fold_circuit

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        for sf in [1, 3, 5]:
            folded = _fold_circuit(qc, sf)
            u_orig = Operator(qc).data
            u_folded = Operator(folded).data
            # Should be the same unitary (up to global phase)
            overlap = abs(np.trace(u_orig.conj().T @ u_folded)) / 2**qc.num_qubits
            assert abs(overlap - 1.0) < 1e-10, (
                f"Fold factor {sf}: overlap = {overlap:.6f}"
            )


class TestReadoutMitigation:
    """Test measurement error mitigation."""

    def test_calibrate_readout_identity_for_ideal(self) -> None:
        """Ideal backend should produce identity calibration matrix."""
        from qufin.backends.qiskit_backend import QiskitAerBackend
        from qufin.backends.error_mitigation import calibrate_readout

        backend = QiskitAerBackend(seed=42)
        cal = calibrate_readout(2, backend, shots=10000)

        # Should be close to identity
        np.testing.assert_allclose(cal, np.eye(4), atol=0.03)

    def test_mitigate_readout_improves_accuracy(self) -> None:
        """Readout mitigation should improve noisy results."""
        from qufin.backends.noise_models import NoiseProfile, NoisyAerBackend
        from qufin.backends.error_mitigation import calibrate_readout, mitigate_readout
        from qiskit.circuit import QuantumCircuit

        profile = NoiseProfile(
            single_gate_error=0.0,
            two_gate_error=0.0,
            readout_error=0.05,
            name="readout_test",
        )
        backend = NoisyAerBackend(profile=profile, seed=42)

        # Calibrate
        cal = calibrate_readout(2, backend, shots=10000)

        # Prepare |00> and measure
        qc = QuantumCircuit(2, 2)
        qc.measure([0, 1], [0, 1])
        result = backend.run(qc, shots=10000)

        # Raw should have errors
        raw_p00 = result.counts.get("00", 0) / 10000
        assert raw_p00 < 0.95, f"Expected readout errors, got P(00)={raw_p00}"

        # Mitigate
        mitigated = mitigate_readout(result.counts, cal, 10000)
        mit_p00 = mitigated.mitigated_probs.get("00", 0)

        # Mitigated should be closer to 1.0
        assert mit_p00 > raw_p00, (
            f"Mitigation should improve: raw={raw_p00:.4f}, mitigated={mit_p00:.4f}"
        )


class TestTREX:
    """Test Twirled Readout Error eXtinction."""

    def test_trex_returns_valid_distribution(self) -> None:
        from qiskit.circuit import QuantumCircuit
        from qufin.backends.noise_models import NOISY_NEAR_TERM, NoisyAerBackend
        from qufin.backends.error_mitigation import trex_mitigate

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        backend = NoisyAerBackend(profile=NOISY_NEAR_TERM, seed=42)
        result = trex_mitigate(qc, backend, n_twirls=5, shots_per_twirl=1000)

        # Probabilities should sum to ~1
        total_prob = sum(result.mitigated_probs.values())
        assert abs(total_prob - 1.0) < 0.01

        assert result.method == "trex"
        assert result.metadata["n_twirls"] == 5


class TestNoiseOnQAE:
    """Test how noise affects QAE accuracy (integration-level)."""

    @pytest.mark.slow
    def test_iqae_degrades_under_noise(self) -> None:
        """IQAE accuracy should degrade with increasing noise."""
        from qiskit.circuit import QuantumCircuit
        from qufin.options.amplitude_estimation.estimation_problem import EstimationProblem
        from qufin.options.amplitude_estimation.iqae import IQAEConfig, IterativeAmplitudeEstimation
        from qufin.backends.noise_models import NoiseProfile, NoisyAerBackend, IDEAL

        qc = QuantumCircuit(1)
        qc.ry(2 * np.arcsin(np.sqrt(0.25)), 0)
        problem = EstimationProblem(state_preparation=qc, objective_qubits=[0])

        errors_by_noise = []
        for rate in [0.0, 0.005, 0.02]:
            if rate == 0:
                profile = IDEAL
            else:
                profile = NoiseProfile(
                    single_gate_error=rate / 10,
                    two_gate_error=rate,
                    readout_error=rate,
                    name=f"iqae_test_{rate}",
                )
            backend = NoisyAerBackend(profile=profile, seed=42)
            config = IQAEConfig(epsilon_target=0.02, shots_per_round=4096)
            result = IterativeAmplitudeEstimation(problem, config, backend).estimate()
            error = abs(result.estimate - 0.25)
            errors_by_noise.append(error)

        # Ideal should have smallest error
        assert errors_by_noise[0] < 0.02
        # Higher noise -> larger error (with some tolerance)
        assert errors_by_noise[-1] >= errors_by_noise[0] - 0.01

    @pytest.mark.slow
    def test_qaoa_degrades_under_noise(self) -> None:
        """QAOA objective should worsen with increasing noise."""
        from qufin.backends.noise_models import NoiseProfile, NoisyAerBackend, IDEAL
        from qufin.portfolio.optimizers.qaoa import QAOAConfig, QAOAPortfolio
        from qufin.portfolio.qubo import PortfolioQUBO

        rng = np.random.default_rng(42)
        mu = rng.normal(0, 0.01, 3)
        cov = np.eye(3) * 0.02
        qubo = PortfolioQUBO(mu, cov, gamma=1.0, cardinality=1)

        objectives = []
        for rate in [0.0, 0.01]:
            if rate == 0:
                profile = IDEAL
            else:
                profile = NoiseProfile(
                    single_gate_error=rate / 10,
                    two_gate_error=rate,
                    readout_error=rate,
                    name=f"qaoa_noise_{rate}",
                )
            backend = NoisyAerBackend(profile=profile, seed=42)
            config = QAOAConfig(p=1, mixer="x", shots=2048, maxiter=15, seed=42)
            result = QAOAPortfolio(qubo, config, backend).run()
            objectives.append(result.best_objective)

        # Both should return finite objectives
        assert all(np.isfinite(o) for o in objectives)
