"""Unit tests for Modified Real Quantum Amplitude Estimation (mRQAE).

Covers the deterministic helpers (schedule, counting, least-squares fit,
direct encoding) plus end-to-end estimation on a known Bernoulli amplitude.

The end-to-end test is intentionally *not* marked ``slow`` (a 1-qubit problem
at low Grover depth runs in well under a second) so it both drives coverage in
CI and acts as a regression guard for the fit-model convention: mRQAE must
recover ``a = sin^2(theta)``, not ``sin^2(2*theta)``.
"""

from __future__ import annotations

import numpy as np
import pytest

from qufin.backends.base import CircuitResult
from qufin.options.amplitude_estimation.estimation_problem import EstimationProblem
from qufin.options.amplitude_estimation.mrqae import (
    ModifiedRealQAE,
    MRQAEConfig,
    MRQAEResult,
    direct_encode_distribution,
)


def _simple_bernoulli_problem(theta: float) -> EstimationProblem:
    """A|0> = cos(theta)|0> + sin(theta)|1>, so P(|1>) = sin^2(theta)."""
    from qiskit.circuit import QuantumCircuit

    qc = QuantumCircuit(1)
    qc.ry(2 * theta, 0)
    return EstimationProblem(state_preparation=qc, objective_qubits=[0], n_qubits=1)


def _estimator(theta: float = np.pi / 4, **cfg: object) -> ModifiedRealQAE:
    """mRQAE bound to a Bernoulli problem; helpers don't touch the backend."""
    from qufin.backends.qiskit_backend import QiskitAerBackend

    problem = _simple_bernoulli_problem(theta)
    config = MRQAEConfig(**cfg)  # type: ignore[arg-type]
    return ModifiedRealQAE(problem, config, QiskitAerBackend(seed=42))


class TestConfigAndResult:
    def test_config_defaults(self) -> None:
        cfg = MRQAEConfig()
        assert cfg.epsilon == 0.01
        assert cfg.shots_per_round == 1024
        assert cfg.max_depth == 50
        assert cfg.schedule == "linear"
        assert cfg.seed is None

    def test_result_defaults(self) -> None:
        res = MRQAEResult()
        assert res.estimate == 0.0
        assert res.confidence_interval == (0.0, 0.0)
        assert res.n_oracle_calls == 0
        assert res.n_rounds == 0
        assert res.depths_used == []
        assert res.round_details == []


class TestSchedule:
    def test_linear_schedule(self) -> None:
        est = _estimator(max_depth=5, schedule="linear")
        assert est._generate_schedule() == [0, 1, 2, 3, 4, 5]

    def test_exponential_schedule(self) -> None:
        est = _estimator(max_depth=8, schedule="exponential")
        assert est._generate_schedule() == [0, 1, 2, 4, 8]


class TestCountGood:
    def test_counts_objective_one(self) -> None:
        est = _estimator()
        result = CircuitResult(counts={"1": 800, "0": 200}, shots=1000)
        n_good, n_total = est._count_good(result)
        assert n_good == 800
        assert n_total == 1000

    def test_empty_counts(self) -> None:
        est = _estimator()
        n_good, n_total = est._count_good(CircuitResult(counts={}, shots=0))
        assert n_good == 0
        assert n_total == 0


class TestFitAmplitude:
    @pytest.mark.parametrize("theta_t", [0.3, np.pi / 4, 0.9, 1.2])
    def test_recovers_known_theta(self, theta_t: float) -> None:
        """Feeding noiseless p_k = sin^2((2k+1)*theta_t) must recover sin^2(theta_t)."""
        est = _estimator(shots_per_round=4096)
        depths = [0, 1, 2, 3, 4]
        measurements = [float(np.sin((2 * k + 1) * theta_t) ** 2) for k in depths]
        amplitude, uncertainty = est._fit_amplitude(depths, measurements)
        assert abs(amplitude - np.sin(theta_t) ** 2) < 1e-3
        assert uncertainty >= 0.0

    def test_single_depth_branch(self) -> None:
        """With one depth the fit falls back to a coarse single-shot estimate."""
        est = _estimator(shots_per_round=1024)
        amplitude, uncertainty = est._fit_amplitude([0], [0.5])
        assert 0.0 <= amplitude <= 1.0
        assert uncertainty > 0.0


class TestDirectEncode:
    def test_two_qubit_encoding_shape(self) -> None:
        values = np.array([0.0, 1.0, 2.0, 3.0])
        probs = np.array([0.1, 0.2, 0.3, 0.4])
        qc = direct_encode_distribution(values, probs, n_qubits=2)
        assert qc.num_qubits == 3  # 2 register + 1 ancilla

    def test_single_qubit_encoding_uses_cry(self) -> None:
        qc = direct_encode_distribution(
            np.array([0.0, 1.0]), np.array([0.5, 0.5]), n_qubits=1
        )
        assert qc.num_qubits == 2

    def test_unnormalized_probabilities_are_normalized(self) -> None:
        # Weights summing to 10 must not raise; they are normalized internally.
        qc = direct_encode_distribution(
            np.array([0.0, 1.0, 2.0, 3.0]),
            np.array([1.0, 2.0, 3.0, 4.0]),
            n_qubits=2,
        )
        assert qc.num_qubits == 3

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="length 2"):
            direct_encode_distribution(
                np.array([0.0, 1.0, 2.0]), np.array([0.3, 0.3, 0.4]), n_qubits=2
            )


class TestEstimateEndToEnd:
    def test_recovers_bernoulli_amplitude(self) -> None:
        """Regression guard: estimate must return sin^2(theta), not sin^2(2*theta)."""
        from qufin.backends.qiskit_backend import QiskitAerBackend

        theta = np.pi / 4
        expected = np.sin(theta) ** 2  # 0.5
        problem = _simple_bernoulli_problem(theta)
        config = MRQAEConfig(shots_per_round=2048, max_depth=3, seed=42)
        est = ModifiedRealQAE(problem, config, QiskitAerBackend(seed=42))

        res = est.estimate()

        assert 0.0 <= res.estimate <= 1.0
        assert abs(res.estimate - expected) < 0.1
        assert res.estimate < 0.9  # would be ~1.0 under the old factor-of-2 bug
        assert res.n_oracle_calls > 0
        assert res.n_rounds >= 1
        assert res.depths_used
        assert res.confidence_interval[0] <= res.confidence_interval[1]
        assert res.wall_time_s >= 0.0

    def test_single_round_uses_p0(self) -> None:
        """max_depth=0 exercises the depth-0-only estimation branch."""
        from qufin.backends.qiskit_backend import QiskitAerBackend

        theta = np.pi / 4
        problem = _simple_bernoulli_problem(theta)
        config = MRQAEConfig(shots_per_round=4096, max_depth=0, seed=42)
        est = ModifiedRealQAE(problem, config, QiskitAerBackend(seed=42))

        res = est.estimate()

        assert res.n_rounds == 1
        assert res.depths_used == [0]
        assert abs(res.estimate - np.sin(theta) ** 2) < 0.1

    @pytest.mark.slow
    @pytest.mark.parametrize("theta", [np.pi / 6, np.pi / 8])
    def test_accuracy_across_amplitudes(self, theta: float) -> None:
        from qufin.backends.qiskit_backend import QiskitAerBackend

        problem = _simple_bernoulli_problem(theta)
        config = MRQAEConfig(shots_per_round=4096, max_depth=6, seed=42)
        est = ModifiedRealQAE(problem, config, QiskitAerBackend(seed=42))
        res = est.estimate()
        assert abs(res.estimate - np.sin(theta) ** 2) < 0.1


class TestEuropeanProblemRegression:
    """Guard against the non-invertible state-preparation regression.

    ``build_european_estimation_problem`` must use an invertible (unitary) state
    preparation so that ``A^dag`` exists; a non-invertible ``initialize`` breaks
    every QAE algorithm when it constructs the Grover operator.
    """

    def test_european_state_prep_is_invertible(self) -> None:
        from qufin.options.amplitude_estimation.european_qae import (
            EuropeanQAESpec,
            build_european_estimation_problem,
        )

        problem, rescale = build_european_estimation_problem(EuropeanQAESpec(n_qubits=3))
        assert rescale > 0.0
        # Would raise CircuitError under the old qc.initialize() implementation.
        grover = problem.build_grover_operator()
        assert grover.num_qubits == problem.n_qubits
        assert problem.state_preparation.inverse().num_qubits == problem.n_qubits
