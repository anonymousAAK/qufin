"""Unit tests for Quantum Amplitude Estimation components."""

from __future__ import annotations

import numpy as np
import pytest

from qufin.options.amplitude_estimation.canonical import (
    CanonicalAmplitudeEstimation,
    CanonicalQAEConfig,
)
from qufin.options.amplitude_estimation.estimation_problem import EstimationProblem
from qufin.options.amplitude_estimation.iqae import (
    IQAEConfig,
    IterativeAmplitudeEstimation,
)


def _simple_bernoulli_problem(theta: float) -> EstimationProblem:
    """Create a simple 1-qubit estimation problem.

    A|0> = cos(theta)|0> + sin(theta)|1>
    So the probability of measuring |1> is sin^2(theta).
    """
    from qiskit.circuit import QuantumCircuit

    qc = QuantumCircuit(1)
    qc.ry(2 * theta, 0)

    return EstimationProblem(
        state_preparation=qc,
        objective_qubits=[0],
        n_qubits=1,
    )


class TestEstimationProblem:
    def test_basic_creation(self) -> None:
        problem = _simple_bernoulli_problem(np.pi / 4)
        assert problem.n_qubits == 1
        assert problem.objective_qubits == [0]

    def test_grover_operator_construction(self) -> None:
        problem = _simple_bernoulli_problem(np.pi / 6)
        grover = problem.build_grover_operator()
        assert grover is not None
        assert grover.num_qubits == 1


class TestCanonicalQAE:
    @pytest.mark.slow
    def test_known_amplitude(self) -> None:
        """Test QAE on a known amplitude: a = sin^2(pi/4) = 0.5."""
        from qufin.backends.qiskit_backend import QiskitAerBackend

        theta = np.pi / 4
        expected_a = np.sin(theta) ** 2  # 0.5
        problem = _simple_bernoulli_problem(theta)

        backend = QiskitAerBackend(seed=42)
        config = CanonicalQAEConfig(n_eval_qubits=3, shots=4096, seed=42)
        qae = CanonicalAmplitudeEstimation(problem, config, backend)
        result = qae.estimate()

        assert result.estimate >= 0
        assert result.estimate <= 1
        # With 3 eval qubits, precision is pi/8 ~ 0.39
        # So estimate should be within a large tolerance
        assert abs(result.estimate - expected_a) < 0.4
        assert result.n_oracle_calls == 7  # 2^3 - 1
        assert result.wall_time_s > 0

    @pytest.mark.slow
    def test_small_amplitude(self) -> None:
        """Test QAE on a small amplitude."""
        from qufin.backends.qiskit_backend import QiskitAerBackend

        theta = np.pi / 8
        problem = _simple_bernoulli_problem(theta)

        backend = QiskitAerBackend(seed=42)
        config = CanonicalQAEConfig(n_eval_qubits=4, shots=8192, seed=42)
        qae = CanonicalAmplitudeEstimation(problem, config, backend)
        result = qae.estimate()

        assert 0 <= result.estimate <= 1
        assert result.confidence_interval[0] <= result.confidence_interval[1]


class TestIQAE:
    @pytest.mark.slow
    def test_known_amplitude(self) -> None:
        """Test IQAE on a known amplitude."""
        from qufin.backends.qiskit_backend import QiskitAerBackend

        theta = np.pi / 4
        problem = _simple_bernoulli_problem(theta)

        backend = QiskitAerBackend(seed=42)
        config = IQAEConfig(
            epsilon_target=0.05,
            alpha=0.05,
            shots_per_round=2048,
            max_iterations=10,
            seed=42,
        )
        iqae = IterativeAmplitudeEstimation(problem, config, backend)
        result = iqae.estimate()

        assert 0 <= result.estimate <= 1
        assert result.n_rounds > 0
        assert result.n_oracle_calls > 0
        assert result.confidence_interval[0] <= result.estimate
        assert result.estimate <= result.confidence_interval[1]

    @pytest.mark.slow
    def test_iqae_convergence(self) -> None:
        """IQAE should produce progressively tighter confidence intervals."""
        from qufin.backends.qiskit_backend import QiskitAerBackend

        problem = _simple_bernoulli_problem(np.pi / 6)
        backend = QiskitAerBackend(seed=42)

        config = IQAEConfig(
            epsilon_target=0.1,
            shots_per_round=1024,
            max_iterations=15,
            seed=42,
        )
        iqae = IterativeAmplitudeEstimation(problem, config, backend)
        result = iqae.estimate()

        ci_width = result.confidence_interval[1] - result.confidence_interval[0]
        assert ci_width < 0.5  # should have narrowed from initial [0,1]
