"""Unit tests for MLAE and FQAE."""

from __future__ import annotations

import numpy as np
import pytest

from qufin.options.amplitude_estimation.estimation_problem import EstimationProblem
from qufin.options.amplitude_estimation.mlae import (
    MLAEConfig,
    MaximumLikelihoodAmplitudeEstimation,
)
from qufin.options.amplitude_estimation.fqae import (
    FQAEConfig,
    FaithfulAmplitudeEstimation,
)


def _simple_bernoulli_problem(theta: float) -> EstimationProblem:
    from qiskit.circuit import QuantumCircuit

    qc = QuantumCircuit(1)
    qc.ry(2 * theta, 0)
    return EstimationProblem(
        state_preparation=qc,
        objective_qubits=[0],
        n_qubits=1,
    )


class TestMLAE:
    @pytest.mark.slow
    def test_known_amplitude(self) -> None:
        """MLAE should estimate a known amplitude."""
        from qufin.backends.qiskit_backend import QiskitAerBackend

        theta = np.pi / 4
        expected = np.sin(theta) ** 2  # 0.5
        problem = _simple_bernoulli_problem(theta)

        backend = QiskitAerBackend(seed=42)
        config = MLAEConfig(
            evaluation_schedule=[0, 1, 2, 4, 8],
            n_shots_per_round=2048,
            seed=42,
        )
        mlae = MaximumLikelihoodAmplitudeEstimation(problem, config, backend)
        result = mlae.estimate()

        assert 0 <= result.estimate <= 1
        assert abs(result.estimate - expected) < 0.15
        assert result.n_oracle_calls > 0
        assert result.n_rounds == 5
        assert result.log_likelihood < 0  # negative log-likelihood

    @pytest.mark.slow
    def test_small_amplitude(self) -> None:
        from qufin.backends.qiskit_backend import QiskitAerBackend

        theta = np.pi / 8
        expected = np.sin(theta) ** 2
        problem = _simple_bernoulli_problem(theta)

        backend = QiskitAerBackend(seed=42)
        config = MLAEConfig(
            evaluation_schedule=[0, 1, 2, 4],
            n_shots_per_round=4096,
            seed=42,
        )
        mlae = MaximumLikelihoodAmplitudeEstimation(problem, config, backend)
        result = mlae.estimate()

        assert 0 <= result.estimate <= 1
        assert result.confidence_interval[0] <= result.confidence_interval[1]


class TestFQAE:
    @pytest.mark.slow
    def test_known_amplitude(self) -> None:
        """FQAE should estimate a known amplitude."""
        from qufin.backends.qiskit_backend import QiskitAerBackend

        theta = np.pi / 4
        expected = np.sin(theta) ** 2
        problem = _simple_bernoulli_problem(theta)

        backend = QiskitAerBackend(seed=42)
        config = FQAEConfig(max_depth=4, n_shots_per_round=2048, seed=42)
        fqae = FaithfulAmplitudeEstimation(problem, config, backend)
        result = fqae.estimate()

        assert 0 <= result.estimate <= 1
        assert abs(result.estimate - expected) < 0.3
        assert result.n_rounds == 5  # depths 0,1,2,3,4
        assert result.max_depth_used == 4

    @pytest.mark.slow
    def test_low_depth(self) -> None:
        """FQAE with depth=1 should still produce a result."""
        from qufin.backends.qiskit_backend import QiskitAerBackend

        problem = _simple_bernoulli_problem(np.pi / 6)
        backend = QiskitAerBackend(seed=42)
        config = FQAEConfig(max_depth=1, n_shots_per_round=4096, seed=42)
        fqae = FaithfulAmplitudeEstimation(problem, config, backend)
        result = fqae.estimate()

        assert 0 <= result.estimate <= 1
        assert result.max_depth_used == 1
