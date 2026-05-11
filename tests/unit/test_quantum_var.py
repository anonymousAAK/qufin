"""Unit tests for quantum VaR (Woerner & Egger 1806.06893)."""

from __future__ import annotations

import numpy as np
import pytest

from qufin.backends.qiskit_backend import QiskitAerBackend
from qufin.options.distributions import normal_distribution
from qufin.risk.quantum_var import (
    QuantumVaRConfig,
    _build_tail_probability_problem,
    build_loss_distribution,
    quantum_var,
)


@pytest.fixture
def loss_dist():
    """Simple 2-qubit loss distribution for fast testing."""
    return normal_distribution(n_qubits=2, mean=0.0, std=0.02, n_sigma=3.0)


@pytest.fixture
def backend():
    return QiskitAerBackend()


class TestBuildLossDistribution:
    def test_from_returns(self) -> None:
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.02, 500)
        dist = build_loss_distribution(returns, n_qubits=3)
        assert dist.n_qubits == 3
        assert len(dist.probabilities) == 8
        assert np.isclose(dist.probabilities.sum(), 1.0)

    def test_values_range(self) -> None:
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0, 0.01, 200)
        dist = build_loss_distribution(returns, n_qubits=3)
        assert dist.low < dist.high
        assert len(dist.values) == 8


class TestTailProbabilityProblem:
    def test_builds_circuit(self, loss_dist) -> None:
        problem = _build_tail_probability_problem(loss_dist, threshold=0.0)
        assert problem.n_qubits == loss_dist.n_qubits + 1
        assert problem.objective_qubits == [loss_dist.n_qubits]

    def test_low_threshold_high_probability(self, loss_dist, backend) -> None:
        """Very low threshold -> high probability of exceeding."""
        problem = _build_tail_probability_problem(loss_dist, threshold=loss_dist.low - 1.0)
        # Build a measurable circuit
        from qiskit.circuit import ClassicalRegister

        qc = problem.state_preparation.copy()
        n = qc.num_qubits
        cr = ClassicalRegister(n, "meas")
        qc.add_register(cr)
        qc.measure(range(n), range(n))
        result = backend.run(qc, shots=1024)
        # Ancilla should be 1 for most measurements
        n_good = 0
        for bs, count in result.counts.items():
            if bs[0] == "1":  # MSB is the ancilla
                n_good += count
        assert n_good / result.shots > 0.8


class TestQuantumVaR:
    def test_basic_run(self, loss_dist, backend) -> None:
        config = QuantumVaRConfig(
            confidence_level=0.95,
            n_bisection_steps=2,
            qae_method="iqae",
            qae_epsilon=0.1,
            qae_shots=256,
            seed=42,
        )
        result = quantum_var(loss_dist, backend, config)
        assert result.n_qae_calls > 0
        assert len(result.bisection_history) == 2
        assert result.confidence_level == 0.95

    def test_var_in_distribution_range(self, loss_dist, backend) -> None:
        config = QuantumVaRConfig(
            confidence_level=0.95,
            n_bisection_steps=2,
            qae_method="iqae",
            qae_epsilon=0.1,
            qae_shots=256,
            seed=42,
        )
        result = quantum_var(loss_dist, backend, config)
        assert loss_dist.low <= result.var_estimate <= loss_dist.high

    def test_wall_time_recorded(self, loss_dist, backend) -> None:
        config = QuantumVaRConfig(
            n_bisection_steps=1,
            qae_epsilon=0.1,
            qae_shots=128,
        )
        result = quantum_var(loss_dist, backend, config)
        assert result.wall_time_s > 0
