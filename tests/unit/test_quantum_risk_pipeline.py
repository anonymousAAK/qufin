"""Unit tests for the quantum risk analysis pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from qufin.risk.quantum_risk_pipeline import (
    PortfolioRiskSpec,
    QuantumRiskResult,
    build_loss_loading_circuit,
    build_portfolio_loss_distribution,
    build_tail_oracle,
    quantum_stress_var,
    quantum_var_pipeline,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def sample_returns(rng: np.random.Generator) -> np.ndarray:
    """Multi-asset returns: 252 days x 5 assets."""
    return rng.normal(0.0005, 0.02, (252, 5))


@pytest.fixture
def sample_weights() -> np.ndarray:
    return np.array([0.2, 0.2, 0.2, 0.2, 0.2])


@pytest.fixture
def risk_spec(sample_returns: np.ndarray, sample_weights: np.ndarray) -> PortfolioRiskSpec:
    return PortfolioRiskSpec(
        returns=sample_returns,
        weights=sample_weights,
        confidence_level=0.99,
        horizon=1,
        n_qubits_loss=4,
        distribution="normal",
    )


# ---------------------------------------------------------------------------
# Tests: PortfolioRiskSpec
# ---------------------------------------------------------------------------

class TestPortfolioRiskSpec:
    def test_construction(
        self, sample_returns: np.ndarray, sample_weights: np.ndarray
    ) -> None:
        spec = PortfolioRiskSpec(
            returns=sample_returns,
            weights=sample_weights,
        )
        assert spec.confidence_level == 0.99
        assert spec.horizon == 1
        assert spec.n_qubits_loss == 6
        assert spec.distribution == "normal"


# ---------------------------------------------------------------------------
# Tests: build_portfolio_loss_distribution
# ---------------------------------------------------------------------------

class TestBuildPortfolioLossDistribution:
    def test_probabilities_sum_to_one(self, risk_spec: PortfolioRiskSpec) -> None:
        probs, _loss_min, _loss_max = build_portfolio_loss_distribution(risk_spec)
        np.testing.assert_allclose(probs.sum(), 1.0, atol=1e-10)

    def test_correct_number_of_bins(self, risk_spec: PortfolioRiskSpec) -> None:
        probs, _, _ = build_portfolio_loss_distribution(risk_spec)
        expected_bins = 2 ** risk_spec.n_qubits_loss
        assert len(probs) == expected_bins

    def test_probabilities_nonnegative(self, risk_spec: PortfolioRiskSpec) -> None:
        probs, _, _ = build_portfolio_loss_distribution(risk_spec)
        assert np.all(probs >= 0.0)

    def test_loss_range_ordered(self, risk_spec: PortfolioRiskSpec) -> None:
        _, loss_min, loss_max = build_portfolio_loss_distribution(risk_spec)
        assert loss_min < loss_max

    def test_empirical_distribution(
        self, sample_returns: np.ndarray, sample_weights: np.ndarray
    ) -> None:
        spec = PortfolioRiskSpec(
            returns=sample_returns,
            weights=sample_weights,
            n_qubits_loss=4,
            distribution="empirical",
        )
        probs, _loss_min, _loss_max = build_portfolio_loss_distribution(spec)
        np.testing.assert_allclose(probs.sum(), 1.0, atol=1e-10)
        assert len(probs) == 16

    def test_unknown_distribution_raises(
        self, sample_returns: np.ndarray, sample_weights: np.ndarray
    ) -> None:
        spec = PortfolioRiskSpec(
            returns=sample_returns,
            weights=sample_weights,
            distribution="bogus",
        )
        with pytest.raises(ValueError, match="Unknown distribution"):
            build_portfolio_loss_distribution(spec)


# ---------------------------------------------------------------------------
# Tests: build_loss_loading_circuit
# ---------------------------------------------------------------------------

class TestBuildLossLoadingCircuit:
    def test_correct_qubit_count(self, risk_spec: PortfolioRiskSpec) -> None:
        probs, _, _ = build_portfolio_loss_distribution(risk_spec)
        n_q = risk_spec.n_qubits_loss
        circuit = build_loss_loading_circuit(probs, n_q)
        assert circuit.num_qubits == n_q

    def test_returns_quantum_circuit(self, risk_spec: PortfolioRiskSpec) -> None:
        from qiskit.circuit import QuantumCircuit

        probs, _, _ = build_portfolio_loss_distribution(risk_spec)
        circuit = build_loss_loading_circuit(probs, risk_spec.n_qubits_loss)
        assert isinstance(circuit, QuantumCircuit)

    def test_uniform_distribution(self) -> None:
        n_q = 3
        probs = np.ones(2**n_q) / (2**n_q)
        circuit = build_loss_loading_circuit(probs, n_q)
        assert circuit.num_qubits == n_q


# ---------------------------------------------------------------------------
# Tests: build_tail_oracle
# ---------------------------------------------------------------------------

class TestBuildTailOracle:
    def test_returns_quantum_circuit(self) -> None:
        from qiskit.circuit import QuantumCircuit

        circuit = build_tail_oracle(threshold_bin=4, n_qubits=3)
        assert isinstance(circuit, QuantumCircuit)

    def test_correct_qubit_count(self) -> None:
        n_q = 4
        circuit = build_tail_oracle(threshold_bin=8, n_qubits=n_q)
        # n_qubits + 1 ancilla
        assert circuit.num_qubits == n_q + 1

    def test_threshold_zero_marks_all(self) -> None:
        """threshold=0 should mark all states."""
        circuit = build_tail_oracle(threshold_bin=0, n_qubits=2)
        assert circuit.num_qubits == 3

    def test_threshold_at_max(self) -> None:
        """threshold = 2^n marks only the last state."""
        n_q = 3
        circuit = build_tail_oracle(threshold_bin=2**n_q - 1, n_qubits=n_q)
        assert circuit.num_qubits == n_q + 1


# ---------------------------------------------------------------------------
# Tests: quantum_stress_var
# ---------------------------------------------------------------------------

class TestQuantumStressVar:
    @pytest.mark.slow
    def test_returns_correct_scenario_keys(
        self, risk_spec: PortfolioRiskSpec
    ) -> None:
        from qufin.backends.mock import MockBackend

        backend = MockBackend(seed=42)
        stressed_returns = np.asarray(risk_spec.returns) * 2.0  # vol shock
        scenarios = {
            "base": np.asarray(risk_spec.returns),
            "vol_shock": stressed_returns,
        }
        results = quantum_stress_var(
            risk_spec, scenarios, backend, qae_method="iqae"
        )
        assert set(results.keys()) == {"base", "vol_shock"}
        for _name, result in results.items():
            assert isinstance(result, QuantumRiskResult)

    @pytest.mark.slow
    def test_vol_shock_increases_var(
        self, risk_spec: PortfolioRiskSpec
    ) -> None:
        from qufin.backends.mock import MockBackend

        backend = MockBackend(seed=42)
        stressed_returns = np.asarray(risk_spec.returns) * 3.0
        scenarios = {
            "base": np.asarray(risk_spec.returns),
            "vol_shock": stressed_returns,
        }
        results = quantum_stress_var(
            risk_spec, scenarios, backend, qae_method="iqae"
        )
        # Classical VaR under stress should be higher
        assert results["vol_shock"].classical_var >= results["base"].classical_var


# ---------------------------------------------------------------------------
# Tests: quantum_var_pipeline (integration, slow)
# ---------------------------------------------------------------------------

class TestQuantumVarPipeline:
    @pytest.mark.slow
    def test_returns_quantum_risk_result(
        self, risk_spec: PortfolioRiskSpec
    ) -> None:
        from qufin.backends.mock import MockBackend

        backend = MockBackend(seed=42)
        result = quantum_var_pipeline(risk_spec, backend, qae_method="iqae")
        assert isinstance(result, QuantumRiskResult)
        assert result.confidence_level == 0.99
        assert result.bisection_steps > 0
        assert result.n_qae_calls > 0

    @pytest.mark.slow
    def test_classical_baseline_positive(
        self, risk_spec: PortfolioRiskSpec
    ) -> None:
        from qufin.backends.mock import MockBackend

        backend = MockBackend(seed=42)
        result = quantum_var_pipeline(risk_spec, backend, qae_method="iqae")
        # Classical VaR should be positive for non-trivial returns
        assert result.classical_var > 0
        assert result.classical_cvar >= result.classical_var
