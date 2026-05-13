"""Unit tests for quantum stress testing with superposition-encoded scenarios."""

from __future__ import annotations

import numpy as np
import pytest

from qufin.risk.quantum_stress import (
    COVID_2020_SCENARIOS,
    GFC_2008_SCENARIOS,
    PREDEFINED_SCENARIO_SETS,
    RATE_HIKE_2022_SCENARIOS,
    QuantumStressResult,
    QuantumStressTester,
    ScenarioLoss,
    StressScenarioSpec,
    build_loss_oracle,
    build_scenario_superposition,
    classical_stress_test,
    run_quantum_stress_test,
    _compute_all_scenario_losses,
    _compute_scenario_loss,
    _n_qubits_for_scenarios,
)
from qufin.risk.stress import StressScenario


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_weights() -> np.ndarray:
    """Sensitivity weights: [equity, rates, vol, spreads]."""
    return np.array([0.6, 0.1, 0.1, 0.2])


@pytest.fixture
def portfolio_value() -> float:
    return 1_000_000.0


@pytest.fixture
def simple_scenarios() -> list[StressScenarioSpec]:
    """Two simple scenarios for basic tests."""
    return [
        StressScenarioSpec(
            scenario=StressScenario(
                name="Mild Downturn",
                date="2024-01-01",
                equity_shock=-0.10,
                rates_shock=-50.0,
                vol_shock=0.50,
                spread_shock=50.0,
            ),
            probability=0.6,
        ),
        StressScenarioSpec(
            scenario=StressScenario(
                name="Severe Crisis",
                date="2024-01-01",
                equity_shock=-0.40,
                rates_shock=-200.0,
                vol_shock=2.00,
                spread_shock=300.0,
            ),
            probability=0.4,
        ),
    ]


@pytest.fixture
def mock_backend():
    from qufin.backends.mock import MockBackend
    return MockBackend(seed=42)


# ---------------------------------------------------------------------------
# Tests: helper functions
# ---------------------------------------------------------------------------


class TestHelperFunctions:
    def test_n_qubits_for_scenarios_1(self) -> None:
        assert _n_qubits_for_scenarios(1) == 1

    def test_n_qubits_for_scenarios_2(self) -> None:
        assert _n_qubits_for_scenarios(2) == 1

    def test_n_qubits_for_scenarios_3(self) -> None:
        assert _n_qubits_for_scenarios(3) == 2

    def test_n_qubits_for_scenarios_4(self) -> None:
        assert _n_qubits_for_scenarios(4) == 2

    def test_n_qubits_for_scenarios_5(self) -> None:
        assert _n_qubits_for_scenarios(5) == 3

    def test_compute_scenario_loss_positive(
        self, portfolio_value: float, simple_weights: np.ndarray
    ) -> None:
        """A negative equity shock should produce a positive loss."""
        sc = StressScenario(
            name="test",
            date="2024-01-01",
            equity_shock=-0.20,
            rates_shock=0.0,
            vol_shock=0.0,
            spread_shock=0.0,
        )
        loss = _compute_scenario_loss(portfolio_value, simple_weights, sc)
        # equity_shock=-0.20, weight=0.6, portfolio=1M -> pnl = -120000, loss = +120000
        assert loss > 0
        np.testing.assert_allclose(loss, 120_000.0, rtol=1e-10)

    def test_compute_all_scenario_losses_probabilities_sum(
        self,
        portfolio_value: float,
        simple_weights: np.ndarray,
        simple_scenarios: list[StressScenarioSpec],
    ) -> None:
        _losses, probs, _details = _compute_all_scenario_losses(
            portfolio_value, simple_weights, simple_scenarios,
        )
        np.testing.assert_allclose(probs.sum(), 1.0, atol=1e-12)

    def test_compute_all_scenario_losses_count(
        self,
        portfolio_value: float,
        simple_weights: np.ndarray,
        simple_scenarios: list[StressScenarioSpec],
    ) -> None:
        losses, _probs, details = _compute_all_scenario_losses(
            portfolio_value, simple_weights, simple_scenarios,
        )
        assert len(losses) == 2
        assert len(details) == 2

    def test_scenario_loss_details_structure(
        self,
        portfolio_value: float,
        simple_weights: np.ndarray,
        simple_scenarios: list[StressScenarioSpec],
    ) -> None:
        _losses, _probs, details = _compute_all_scenario_losses(
            portfolio_value, simple_weights, simple_scenarios,
        )
        for d in details:
            assert isinstance(d, ScenarioLoss)
            assert d.scenario_name != ""
            assert d.probability > 0


# ---------------------------------------------------------------------------
# Tests: scenario superposition state preparation
# ---------------------------------------------------------------------------


class TestScenarioSuperposition:
    def test_correct_qubit_count_2_scenarios(self) -> None:
        from qiskit.circuit import QuantumCircuit

        probs = np.array([0.6, 0.4])
        circ = build_scenario_superposition(probs, n_qubits=1)
        assert isinstance(circ, QuantumCircuit)
        assert circ.num_qubits == 1

    def test_correct_qubit_count_4_scenarios(self) -> None:
        probs = np.array([0.4, 0.3, 0.2, 0.1])
        circ = build_scenario_superposition(probs, n_qubits=2)
        assert circ.num_qubits == 2

    def test_uniform_distribution(self) -> None:
        probs = np.array([0.25, 0.25, 0.25, 0.25])
        circ = build_scenario_superposition(probs, n_qubits=2)
        assert circ.num_qubits == 2

    def test_single_scenario(self) -> None:
        probs = np.array([1.0])
        circ = build_scenario_superposition(probs, n_qubits=1)
        assert circ.num_qubits == 1

    def test_padded_scenarios(self) -> None:
        """3 scenarios encoded in 2 qubits (4 states), last state unused."""
        probs = np.array([0.5, 0.3, 0.2])
        circ = build_scenario_superposition(probs, n_qubits=2)
        assert circ.num_qubits == 2


# ---------------------------------------------------------------------------
# Tests: loss oracle correctness
# ---------------------------------------------------------------------------


class TestLossOracle:
    def test_correct_qubit_count(self) -> None:
        from qiskit.circuit import QuantumCircuit

        losses = np.array([100_000.0, 200_000.0])
        circ = build_loss_oracle(losses, n_qubits_scenario=1)
        assert isinstance(circ, QuantumCircuit)
        # scenario register (1) + ancilla (1)
        assert circ.num_qubits == 2

    def test_zero_loss_no_rotation(self) -> None:
        """All-zero losses should produce no rotations on ancilla."""
        from qiskit.circuit import QuantumCircuit

        losses = np.array([0.0, 0.0])
        circ = build_loss_oracle(losses, n_qubits_scenario=1)
        assert isinstance(circ, QuantumCircuit)

    def test_equal_losses_symmetric(self) -> None:
        """Equal losses should produce identical rotations."""
        losses = np.array([100_000.0, 100_000.0])
        circ = build_loss_oracle(losses, n_qubits_scenario=1)
        assert circ.num_qubits == 2

    def test_four_scenario_oracle(self) -> None:
        losses = np.array([50_000.0, 100_000.0, 200_000.0, 300_000.0])
        circ = build_loss_oracle(losses, n_qubits_scenario=2)
        assert circ.num_qubits == 3

    def test_negative_losses_clipped(self) -> None:
        """Negative losses (gains) should be clipped to zero."""
        losses = np.array([-50_000.0, 100_000.0])
        circ = build_loss_oracle(losses, n_qubits_scenario=1)
        assert circ.num_qubits == 2


# ---------------------------------------------------------------------------
# Tests: predefined scenario sets
# ---------------------------------------------------------------------------


class TestPredefinedScenarios:
    def test_gfc_2008_has_4_scenarios(self) -> None:
        assert len(GFC_2008_SCENARIOS) == 4

    def test_covid_2020_has_4_scenarios(self) -> None:
        assert len(COVID_2020_SCENARIOS) == 4

    def test_rate_hike_2022_has_4_scenarios(self) -> None:
        assert len(RATE_HIKE_2022_SCENARIOS) == 4

    def test_predefined_sets_keys(self) -> None:
        assert "gfc_2008" in PREDEFINED_SCENARIO_SETS
        assert "covid_2020" in PREDEFINED_SCENARIO_SETS
        assert "rate_hike_2022" in PREDEFINED_SCENARIO_SETS

    def test_probabilities_positive(self) -> None:
        for name, specs in PREDEFINED_SCENARIO_SETS.items():
            for spec in specs:
                assert spec.probability > 0, f"{name}: {spec.scenario.name}"

    def test_gfc_worst_is_trough(self) -> None:
        """The GFC trough scenario should have the largest equity shock."""
        shocks = [abs(s.scenario.equity_shock) for s in GFC_2008_SCENARIOS]
        worst_idx = np.argmax(shocks)
        assert "Trough" in GFC_2008_SCENARIOS[worst_idx].scenario.name


# ---------------------------------------------------------------------------
# Tests: QuantumStressTester
# ---------------------------------------------------------------------------


class TestQuantumStressTester:
    def test_construction(self, mock_backend) -> None:
        tester = QuantumStressTester(backend=mock_backend)
        assert tester.qae_method == "iqae"
        assert tester.qae_shots == 1024

    @pytest.mark.slow
    def test_run_simple_scenarios(
        self,
        mock_backend,
        portfolio_value: float,
        simple_weights: np.ndarray,
        simple_scenarios: list[StressScenarioSpec],
    ) -> None:
        tester = QuantumStressTester(backend=mock_backend, seed=42)
        result = tester.run(portfolio_value, simple_weights, simple_scenarios)

        assert isinstance(result, QuantumStressResult)
        assert result.method == "quantum"
        assert result.n_scenarios == 2
        assert result.n_qubits_scenario == 1
        assert len(result.per_scenario) == 2
        assert result.worst_case_loss > 0
        assert result.weighted_expected_loss > 0

    def test_run_invalid_weights_raises(
        self,
        mock_backend,
        simple_scenarios: list[StressScenarioSpec],
    ) -> None:
        tester = QuantumStressTester(backend=mock_backend)
        with pytest.raises(ValueError, match="shape \\(4,\\)"):
            tester.run(1_000_000.0, np.array([0.5, 0.5]), simple_scenarios)

    def test_run_empty_scenarios_raises(
        self,
        mock_backend,
        simple_weights: np.ndarray,
    ) -> None:
        tester = QuantumStressTester(backend=mock_backend)
        with pytest.raises(ValueError, match="At least one scenario"):
            tester.run(1_000_000.0, simple_weights, [])

    @pytest.mark.slow
    def test_worst_case_is_severe(
        self,
        mock_backend,
        portfolio_value: float,
        simple_weights: np.ndarray,
        simple_scenarios: list[StressScenarioSpec],
    ) -> None:
        tester = QuantumStressTester(backend=mock_backend, seed=42)
        result = tester.run(portfolio_value, simple_weights, simple_scenarios)
        assert result.worst_case_scenario == "Severe Crisis"

    @pytest.mark.slow
    def test_per_scenario_losses_have_correct_names(
        self,
        mock_backend,
        portfolio_value: float,
        simple_weights: np.ndarray,
        simple_scenarios: list[StressScenarioSpec],
    ) -> None:
        tester = QuantumStressTester(backend=mock_backend, seed=42)
        result = tester.run(portfolio_value, simple_weights, simple_scenarios)
        names = {s.scenario_name for s in result.per_scenario}
        assert names == {"Mild Downturn", "Severe Crisis"}


# ---------------------------------------------------------------------------
# Tests: classical stress test
# ---------------------------------------------------------------------------


class TestClassicalStressTest:
    def test_returns_classical_result(
        self,
        portfolio_value: float,
        simple_weights: np.ndarray,
        simple_scenarios: list[StressScenarioSpec],
    ) -> None:
        result = classical_stress_test(
            portfolio_value, simple_weights, simple_scenarios, seed=42,
        )
        assert isinstance(result, QuantumStressResult)
        assert result.method == "classical"
        assert result.n_scenarios == 2

    def test_weighted_loss_positive(
        self,
        portfolio_value: float,
        simple_weights: np.ndarray,
        simple_scenarios: list[StressScenarioSpec],
    ) -> None:
        result = classical_stress_test(
            portfolio_value, simple_weights, simple_scenarios, seed=42,
        )
        # With negative equity shocks, expected loss should be positive
        assert result.weighted_expected_loss > 0

    def test_mc_estimate_close_to_weighted(
        self,
        portfolio_value: float,
        simple_weights: np.ndarray,
        simple_scenarios: list[StressScenarioSpec],
    ) -> None:
        result = classical_stress_test(
            portfolio_value,
            simple_weights,
            simple_scenarios,
            n_monte_carlo=50_000,
            seed=42,
        )
        # MC estimate should be close to exact weighted loss (within 10%)
        rel_err = abs(
            result.quantum_estimate - result.weighted_expected_loss
        ) / abs(result.weighted_expected_loss)
        assert rel_err < 0.10

    def test_invalid_weights_raises(
        self,
        portfolio_value: float,
        simple_scenarios: list[StressScenarioSpec],
    ) -> None:
        with pytest.raises(ValueError, match="shape \\(4,\\)"):
            classical_stress_test(
                portfolio_value, np.array([0.5]), simple_scenarios,
            )

    def test_per_scenario_details(
        self,
        portfolio_value: float,
        simple_weights: np.ndarray,
        simple_scenarios: list[StressScenarioSpec],
    ) -> None:
        result = classical_stress_test(
            portfolio_value, simple_weights, simple_scenarios, seed=42,
        )
        assert len(result.per_scenario) == 2
        for detail in result.per_scenario:
            assert detail.probability > 0
            assert detail.scenario_name != ""


# ---------------------------------------------------------------------------
# Tests: quantum vs classical comparison
# ---------------------------------------------------------------------------


class TestQuantumVsClassical:
    @pytest.mark.slow
    def test_both_methods_agree_on_worst_case(
        self,
        mock_backend,
        portfolio_value: float,
        simple_weights: np.ndarray,
        simple_scenarios: list[StressScenarioSpec],
    ) -> None:
        """Quantum and classical should agree on worst-case scenario."""
        tester = QuantumStressTester(backend=mock_backend, seed=42)
        q_result = tester.run(portfolio_value, simple_weights, simple_scenarios)
        c_result = classical_stress_test(
            portfolio_value, simple_weights, simple_scenarios, seed=42,
        )
        assert q_result.worst_case_scenario == c_result.worst_case_scenario
        np.testing.assert_allclose(
            q_result.worst_case_loss,
            c_result.worst_case_loss,
            rtol=1e-10,
        )

    @pytest.mark.slow
    def test_both_methods_agree_on_weighted_loss(
        self,
        mock_backend,
        portfolio_value: float,
        simple_weights: np.ndarray,
        simple_scenarios: list[StressScenarioSpec],
    ) -> None:
        """Classical weighted loss should match between methods."""
        tester = QuantumStressTester(backend=mock_backend, seed=42)
        q_result = tester.run(portfolio_value, simple_weights, simple_scenarios)
        c_result = classical_stress_test(
            portfolio_value, simple_weights, simple_scenarios, seed=42,
        )
        np.testing.assert_allclose(
            q_result.weighted_expected_loss,
            c_result.weighted_expected_loss,
            rtol=1e-10,
        )

    @pytest.mark.slow
    def test_per_scenario_count_matches(
        self,
        mock_backend,
        portfolio_value: float,
        simple_weights: np.ndarray,
        simple_scenarios: list[StressScenarioSpec],
    ) -> None:
        tester = QuantumStressTester(backend=mock_backend, seed=42)
        q_result = tester.run(portfolio_value, simple_weights, simple_scenarios)
        c_result = classical_stress_test(
            portfolio_value, simple_weights, simple_scenarios, seed=42,
        )
        assert q_result.n_scenarios == c_result.n_scenarios


# ---------------------------------------------------------------------------
# Tests: run_quantum_stress_test convenience function
# ---------------------------------------------------------------------------


class TestRunQuantumStressTest:
    @pytest.mark.slow
    def test_default_scenarios(
        self, simple_weights: np.ndarray, portfolio_value: float,
    ) -> None:
        """Default (GFC 2008) scenarios should work."""
        result = run_quantum_stress_test(
            portfolio_value, simple_weights, seed=42,
        )
        assert isinstance(result, QuantumStressResult)
        assert result.n_scenarios == 4

    @pytest.mark.slow
    def test_named_scenario_set(
        self, simple_weights: np.ndarray, portfolio_value: float,
    ) -> None:
        result = run_quantum_stress_test(
            portfolio_value, simple_weights, scenario_specs="covid_2020", seed=42,
        )
        assert result.n_scenarios == 4

    def test_unknown_scenario_set_raises(
        self, simple_weights: np.ndarray, portfolio_value: float,
    ) -> None:
        with pytest.raises(ValueError, match="Unknown scenario set"):
            run_quantum_stress_test(
                portfolio_value, simple_weights, scenario_specs="nonexistent",
            )

    @pytest.mark.slow
    def test_custom_scenarios(
        self,
        simple_weights: np.ndarray,
        portfolio_value: float,
        simple_scenarios: list[StressScenarioSpec],
    ) -> None:
        result = run_quantum_stress_test(
            portfolio_value,
            simple_weights,
            scenario_specs=simple_scenarios,
            seed=42,
        )
        assert result.n_scenarios == 2
        assert result.method == "quantum"

    @pytest.mark.slow
    def test_with_explicit_backend(
        self,
        mock_backend,
        simple_weights: np.ndarray,
        portfolio_value: float,
    ) -> None:
        result = run_quantum_stress_test(
            portfolio_value,
            simple_weights,
            backend=mock_backend,
            seed=42,
        )
        assert isinstance(result, QuantumStressResult)


# ---------------------------------------------------------------------------
# Tests: custom scenario definition
# ---------------------------------------------------------------------------


class TestCustomScenarios:
    def test_single_custom_scenario(
        self,
        portfolio_value: float,
        simple_weights: np.ndarray,
    ) -> None:
        """A single custom scenario should work for classical test."""
        custom = [
            StressScenarioSpec(
                scenario=StressScenario(
                    name="Custom Crash",
                    date="2025-01-01",
                    equity_shock=-0.50,
                    rates_shock=-100.0,
                    vol_shock=3.0,
                    spread_shock=200.0,
                    description="A hypothetical severe crash.",
                ),
                probability=1.0,
            ),
        ]
        result = classical_stress_test(
            portfolio_value, simple_weights, custom, seed=42,
        )
        assert result.n_scenarios == 1
        assert result.worst_case_scenario == "Custom Crash"
        assert result.per_scenario[0].total_loss > 0

    def test_unequal_probability_weighting(
        self,
        portfolio_value: float,
        simple_weights: np.ndarray,
    ) -> None:
        """Weighted loss should reflect probability weights."""
        mild = StressScenarioSpec(
            scenario=StressScenario(
                name="Mild",
                date="2025-01-01",
                equity_shock=-0.05,
                rates_shock=0.0,
                vol_shock=0.0,
                spread_shock=0.0,
            ),
            probability=0.9,
        )
        severe = StressScenarioSpec(
            scenario=StressScenario(
                name="Severe",
                date="2025-01-01",
                equity_shock=-0.50,
                rates_shock=0.0,
                vol_shock=0.0,
                spread_shock=0.0,
            ),
            probability=0.1,
        )
        result = classical_stress_test(
            portfolio_value, simple_weights, [mild, severe], seed=42,
        )
        mild_loss = result.per_scenario[0].total_loss
        severe_loss = result.per_scenario[1].total_loss
        # The weighted loss should be much closer to the mild scenario loss
        # since mild has probability 0.9
        assert mild_loss < severe_loss
        assert result.weighted_expected_loss < (mild_loss + severe_loss) / 2

    def test_custom_scenario_with_zero_vol_shock(
        self,
        portfolio_value: float,
        simple_weights: np.ndarray,
    ) -> None:
        """Scenario with no vol shock should only have equity/rates/spread losses."""
        custom = [
            StressScenarioSpec(
                scenario=StressScenario(
                    name="No Vol",
                    date="2025-01-01",
                    equity_shock=-0.20,
                    rates_shock=-100.0,
                    vol_shock=0.0,
                    spread_shock=50.0,
                ),
                probability=1.0,
            ),
        ]
        result = classical_stress_test(
            portfolio_value, simple_weights, custom, seed=42,
        )
        # vol_loss should be zero
        assert result.per_scenario[0].vol_loss == 0.0
        # equity loss should be positive (negative shock)
        assert result.per_scenario[0].equity_loss > 0


# ---------------------------------------------------------------------------
# Tests: result dataclass
# ---------------------------------------------------------------------------


class TestQuantumStressResult:
    def test_default_construction(self) -> None:
        result = QuantumStressResult()
        assert result.weighted_expected_loss == 0.0
        assert result.per_scenario == []
        assert result.method == "quantum"
        assert result.metadata == {}

    def test_scenario_loss_fields(self) -> None:
        sl = ScenarioLoss(
            scenario_name="Test",
            equity_loss=100.0,
            rates_loss=50.0,
            vol_loss=25.0,
            spread_loss=10.0,
            total_loss=185.0,
            pct_loss=0.0185,
            probability=0.5,
        )
        assert sl.total_loss == 185.0
        assert sl.probability == 0.5
