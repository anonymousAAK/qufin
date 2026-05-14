"""Quantum stress testing with superposition-encoded scenarios.

Encodes multiple stress scenarios as a superposition state where each
computational basis state represents a scenario weighted by its
probability. A portfolio loss oracle evaluates losses under each
scenario, and quantum amplitude estimation (QAE) computes the
probability-weighted expected loss across all scenarios simultaneously.

This provides a quadratic speed-up over classical scenario-weighted
Monte Carlo for large scenario sets, following the paradigm of
Woerner & Egger (1806.06893) applied to stress testing.

References
----------
Woerner & Egger, "Quantum Risk Analysis", npj Quantum Information 5:15 (2019).
Egger et al., "Credit Risk Analysis using Quantum Computers", IEEE TQE (2021).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from qufin.backends.base import Backend
from qufin.options.amplitude_estimation.estimation_problem import EstimationProblem
from qufin.risk.stress import StressScenario

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StressScenarioSpec:
    """A stress scenario with probability weight for quantum encoding.

    Parameters
    ----------
    scenario : StressScenario
        The underlying stress scenario (shocks to risk factors).
    probability : float
        Prior probability weight for this scenario (must be > 0).
        Weights are normalised before encoding.
    """

    scenario: StressScenario
    probability: float = 1.0


@dataclass
class ScenarioLoss:
    """Loss result for a single stress scenario."""

    scenario_name: str = ""
    equity_loss: float = 0.0
    rates_loss: float = 0.0
    vol_loss: float = 0.0
    spread_loss: float = 0.0
    total_loss: float = 0.0
    pct_loss: float = 0.0
    probability: float = 0.0


@dataclass
class QuantumStressResult:
    """Result from quantum or classical stress testing.

    Attributes
    ----------
    per_scenario : list[ScenarioLoss]
        Loss breakdown for each scenario.
    weighted_expected_loss : float
        Probability-weighted expected total loss across all scenarios.
    worst_case_loss : float
        Maximum loss across all scenarios.
    worst_case_scenario : str
        Name of the worst-case scenario.
    quantum_estimate : float
        QAE estimate of the weighted expected loss (0.0 for classical).
    method : str
        ``"quantum"`` or ``"classical"``.
    n_scenarios : int
        Number of scenarios evaluated.
    n_qubits_scenario : int
        Qubits used for scenario register.
    wall_time_s : float
        Wall-clock time in seconds.
    metadata : dict
        Extra information (circuit depth, QAE calls, etc.).
    """

    per_scenario: list[ScenarioLoss] = field(default_factory=list)
    weighted_expected_loss: float = 0.0
    worst_case_loss: float = 0.0
    worst_case_scenario: str = ""
    quantum_estimate: float = 0.0
    method: str = "quantum"
    n_scenarios: int = 0
    n_qubits_scenario: int = 0
    wall_time_s: float = 0.0
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Pre-defined scenario sets
# ---------------------------------------------------------------------------

GFC_2008_SCENARIOS: list[StressScenarioSpec] = [
    StressScenarioSpec(
        scenario=StressScenario(
            name="GFC Peak (Sep 2008)",
            date="2008-09-15",
            equity_shock=-0.38,
            rates_shock=-200.0,
            vol_shock=2.00,
            spread_shock=300.0,
            description="Lehman collapse, peak market stress.",
        ),
        probability=0.4,
    ),
    StressScenarioSpec(
        scenario=StressScenario(
            name="GFC Trough (Mar 2009)",
            date="2009-03-09",
            equity_shock=-0.57,
            rates_shock=-300.0,
            vol_shock=3.00,
            spread_shock=500.0,
            description="Market bottom, maximum drawdown.",
        ),
        probability=0.3,
    ),
    StressScenarioSpec(
        scenario=StressScenario(
            name="GFC Recovery (Jun 2009)",
            date="2009-06-30",
            equity_shock=-0.25,
            rates_shock=-150.0,
            vol_shock=1.00,
            spread_shock=150.0,
            description="Early recovery phase with lingering stress.",
        ),
        probability=0.2,
    ),
    StressScenarioSpec(
        scenario=StressScenario(
            name="GFC Mild (Pre-Lehman)",
            date="2008-07-01",
            equity_shock=-0.15,
            rates_shock=-75.0,
            vol_shock=0.50,
            spread_shock=100.0,
            description="Pre-Lehman stress, Bear Stearns aftermath.",
        ),
        probability=0.1,
    ),
]

COVID_2020_SCENARIOS: list[StressScenarioSpec] = [
    StressScenarioSpec(
        scenario=StressScenario(
            name="COVID Crash (Mar 2020)",
            date="2020-03-16",
            equity_shock=-0.34,
            rates_shock=-150.0,
            vol_shock=4.00,
            spread_shock=200.0,
            description="Initial pandemic crash, VIX at 82.",
        ),
        probability=0.4,
    ),
    StressScenarioSpec(
        scenario=StressScenario(
            name="COVID Trough (Mar 23 2020)",
            date="2020-03-23",
            equity_shock=-0.30,
            rates_shock=-175.0,
            vol_shock=3.50,
            spread_shock=250.0,
            description="Market trough before Fed intervention.",
        ),
        probability=0.3,
    ),
    StressScenarioSpec(
        scenario=StressScenario(
            name="COVID Recovery (May 2020)",
            date="2020-05-15",
            equity_shock=-0.10,
            rates_shock=-100.0,
            vol_shock=1.50,
            spread_shock=80.0,
            description="Early recovery after Fed backstop.",
        ),
        probability=0.2,
    ),
    StressScenarioSpec(
        scenario=StressScenario(
            name="COVID Mild (Feb 2020)",
            date="2020-02-24",
            equity_shock=-0.12,
            rates_shock=-50.0,
            vol_shock=1.00,
            spread_shock=40.0,
            description="Early sell-off before full crash.",
        ),
        probability=0.1,
    ),
]

RATE_HIKE_2022_SCENARIOS: list[StressScenarioSpec] = [
    StressScenarioSpec(
        scenario=StressScenario(
            name="Rate Hike Peak (Jun 2022)",
            date="2022-06-13",
            equity_shock=-0.23,
            rates_shock=300.0,
            vol_shock=0.60,
            spread_shock=100.0,
            description="Peak rate-hike impact, bear market entry.",
        ),
        probability=0.4,
    ),
    StressScenarioSpec(
        scenario=StressScenario(
            name="Rate Hike H2 (Sep 2022)",
            date="2022-09-30",
            equity_shock=-0.25,
            rates_shock=350.0,
            vol_shock=0.55,
            spread_shock=120.0,
            description="Continued tightening, UK gilt crisis.",
        ),
        probability=0.3,
    ),
    StressScenarioSpec(
        scenario=StressScenario(
            name="Rate Hike Moderate (Mar 2022)",
            date="2022-03-16",
            equity_shock=-0.12,
            rates_shock=150.0,
            vol_shock=0.30,
            spread_shock=50.0,
            description="First rate hike, moderate market reaction.",
        ),
        probability=0.2,
    ),
    StressScenarioSpec(
        scenario=StressScenario(
            name="Rate Hike Mild (Jan 2022)",
            date="2022-01-26",
            equity_shock=-0.08,
            rates_shock=75.0,
            vol_shock=0.20,
            spread_shock=30.0,
            description="Anticipation phase before hikes began.",
        ),
        probability=0.1,
    ),
]

PREDEFINED_SCENARIO_SETS: dict[str, list[StressScenarioSpec]] = {
    "gfc_2008": GFC_2008_SCENARIOS,
    "covid_2020": COVID_2020_SCENARIOS,
    "rate_hike_2022": RATE_HIKE_2022_SCENARIOS,
}


# ---------------------------------------------------------------------------
# Loss computation helpers
# ---------------------------------------------------------------------------


def _compute_scenario_loss(
    portfolio_value: float,
    weights: NDArray[np.float64],
    scenario: StressScenario,
) -> float:
    """Compute total portfolio loss under a single stress scenario.

    Parameters
    ----------
    portfolio_value : float
        Total portfolio market value.
    weights : NDArray, shape (4,)
        Sensitivity weights [equity, rates, vol, spreads].
    scenario : StressScenario
        Stress scenario with factor shocks.

    Returns
    -------
    float
        Total loss (positive means money lost).
    """
    w = np.asarray(weights, dtype=np.float64)
    shocks = np.array([
        scenario.equity_shock,
        scenario.rates_shock / 10_000,
        scenario.vol_shock,
        -scenario.spread_shock / 10_000,
    ])
    pnl = portfolio_value * w * shocks
    # Loss is negative of P&L
    return float(-np.sum(pnl))


def _compute_all_scenario_losses(
    portfolio_value: float,
    weights: NDArray[np.float64],
    scenario_specs: list[StressScenarioSpec],
) -> tuple[NDArray[np.float64], NDArray[np.float64], list[ScenarioLoss]]:
    """Compute losses and normalised probabilities for all scenarios.

    Returns
    -------
    losses : NDArray, shape (n_scenarios,)
        Total loss for each scenario.
    probs : NDArray, shape (n_scenarios,)
        Normalised probability weights.
    details : list[ScenarioLoss]
        Per-scenario breakdown.
    """
    w = np.asarray(weights, dtype=np.float64)
    n = len(scenario_specs)

    raw_probs = np.array([s.probability for s in scenario_specs], dtype=np.float64)
    prob_sum = raw_probs.sum()
    if prob_sum > 0:
        probs = raw_probs / prob_sum
    else:
        probs = np.ones(n) / n

    losses = np.zeros(n, dtype=np.float64)
    details: list[ScenarioLoss] = []

    for i, spec in enumerate(scenario_specs):
        sc = spec.scenario
        shocks = np.array([
            sc.equity_shock,
            sc.rates_shock / 10_000,
            sc.vol_shock,
            -sc.spread_shock / 10_000,
        ])
        factor_pnl = portfolio_value * w * shocks
        total_pnl = float(np.sum(factor_pnl))
        total_loss = -total_pnl
        losses[i] = total_loss

        details.append(ScenarioLoss(
            scenario_name=sc.name,
            equity_loss=float(-factor_pnl[0]),
            rates_loss=float(-factor_pnl[1]),
            vol_loss=float(-factor_pnl[2]),
            spread_loss=float(-factor_pnl[3]),
            total_loss=total_loss,
            pct_loss=total_loss / portfolio_value if portfolio_value != 0 else 0.0,
            probability=float(probs[i]),
        ))

    return losses, probs, details


# ---------------------------------------------------------------------------
# Quantum circuit builders
# ---------------------------------------------------------------------------


def _next_power_of_two(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _n_qubits_for_scenarios(n_scenarios: int) -> int:
    """Number of qubits needed to encode n_scenarios in superposition."""
    if n_scenarios <= 1:
        return 1
    return (n_scenarios - 1).bit_length()


def build_scenario_superposition(
    probs: NDArray[np.float64],
    n_qubits: int,
) -> object:
    """Build a circuit that prepares a weighted superposition of scenarios.

    Creates |psi> = sum_i sqrt(p_i) |i> for i in [0, n_scenarios).
    Unused basis states (if n_scenarios < 2^n_qubits) get zero amplitude.

    Parameters
    ----------
    probs : NDArray, shape (n_scenarios,)
        Normalised scenario probabilities.
    n_qubits : int
        Number of qubits for the scenario register.

    Returns
    -------
    QuantumCircuit
        State preparation circuit.
    """
    from qiskit import transpile
    from qiskit.circuit import QuantumCircuit
    from qiskit.circuit.library import StatePreparation

    n_states = 2 ** n_qubits
    amplitudes = np.zeros(n_states, dtype=np.float64)

    n_scenarios = min(len(probs), n_states)
    for i in range(n_scenarios):
        amplitudes[i] = np.sqrt(max(probs[i], 0.0))

    # Normalise
    norm = np.linalg.norm(amplitudes)
    if norm > 1e-16:
        amplitudes = amplitudes / norm

    qc = QuantumCircuit(n_qubits)
    qc.append(StatePreparation(amplitudes), range(n_qubits))
    # Decompose to basic gates for Aer compatibility
    return transpile(qc, basis_gates=["cx", "u3", "id"], optimization_level=0)


def build_loss_oracle(
    losses: NDArray[np.float64],
    n_qubits_scenario: int,
) -> object:
    """Build a loss oracle that marks the ancilla proportional to loss.

    For each scenario |i>, rotates an ancilla qubit by an angle
    proportional to the normalised loss value, so that the amplitude
    of the ancilla being |1> encodes the expected loss.

    Parameters
    ----------
    losses : NDArray, shape (n_scenarios,)
        Loss values for each scenario (non-negative).
    n_qubits_scenario : int
        Number of qubits in the scenario register.

    Returns
    -------
    QuantumCircuit
        Oracle circuit on n_qubits_scenario + 1 qubits.
    """
    from qiskit.circuit import QuantumCircuit

    n_states = 2 ** n_qubits_scenario
    n_total = n_qubits_scenario + 1
    ancilla = n_qubits_scenario

    qc = QuantumCircuit(n_total)

    # Normalise losses to [0, 1] for rotation encoding
    clipped_losses = np.maximum(losses, 0.0)
    max_loss = float(np.max(clipped_losses)) if len(clipped_losses) > 0 else 1.0
    if max_loss < 1e-16:
        max_loss = 1.0

    n_scenarios = min(len(losses), n_states)

    for i in range(n_scenarios):
        normalised_loss = min(clipped_losses[i] / max_loss, 1.0)
        if normalised_loss < 1e-16:
            continue

        # Rotation angle: sin^2(theta/2) = normalised_loss
        angle = 2.0 * np.arcsin(np.sqrt(normalised_loss))

        # Multi-controlled rotation conditioned on scenario state |i>
        bits = format(i, f"0{n_qubits_scenario}b")

        # Flip zero-bits for controlled operation
        for b_idx, b in enumerate(bits):
            if b == "0":
                qc.x(b_idx)

        # Apply controlled-RY
        if n_qubits_scenario == 1:
            qc.cry(angle, 0, ancilla)
        else:
            # Multi-controlled RY via decomposition:
            # MCX to auxiliary + CRY + MCX undo
            # For simplicity at small qubit counts, use direct mcx + cry pattern
            qc.mcx(list(range(n_qubits_scenario)), ancilla)
            # The mcx sets ancilla to |1> when all controls match.
            # We want a controlled rotation, so we undo the mcx and use
            # a different approach: iterate and apply single-controlled ops
            qc.mcx(list(range(n_qubits_scenario)), ancilla)

            # Use phase-rotation approach: apply RY on ancilla conditional
            # on all scenario qubits matching state |i>
            # Build a sub-circuit with mcx + single-qubit rotation
            _apply_multi_controlled_ry(qc, list(range(n_qubits_scenario)), ancilla, angle)

        # Undo flips
        for b_idx, b in enumerate(bits):
            if b == "0":
                qc.x(b_idx)

    return qc


def _apply_multi_controlled_ry(
    qc: object,
    controls: list[int],
    target: int,
    angle: float,
) -> None:
    """Apply a multi-controlled RY gate using ancilla-free decomposition.

    For small control counts (1-3 qubits), uses direct decomposition.
    """
    if len(controls) == 1:
        qc.cry(angle, controls[0], target)  # type: ignore[union-attr]
    elif len(controls) == 2:
        # Decompose 2-controlled RY using standard Toffoli decomposition
        from qiskit.circuit.library import RYGate
        cry_gate = RYGate(angle).control(2)
        qc.append(cry_gate, [*controls, target])  # type: ignore[union-attr]
    else:
        # For 3+ controls: use Qiskit's built-in multi-controlled gate
        from qiskit.circuit.library import RYGate
        cry_gate = RYGate(angle).control(len(controls))
        qc.append(cry_gate, [*controls, target])  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Quantum stress tester
# ---------------------------------------------------------------------------


class QuantumStressTester:
    """Quantum stress testing with superposition-encoded scenarios.

    Encodes stress scenarios as a weighted superposition state and uses
    quantum amplitude estimation to compute the probability-weighted
    expected portfolio loss across all scenarios simultaneously.

    Parameters
    ----------
    backend : Backend
        Quantum backend for circuit execution.
    qae_method : str
        QAE algorithm: ``"iqae"``, ``"mlae"``, or ``"canonical"``.
    qae_shots : int
        Shots per QAE round.
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        backend: Backend,
        qae_method: str = "iqae",
        qae_shots: int = 1024,
        seed: int | None = 42,
    ) -> None:
        self.backend = backend
        self.qae_method = qae_method
        self.qae_shots = qae_shots
        self.seed = seed

    def run(
        self,
        portfolio_value: float,
        weights: NDArray[np.float64],
        scenario_specs: list[StressScenarioSpec],
    ) -> QuantumStressResult:
        """Run quantum stress test.

        Parameters
        ----------
        portfolio_value : float
            Total portfolio market value.
        weights : NDArray, shape (4,)
            Sensitivity weights [equity, rates, vol, spreads].
        scenario_specs : list[StressScenarioSpec]
            Scenarios with probability weights.

        Returns
        -------
        QuantumStressResult
            Full result with per-scenario and aggregate metrics.
        """
        start = time.perf_counter()
        w = np.asarray(weights, dtype=np.float64)

        if w.shape != (4,):
            raise ValueError(f"weights must have shape (4,), got {w.shape}")
        if not scenario_specs:
            raise ValueError("At least one scenario is required.")

        # Compute scenario losses
        losses, probs, details = _compute_all_scenario_losses(
            portfolio_value, w, scenario_specs,
        )

        n_scenarios = len(scenario_specs)
        n_qubits_scenario = _n_qubits_for_scenarios(n_scenarios)

        # Build quantum circuits
        scenario_circ = build_scenario_superposition(probs, n_qubits_scenario)
        loss_oracle = build_loss_oracle(losses, n_qubits_scenario)

        # Combine: scenario loading + loss oracle
        from qiskit.circuit import QuantumCircuit

        n_total = n_qubits_scenario + 1  # scenario register + ancilla
        combined = QuantumCircuit(n_total)
        combined.compose(scenario_circ, range(n_qubits_scenario), inplace=True)
        combined.compose(loss_oracle, range(n_total), inplace=True)

        # QAE to estimate weighted expected loss
        problem = EstimationProblem(
            state_preparation=combined,
            objective_qubits=[n_qubits_scenario],
            n_qubits=n_total,
        )

        qae_estimate = self._run_qae(problem)

        # Scale QAE estimate back to loss units
        max_loss = float(np.max(np.maximum(losses, 0.0)))
        if max_loss < 1e-16:
            max_loss = 1.0
        quantum_expected_loss = qae_estimate * max_loss

        # Classical weighted expected loss for comparison
        classical_expected_loss = float(np.sum(losses * probs))

        # Worst case
        worst_idx = int(np.argmax(losses))
        worst_loss = float(losses[worst_idx])
        worst_name = scenario_specs[worst_idx].scenario.name

        wall_time = time.perf_counter() - start

        return QuantumStressResult(
            per_scenario=details,
            weighted_expected_loss=classical_expected_loss,
            worst_case_loss=worst_loss,
            worst_case_scenario=worst_name,
            quantum_estimate=quantum_expected_loss,
            method="quantum",
            n_scenarios=n_scenarios,
            n_qubits_scenario=n_qubits_scenario,
            wall_time_s=wall_time,
            metadata={
                "qae_method": self.qae_method,
                "qae_raw_estimate": qae_estimate,
                "max_loss": max_loss,
                "classical_expected_loss": classical_expected_loss,
                "circuit_depth": combined.depth(),
            },
        )

    def _run_qae(self, problem: EstimationProblem) -> float:
        """Run QAE and return the amplitude estimate."""
        if self.qae_method == "iqae":
            from qufin.options.amplitude_estimation.iqae import (
                IQAEConfig,
                IterativeAmplitudeEstimation,
            )

            cfg = IQAEConfig(
                epsilon_target=0.01,
                alpha=0.05,
                shots_per_round=self.qae_shots,
                seed=self.seed,
            )
            qae = IterativeAmplitudeEstimation(problem, cfg, self.backend)
            result = qae.estimate()
            return result.estimate

        elif self.qae_method == "mlae":
            from qufin.options.amplitude_estimation.mlae import (
                MaximumLikelihoodAmplitudeEstimation,
                MLAEConfig,
            )

            cfg = MLAEConfig(
                n_shots_per_round=self.qae_shots,
                seed=self.seed,
            )
            qae = MaximumLikelihoodAmplitudeEstimation(problem, cfg, self.backend)
            result = qae.estimate()
            return result.estimate

        elif self.qae_method == "canonical":
            from qufin.options.amplitude_estimation.canonical import (
                CanonicalAmplitudeEstimation,
                CanonicalQAEConfig,
            )

            cfg = CanonicalQAEConfig(
                n_eval_qubits=4,
                shots=self.qae_shots,
                seed=self.seed,
            )
            qae = CanonicalAmplitudeEstimation(problem, cfg, self.backend)
            result = qae.estimate()
            return result.estimate

        else:
            raise ValueError(f"Unknown QAE method: {self.qae_method!r}")


# ---------------------------------------------------------------------------
# Classical comparison
# ---------------------------------------------------------------------------


def classical_stress_test(
    portfolio_value: float,
    weights: NDArray[np.float64],
    scenario_specs: list[StressScenarioSpec],
    n_monte_carlo: int = 10_000,
    seed: int | None = 42,
) -> QuantumStressResult:
    """Classical scenario-weighted Monte Carlo stress test.

    Computes expected loss by sampling scenarios according to their
    probability weights and adding Monte Carlo noise to factor shocks.

    Parameters
    ----------
    portfolio_value : float
        Total portfolio market value.
    weights : NDArray, shape (4,)
        Sensitivity weights [equity, rates, vol, spreads].
    scenario_specs : list[StressScenarioSpec]
        Scenarios with probability weights.
    n_monte_carlo : int
        Number of Monte Carlo samples.
    seed : int or None
        Random seed.

    Returns
    -------
    QuantumStressResult
        Result with method="classical".
    """
    start = time.perf_counter()
    w = np.asarray(weights, dtype=np.float64)

    if w.shape != (4,):
        raise ValueError(f"weights must have shape (4,), got {w.shape}")
    if not scenario_specs:
        raise ValueError("At least one scenario is required.")

    # Compute deterministic scenario losses
    losses, probs, details = _compute_all_scenario_losses(
        portfolio_value, w, scenario_specs,
    )

    # Classical weighted expected loss (exact)
    classical_expected_loss = float(np.sum(losses * probs))

    # Monte Carlo estimation with noise
    rng = np.random.default_rng(seed)
    n_scenarios = len(scenario_specs)

    # Sample scenario indices according to probabilities
    scenario_indices = rng.choice(n_scenarios, size=n_monte_carlo, p=probs)

    # Add small noise (5% of shock magnitude) to simulate uncertainty
    mc_losses = np.zeros(n_monte_carlo, dtype=np.float64)
    for j in range(n_monte_carlo):
        idx = scenario_indices[j]
        sc = scenario_specs[idx].scenario
        noise = rng.normal(0, 0.05, size=4)
        shocks = np.array([
            sc.equity_shock * (1 + noise[0]),
            sc.rates_shock / 10_000 * (1 + noise[1]),
            sc.vol_shock * (1 + noise[2]),
            -sc.spread_shock / 10_000 * (1 + noise[3]),
        ])
        pnl = portfolio_value * w * shocks
        mc_losses[j] = -float(np.sum(pnl))

    mc_expected_loss = float(np.mean(mc_losses))

    # Worst case
    worst_idx = int(np.argmax(losses))
    worst_loss = float(losses[worst_idx])
    worst_name = scenario_specs[worst_idx].scenario.name

    wall_time = time.perf_counter() - start

    return QuantumStressResult(
        per_scenario=details,
        weighted_expected_loss=classical_expected_loss,
        worst_case_loss=worst_loss,
        worst_case_scenario=worst_name,
        quantum_estimate=mc_expected_loss,
        method="classical",
        n_scenarios=n_scenarios,
        n_qubits_scenario=0,
        wall_time_s=wall_time,
        metadata={
            "n_monte_carlo": n_monte_carlo,
            "mc_expected_loss": mc_expected_loss,
            "mc_std": float(np.std(mc_losses)),
        },
    )


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def run_quantum_stress_test(
    portfolio_value: float,
    weights: NDArray[np.float64],
    scenario_specs: list[StressScenarioSpec] | str | None = None,
    backend: Backend | None = None,
    qae_method: str = "iqae",
    qae_shots: int = 1024,
    seed: int | None = 42,
) -> QuantumStressResult:
    """Run a quantum stress test on a portfolio.

    Convenience wrapper that handles backend creation and scenario
    selection.

    Parameters
    ----------
    portfolio_value : float
        Total portfolio market value.
    weights : NDArray, shape (4,)
        Sensitivity weights [equity, rates, vol, spreads].
    scenario_specs : list[StressScenarioSpec], str, or None
        Scenario specifications. If a string, looks up a predefined set
        (``"gfc_2008"``, ``"covid_2020"``, ``"rate_hike_2022"``).
        If None, uses the GFC 2008 set.
    backend : Backend or None
        Quantum backend. If None, uses MockBackend.
    qae_method : str
        QAE algorithm to use.
    qae_shots : int
        Shots per QAE round.
    seed : int or None
        Random seed.

    Returns
    -------
    QuantumStressResult
        Full stress test result.
    """
    # Resolve scenarios
    if scenario_specs is None:
        specs = GFC_2008_SCENARIOS
    elif isinstance(scenario_specs, str):
        if scenario_specs not in PREDEFINED_SCENARIO_SETS:
            raise ValueError(
                f"Unknown scenario set: {scenario_specs!r}. "
                f"Available: {list(PREDEFINED_SCENARIO_SETS.keys())}"
            )
        specs = PREDEFINED_SCENARIO_SETS[scenario_specs]
    else:
        specs = scenario_specs

    # Resolve backend
    if backend is None:
        from qufin.backends.mock import MockBackend
        backend = MockBackend(seed=seed or 42)

    tester = QuantumStressTester(
        backend=backend,
        qae_method=qae_method,
        qae_shots=qae_shots,
        seed=seed,
    )

    return tester.run(portfolio_value, weights, specs)
