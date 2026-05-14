"""Tests for noise-aware variational optimizer."""

from __future__ import annotations

import numpy as np
import pytest

from qufin.backends.mock import MockBackend
from qufin.backends.noise_aware_optimizer import (
    DepolarizingModel,
    NoiseAwareConfig,
    NoiseAwareOptimizer,
    NoiseChannel,
    circuit_noise_budget,
    compare_noise_aware_vs_agnostic,
    format_comparison_report,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeCircuit:
    """Minimal circuit-like object for testing without Qiskit."""

    def __init__(
        self,
        gate_ops: dict[str, int] | None = None,
        num_qubits: int = 2,
    ) -> None:
        self._ops = gate_ops or {"rx": 4, "cx": 2}
        self.num_qubits = num_qubits

    def count_ops(self) -> dict[str, int]:
        return dict(self._ops)

    def depth(self) -> int:
        return sum(self._ops.values())


def _simple_cost(counts: dict[str, int], shots: int) -> float:
    """Cost = probability of NOT measuring '0'."""
    return 1.0 - counts.get("0", 0) / shots


# ---------------------------------------------------------------------------
# NoiseChannel dataclass
# ---------------------------------------------------------------------------


class TestNoiseChannel:
    def test_creation(self) -> None:
        ch = NoiseChannel("cx", 0.01, (0, 1))
        assert ch.gate_type == "cx"
        assert ch.error_rate == 0.01
        assert ch.qubit_indices == (0, 1)

    def test_single_qubit_channel(self) -> None:
        ch = NoiseChannel("rx", 0.001, (0,))
        assert len(ch.qubit_indices) == 1


# ---------------------------------------------------------------------------
# NoiseAwareConfig
# ---------------------------------------------------------------------------


class TestNoiseAwareConfig:
    def test_defaults(self) -> None:
        cfg = NoiseAwareConfig()
        assert cfg.optimization_method == "noise_aware"
        assert cfg.noise_budget == 0.5
        assert cfg.calibration_drift_range == (0.8, 1.2)
        assert "cx" in cfg.noise_model
        assert cfg.penalty_weight == 0.1

    def test_custom_config(self) -> None:
        cfg = NoiseAwareConfig(
            noise_model={"cx": 0.05},
            optimization_method="robust",
            noise_budget=0.3,
            calibration_drift_range=(0.9, 1.1),
            penalty_weight=0.5,
        )
        assert cfg.optimization_method == "robust"
        assert cfg.noise_budget == 0.3
        assert cfg.noise_model == {"cx": 0.05}

    def test_noise_agnostic_method(self) -> None:
        cfg = NoiseAwareConfig(optimization_method="noise_agnostic")
        assert cfg.optimization_method == "noise_agnostic"


# ---------------------------------------------------------------------------
# DepolarizingModel
# ---------------------------------------------------------------------------


class TestDepolarizingModel:
    def test_from_channels(self) -> None:
        channels = [
            NoiseChannel("cx", 0.01, (0, 1)),
            NoiseChannel("rx", 0.001, (0,)),
        ]
        model = DepolarizingModel(channels)
        assert len(model.channels) == 2
        assert model.gate_error("cx") == 0.01
        assert model.gate_error("rx") == 0.001

    def test_unknown_gate_error(self) -> None:
        model = DepolarizingModel([NoiseChannel("cx", 0.01, (0, 1))])
        assert model.gate_error("unknown_gate") == 0.0

    def test_expected_fidelity_no_noise(self) -> None:
        model = DepolarizingModel([NoiseChannel("rx", 0.0, (0,))])
        circ = _FakeCircuit({"rx": 10})
        assert model.expected_fidelity(circ) == pytest.approx(1.0)

    def test_expected_fidelity_with_noise(self) -> None:
        model = DepolarizingModel([
            NoiseChannel("cx", 0.01, (0, 1)),
            NoiseChannel("rx", 0.001, (0,)),
        ])
        circ = _FakeCircuit({"rx": 4, "cx": 2})
        # F = (1-0.001)^4 * (1-0.01)^2
        expected = (0.999**4) * (0.99**2)
        assert model.expected_fidelity(circ) == pytest.approx(expected, rel=1e-6)

    def test_expected_fidelity_deep_circuit(self) -> None:
        """Deeper circuits should have lower fidelity."""
        model = DepolarizingModel([NoiseChannel("cx", 0.01, (0, 1))])
        shallow = _FakeCircuit({"cx": 2})
        deep = _FakeCircuit({"cx": 20})
        assert model.expected_fidelity(shallow) > model.expected_fidelity(deep)

    def test_from_mock_backend(self) -> None:
        backend = MockBackend()
        model = DepolarizingModel.from_backend(backend)
        # MockBackend has no _profile, should get defaults
        assert len(model.channels) > 0

    def test_noise_gradient_heuristic(self) -> None:
        model = DepolarizingModel([NoiseChannel("cx", 0.01, (0, 1))])
        circ = _FakeCircuit({"cx": 5})
        params = np.array([0.5, 1.0, 0.3])
        grad = model.noise_gradient(circ, params)
        assert grad.shape == (3,)
        # Gradient should be non-zero for non-zero params
        assert not np.allclose(grad, 0.0)


# ---------------------------------------------------------------------------
# NoiseAwareOptimizer
# ---------------------------------------------------------------------------


class TestNoiseAwareOptimizer:
    def test_init(self) -> None:
        backend = MockBackend()
        cfg = NoiseAwareConfig()
        opt = NoiseAwareOptimizer(backend, cfg)
        assert opt.backend is backend
        assert opt.config is cfg
        assert opt.history == []

    def test_noise_penalty_returns_nonneg(self) -> None:
        backend = MockBackend()
        cfg = NoiseAwareConfig(noise_model={"cx": 0.01, "rx": 0.001})
        opt = NoiseAwareOptimizer(backend, cfg)
        circ = _FakeCircuit({"cx": 5, "rx": 10})
        penalty = opt.noise_penalty(circ, np.array([1.0, 2.0]))
        assert penalty >= 0.0

    def test_noise_penalty_zero_for_ideal(self) -> None:
        backend = MockBackend()
        cfg = NoiseAwareConfig(
            noise_model={"cx": 0.0, "rx": 0.0},
            penalty_weight=1.0,
        )
        opt = NoiseAwareOptimizer(backend, cfg)
        circ = _FakeCircuit({"cx": 5, "rx": 10})
        penalty = opt.noise_penalty(circ, np.array([0.0]))
        assert penalty == pytest.approx(0.0, abs=1e-10)

    def test_expected_noise_cost(self) -> None:
        backend = MockBackend()
        cfg = NoiseAwareConfig()
        opt = NoiseAwareOptimizer(backend, cfg)
        circ = _FakeCircuit({"cx": 10})
        cost = opt.expected_noise_cost(circ, {"cx": 0.01})
        expected = 1.0 - 0.99**10
        assert cost == pytest.approx(expected, rel=1e-6)

    def test_optimize_noise_agnostic(self) -> None:
        backend = MockBackend(default_counts={"0": 800, "1": 200})
        cfg = NoiseAwareConfig(
            optimization_method="noise_agnostic",
            maxiter=5,
        )
        opt = NoiseAwareOptimizer(backend, cfg)
        circ = _FakeCircuit()
        params = np.array([0.1, 0.2])
        result = opt.optimize(circ, params, _simple_cost, shots=1024)
        assert "optimal_params" in result
        assert "optimal_cost" in result
        assert result["method"] == "noise_agnostic"
        assert result["noise_penalty"] == 0.0

    def test_optimize_noise_aware(self) -> None:
        backend = MockBackend(default_counts={"0": 800, "1": 200})
        cfg = NoiseAwareConfig(
            optimization_method="noise_aware",
            maxiter=5,
        )
        opt = NoiseAwareOptimizer(backend, cfg)
        circ = _FakeCircuit()
        params = np.array([0.1, 0.2])
        result = opt.optimize(circ, params, _simple_cost, shots=1024)
        assert result["method"] == "noise_aware"
        assert "noise_penalty" in result
        assert "estimated_fidelity" in result
        assert len(result["history"]) > 0

    def test_optimize_robust(self) -> None:
        backend = MockBackend(default_counts={"0": 800, "1": 200})
        cfg = NoiseAwareConfig(
            optimization_method="robust",
            maxiter=5,
            calibration_drift_range=(0.9, 1.1),
        )
        opt = NoiseAwareOptimizer(backend, cfg)
        circ = _FakeCircuit()
        params = np.array([0.1, 0.2])
        result = opt.optimize(circ, params, _simple_cost, shots=1024)
        assert result["method"] == "robust"
        assert "noise_range" in result

    def test_robust_optimize_direct(self) -> None:
        backend = MockBackend(default_counts={"0": 600, "1": 400})
        cfg = NoiseAwareConfig(maxiter=3)
        opt = NoiseAwareOptimizer(backend, cfg)
        circ = _FakeCircuit()
        params = np.array([0.5])
        result = opt.robust_optimize(
            circ, params, _simple_cost, (0.8, 1.2), shots=1024
        )
        assert result["method"] == "robust"
        assert isinstance(result["optimal_params"], np.ndarray)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


class TestCircuitNoiseBudget:
    def test_zero_noise(self) -> None:
        circ = _FakeCircuit({"cx": 10, "rx": 20})
        budget = circuit_noise_budget(circ, {"cx": 0.0, "rx": 0.0})
        assert budget == pytest.approx(0.0)

    def test_positive_noise(self) -> None:
        circ = _FakeCircuit({"cx": 10})
        budget = circuit_noise_budget(circ, {"cx": 0.01})
        expected = 1.0 - 0.99**10
        assert budget == pytest.approx(expected, rel=1e-6)

    def test_missing_gate_uses_zero(self) -> None:
        circ = _FakeCircuit({"rz": 5})
        budget = circuit_noise_budget(circ, {"cx": 0.01})
        assert budget == pytest.approx(0.0)


class TestComparisonHelpers:
    def test_format_comparison_report(self) -> None:
        aware = {
            "optimal_cost": 0.3,
            "estimated_fidelity": 0.95,
            "noise_penalty": 0.02,
            "history": [0.5, 0.4, 0.3],
        }
        agnostic = {
            "optimal_cost": 0.35,
            "estimated_fidelity": 0.90,
            "history": [0.6, 0.5, 0.4, 0.35],
        }
        report = format_comparison_report(aware, agnostic)
        assert "Noise-Aware" in report
        assert "Agnostic" in report
        assert "0.300000" in report
        assert "improvement" in report.lower()

    def test_compare_noise_aware_vs_agnostic(self) -> None:
        backend = MockBackend(default_counts={"0": 700, "1": 300})
        circ = _FakeCircuit()
        params = np.array([0.1, 0.2])
        result = compare_noise_aware_vs_agnostic(
            circ, backend, _simple_cost,
            initial_params=params,
            maxiter=3,
        )
        assert "aware" in result
        assert "agnostic" in result
        assert "summary" in result
        assert isinstance(result["summary"], str)

    def test_compare_default_params(self) -> None:
        backend = MockBackend(default_counts={"0": 500, "1": 500})
        circ = _FakeCircuit()
        result = compare_noise_aware_vs_agnostic(
            circ, backend, _simple_cost, maxiter=2
        )
        assert result["aware"]["method"] == "noise_aware"
        assert result["agnostic"]["method"] == "noise_agnostic"
