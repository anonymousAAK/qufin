"""Noise-aware variational parameter optimization for QAOA/VQE.

Implements optimization strategies that account for hardware noise
when tuning variational circuit parameters. Supports noise-agnostic,
noise-penalized, and robust (worst-case) optimization modes.

References
----------
Fontana et al., "Non-trivial symmetries in quantum landscapes and
  their resilience to noise" (2021).
Sharma et al., "Noise resilience of variational quantum compiling"
  PRA 102:062415 (2020).
Wang et al., "Noise-induced barren plateaus in variational quantum
  algorithms", Nat Commun 12:6961 (2021).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from qufin.backends.base import Backend

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class NoiseChannel:
    """A single noise channel on a gate.

    Parameters
    ----------
    gate_type : str
        Gate name (e.g. "cx", "rx", "rz").
    error_rate : float
        Depolarizing error probability for this gate.
    qubit_indices : tuple[int, ...]
        Qubits this channel acts on.
    """

    gate_type: str
    error_rate: float
    qubit_indices: tuple[int, ...]


@dataclass
class NoiseAwareConfig:
    """Configuration for noise-aware optimization.

    Parameters
    ----------
    noise_model : dict[str, float]
        Depolarizing error rates keyed by gate type,
        e.g. ``{"cx": 0.01, "rx": 0.001}``.
    optimization_method : str
        One of ``"noise_aware"``, ``"robust"``, ``"noise_agnostic"``.
    noise_budget : float
        Maximum acceptable cumulative noise level (0-1).
    calibration_drift_range : tuple[float, float]
        Multiplicative worst-case range for noise parameter drift,
        e.g. ``(0.8, 1.2)`` means +/- 20 %.
    penalty_weight : float
        Relative weight of the noise penalty term.
    maxiter : int
        Maximum optimizer iterations.
    optimizer : str
        SciPy optimizer method name.
    """

    noise_model: dict[str, float] = field(
        default_factory=lambda: {"cx": 0.01, "rx": 0.001, "rz": 0.0005}
    )
    optimization_method: Literal[
        "noise_aware", "robust", "noise_agnostic"
    ] = "noise_aware"
    noise_budget: float = 0.5
    calibration_drift_range: tuple[float, float] = (0.8, 1.2)
    penalty_weight: float = 0.1
    maxiter: int = 200
    optimizer: str = "COBYLA"


# ---------------------------------------------------------------------------
# DepolarizingModel — analytical noise estimation
# ---------------------------------------------------------------------------


class DepolarizingModel:
    """Analytical depolarizing noise model for circuit fidelity estimation.

    Provides fast, differentiable estimates of noise impact without
    running the circuit on a simulator.
    """

    def __init__(self, noise_channels: list[NoiseChannel]) -> None:
        self._channels = list(noise_channels)

    @classmethod
    def from_backend(cls, backend: Backend) -> DepolarizingModel:
        """Extract noise parameters from a backend.

        For ``NoisyAerBackend`` uses the stored ``NoiseProfile``.
        For other backends, returns a default low-noise model.
        """
        channels: list[NoiseChannel] = []
        profile = getattr(backend, "_profile", None)
        if profile is None:
            profile = getattr(backend, "noise_profile", None)
            if callable(profile):
                profile = None

        if profile is not None:
            single_err = getattr(profile, "single_gate_error", 1e-4)
            two_err = getattr(profile, "two_gate_error", 1e-3)
            for gate in ("rx", "ry", "rz", "h", "x", "s"):
                channels.append(
                    NoiseChannel(gate, single_err, (0,))
                )
            for gate in ("cx", "cz", "ecr"):
                channels.append(
                    NoiseChannel(gate, two_err, (0, 1))
                )
        else:
            channels.append(NoiseChannel("cx", 1e-3, (0, 1)))
            channels.append(NoiseChannel("rx", 1e-4, (0,)))

        return cls(channels)

    @property
    def channels(self) -> list[NoiseChannel]:
        """Return the list of noise channels."""
        return list(self._channels)

    def gate_error(self, gate_type: str) -> float:
        """Return the error rate for a gate type, or 0 if unknown."""
        for ch in self._channels:
            if ch.gate_type == gate_type:
                return ch.error_rate
        return 0.0

    def expected_fidelity(self, circuit: Any) -> float:
        """Estimate overall circuit fidelity under depolarizing noise.

        Uses the product formula  F = prod(1 - e_g)  over all gates g
        in the circuit, where e_g is the depolarizing error rate.

        Parameters
        ----------
        circuit : object
            Must expose an iterable ``data`` attribute whose elements
            have an ``operation.name`` attribute (Qiskit QuantumCircuit).
            If ``data`` is not available, falls back to ``depth()``
            with average error.

        Returns
        -------
        float
            Estimated fidelity in [0, 1].
        """
        gate_counts = _count_gates(circuit)
        fidelity = 1.0
        for gate_type, count in gate_counts.items():
            err = self.gate_error(gate_type)
            fidelity *= (1.0 - err) ** count
        return float(fidelity)

    def noise_gradient(
        self,
        circuit: Any,
        params: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Estimate gradient of noise cost w.r.t. variational parameters.

        Uses finite differences on the expected fidelity. The circuit
        must be rebuildable from parameters via a ``bind_parameters``
        method or equivalent.

        For circuits without ``bind_parameters``, returns a heuristic
        gradient proportional to the parameter magnitudes (larger
        rotation angles produce deeper effective circuits).

        Parameters
        ----------
        circuit : object
            Parameterized circuit template.
        params : NDArray
            Current parameter values.

        Returns
        -------
        NDArray of shape ``(len(params),)``.
        """
        n = len(params)
        grad = np.zeros(n, dtype=np.float64)

        has_bind = hasattr(circuit, "bind_parameters")
        if has_bind and hasattr(circuit, "parameters") and len(circuit.parameters) == n:
            eps = 1e-5
            for i in range(n):
                p_plus = params.copy()
                p_minus = params.copy()
                p_plus[i] += eps
                p_minus[i] -= eps
                param_list = list(circuit.parameters)
                f_plus = self.expected_fidelity(
                    circuit.assign_parameters(
                        dict(zip(param_list, p_plus, strict=False))
                    )
                )
                f_minus = self.expected_fidelity(
                    circuit.assign_parameters(
                        dict(zip(param_list, p_minus, strict=False))
                    )
                )
                grad[i] = (f_plus - f_minus) / (2 * eps)
        else:
            # Heuristic: larger angles -> more noise sensitivity
            gate_counts = _count_gates(circuit)
            total_gates = sum(gate_counts.values()) or 1
            avg_err = sum(
                self.gate_error(g) * c for g, c in gate_counts.items()
            ) / total_gates
            grad = -avg_err * np.abs(params) / (np.linalg.norm(params) + 1e-12)

        return grad


# ---------------------------------------------------------------------------
# NoiseAwareOptimizer
# ---------------------------------------------------------------------------


class NoiseAwareOptimizer:
    """Variational optimizer that incorporates a noise penalty.

    Wraps a cost function with a differentiable noise penalty so the
    optimizer favours parameter regions with lower noise impact.

    Parameters
    ----------
    backend : Backend
        Quantum backend for circuit execution.
    config : NoiseAwareConfig
        Optimization configuration.
    """

    def __init__(self, backend: Backend, config: NoiseAwareConfig) -> None:
        self.backend = backend
        self.config = config
        self._dep_model = DepolarizingModel.from_backend(backend)
        self._history: list[float] = []

    @property
    def history(self) -> list[float]:
        """Objective value history across optimizer iterations."""
        return list(self._history)

    # -- public API --

    def optimize(
        self,
        circuit: Any,
        initial_params: NDArray[np.float64],
        cost_fn: Callable[[dict[str, int], int], float],
        shots: int = 4096,
    ) -> dict[str, Any]:
        """Optimize variational parameters with optional noise penalty.

        Parameters
        ----------
        circuit : object
            Parameterized quantum circuit. Must support
            ``assign_parameters`` or be callable with params.
        initial_params : NDArray
            Starting parameter values.
        cost_fn : callable(counts, shots) -> float
            Maps measurement counts to a scalar cost.
        shots : int
            Shots per circuit execution.

        Returns
        -------
        Dict with keys: optimal_params, optimal_cost, history,
        noise_penalty, estimated_fidelity, method.
        """
        self._history = []
        method = self.config.optimization_method

        if method == "noise_agnostic":
            return self._optimize_agnostic(
                circuit, initial_params, cost_fn, shots
            )
        elif method == "robust":
            return self.robust_optimize(
                circuit,
                initial_params,
                cost_fn,
                self.config.calibration_drift_range,
                shots=shots,
            )
        else:  # noise_aware
            return self._optimize_noise_aware(
                circuit, initial_params, cost_fn, shots
            )

    def noise_penalty(
        self,
        circuit: Any,
        params: NDArray[np.float64],
        noise_model: dict[str, float] | None = None,
    ) -> float:
        """Compute a differentiable noise penalty for the current circuit.

        The penalty is  ``weight * (1 - expected_fidelity)``.

        Parameters
        ----------
        circuit : object
            Quantum circuit (bound or unbound).
        params : NDArray
            Current parameter values (used for heuristic scaling).
        noise_model : dict | None
            Override noise rates; if None uses config.

        Returns
        -------
        float  >= 0.
        """
        nm = noise_model or self.config.noise_model
        channels = [
            NoiseChannel(gate, rate, (0,) if rate < 0.005 else (0, 1))
            for gate, rate in nm.items()
        ]
        model = DepolarizingModel(channels)
        fidelity = model.expected_fidelity(circuit)
        # Scale by param norm to penalise large rotations
        param_scale = 1.0 + 0.01 * float(np.linalg.norm(params))
        return float(self.config.penalty_weight * (1.0 - fidelity) * param_scale)

    def expected_noise_cost(
        self,
        circuit: Any,
        noise_rates: dict[str, float],
    ) -> float:
        """Compute expected noise contribution for given rates.

        Returns the infidelity ``1 - F`` under the supplied noise model.
        """
        channels = [
            NoiseChannel(gate, rate, (0,) if rate < 0.005 else (0, 1))
            for gate, rate in noise_rates.items()
        ]
        model = DepolarizingModel(channels)
        return float(1.0 - model.expected_fidelity(circuit))

    def robust_optimize(
        self,
        circuit: Any,
        params: NDArray[np.float64],
        cost_fn: Callable[[dict[str, int], int], float],
        noise_range: tuple[float, float],
        shots: int = 4096,
    ) -> dict[str, Any]:
        """Optimize robust to calibration drift (worst-case over noise range).

        Evaluates the cost at the nominal noise level plus a worst-case
        penalty sampled from the calibration drift range.

        Parameters
        ----------
        circuit : object
            Parameterized circuit.
        params : NDArray
            Initial parameters.
        cost_fn : callable(counts, shots) -> float
            Cost function.
        noise_range : tuple[float, float]
            Multiplicative drift range, e.g. (0.8, 1.2).
        shots : int
            Shots per execution.

        Returns
        -------
        Dict with optimisation results.
        """
        self._history = []
        lo, hi = noise_range

        def robust_obj(p: NDArray[np.float64]) -> float:
            bound_circuit = _bind_params(circuit, p)
            result = self.backend.run(bound_circuit, shots=shots)
            base_cost = cost_fn(result.counts, shots)

            # Worst-case noise penalty over drift range
            penalties = []
            for scale in (lo, 1.0, hi):
                scaled_nm = {
                    g: r * scale for g, r in self.config.noise_model.items()
                }
                penalties.append(
                    self.noise_penalty(bound_circuit, p, scaled_nm)
                )
            worst_penalty = max(penalties)

            total = base_cost + worst_penalty
            self._history.append(total)
            return total

        opt = minimize(
            robust_obj,
            params,
            method=self.config.optimizer,
            options={"maxiter": self.config.maxiter},
        )

        bound_final = _bind_params(circuit, opt.x)
        final_fidelity = self._dep_model.expected_fidelity(bound_final)

        return {
            "optimal_params": opt.x,
            "optimal_cost": float(opt.fun),
            "history": list(self._history),
            "noise_penalty": self.noise_penalty(bound_final, opt.x),
            "estimated_fidelity": final_fidelity,
            "method": "robust",
            "noise_range": noise_range,
        }

    # -- private helpers --

    def _optimize_agnostic(
        self,
        circuit: Any,
        initial_params: NDArray[np.float64],
        cost_fn: Callable[[dict[str, int], int], float],
        shots: int,
    ) -> dict[str, Any]:
        """Standard optimization ignoring noise."""

        def objective(p: NDArray[np.float64]) -> float:
            bound_circuit = _bind_params(circuit, p)
            result = self.backend.run(bound_circuit, shots=shots)
            val = cost_fn(result.counts, shots)
            self._history.append(val)
            return val

        opt = minimize(
            objective,
            initial_params,
            method=self.config.optimizer,
            options={"maxiter": self.config.maxiter},
        )

        bound_final = _bind_params(circuit, opt.x)
        return {
            "optimal_params": opt.x,
            "optimal_cost": float(opt.fun),
            "history": list(self._history),
            "noise_penalty": 0.0,
            "estimated_fidelity": self._dep_model.expected_fidelity(
                bound_final
            ),
            "method": "noise_agnostic",
        }

    def _optimize_noise_aware(
        self,
        circuit: Any,
        initial_params: NDArray[np.float64],
        cost_fn: Callable[[dict[str, int], int], float],
        shots: int,
    ) -> dict[str, Any]:
        """Optimization with noise penalty added to cost."""

        def objective(p: NDArray[np.float64]) -> float:
            bound_circuit = _bind_params(circuit, p)
            result = self.backend.run(bound_circuit, shots=shots)
            base_cost = cost_fn(result.counts, shots)
            penalty = self.noise_penalty(bound_circuit, p)
            total = base_cost + penalty
            self._history.append(total)
            return total

        opt = minimize(
            objective,
            initial_params,
            method=self.config.optimizer,
            options={"maxiter": self.config.maxiter},
        )

        bound_final = _bind_params(circuit, opt.x)
        final_penalty = self.noise_penalty(bound_final, opt.x)
        final_fidelity = self._dep_model.expected_fidelity(bound_final)

        return {
            "optimal_params": opt.x,
            "optimal_cost": float(opt.fun),
            "history": list(self._history),
            "noise_penalty": final_penalty,
            "estimated_fidelity": final_fidelity,
            "method": "noise_aware",
        }


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def circuit_noise_budget(
    circuit: Any,
    noise_model: dict[str, float],
) -> float:
    """Compute total noise budget consumption for a circuit.

    Returns the fraction of fidelity lost: ``1 - prod(1 - e_g)``.

    Parameters
    ----------
    circuit : object
        Quantum circuit.
    noise_model : dict[str, float]
        Gate-type -> error rate mapping.

    Returns
    -------
    float in [0, 1].  0 = no noise; 1 = fully depolarized.
    """
    gate_counts = _count_gates(circuit)
    fidelity = 1.0
    for gate_type, count in gate_counts.items():
        err = noise_model.get(gate_type, 0.0)
        fidelity *= (1.0 - err) ** count
    return float(1.0 - fidelity)


def compare_noise_aware_vs_agnostic(
    circuit: Any,
    backend: Backend,
    problem: Callable[[dict[str, int], int], float],
    initial_params: NDArray[np.float64] | None = None,
    shots: int = 4096,
    maxiter: int = 50,
) -> dict[str, Any]:
    """Run a side-by-side comparison of noise-aware vs agnostic optimization.

    Parameters
    ----------
    circuit : object
        Parameterized circuit template.
    backend : Backend
        Noisy backend.
    problem : callable(counts, shots) -> float
        Cost function.
    initial_params : NDArray | None
        Starting parameters. If None, uses zeros.
    shots : int
        Shots per evaluation.
    maxiter : int
        Max iterations for each optimizer.

    Returns
    -------
    Dict with ``aware`` and ``agnostic`` result sub-dicts plus a ``summary``.
    """
    if initial_params is None:
        initial_params = np.zeros(2, dtype=np.float64)

    # Noise-agnostic run
    agnostic_cfg = NoiseAwareConfig(
        optimization_method="noise_agnostic",
        maxiter=maxiter,
    )
    agnostic_opt = NoiseAwareOptimizer(backend, agnostic_cfg)
    agnostic_result = agnostic_opt.optimize(
        circuit, initial_params.copy(), problem, shots=shots
    )

    # Noise-aware run
    aware_cfg = NoiseAwareConfig(
        optimization_method="noise_aware",
        maxiter=maxiter,
    )
    aware_opt = NoiseAwareOptimizer(backend, aware_cfg)
    aware_result = aware_opt.optimize(
        circuit, initial_params.copy(), problem, shots=shots
    )

    return {
        "aware": aware_result,
        "agnostic": agnostic_result,
        "summary": format_comparison_report(aware_result, agnostic_result),
    }


def format_comparison_report(
    aware_result: dict[str, Any],
    agnostic_result: dict[str, Any],
) -> str:
    """Format a human-readable comparison report.

    Parameters
    ----------
    aware_result : dict
        Result dict from noise-aware optimization.
    agnostic_result : dict
        Result dict from noise-agnostic optimization.

    Returns
    -------
    str  Multi-line formatted report.
    """
    aware_cost = aware_result.get("optimal_cost", float("nan"))
    agnostic_cost = agnostic_result.get("optimal_cost", float("nan"))
    aware_fid = aware_result.get("estimated_fidelity", float("nan"))
    agnostic_fid = agnostic_result.get("estimated_fidelity", float("nan"))
    aware_penalty = aware_result.get("noise_penalty", 0.0)

    lines = [
        "Noise-Aware vs Agnostic Comparison",
        "=" * 40,
        f"Noise-Aware  cost:     {aware_cost:.6f}",
        f"Agnostic     cost:     {agnostic_cost:.6f}",
        f"Noise-Aware  fidelity: {aware_fid:.6f}",
        f"Agnostic     fidelity: {agnostic_fid:.6f}",
        f"Noise penalty (aware): {aware_penalty:.6f}",
        f"Aware iterations:      {len(aware_result.get('history', []))}",
        f"Agnostic iterations:   {len(agnostic_result.get('history', []))}",
    ]

    if agnostic_cost != 0 and not np.isnan(agnostic_cost):
        improvement = (agnostic_cost - aware_cost) / abs(agnostic_cost) * 100
        lines.append(f"Cost improvement:      {improvement:+.2f}%")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------


def _count_gates(circuit: Any) -> dict[str, int]:
    """Count gates by type in a circuit.

    Handles Qiskit QuantumCircuit (``data`` attribute), plain dicts,
    and objects with a ``count_ops()`` method.
    """
    # Try count_ops() first (Qiskit QuantumCircuit)
    if hasattr(circuit, "count_ops"):
        ops = circuit.count_ops()
        return {
            k: v for k, v in ops.items()
            if k not in ("measure", "barrier")
        }

    # Try iterating over .data
    if hasattr(circuit, "data"):
        counts: dict[str, int] = {}
        for instruction in circuit.data:
            name = getattr(
                getattr(instruction, "operation", instruction),
                "name",
                str(instruction),
            )
            if name not in ("measure", "barrier"):
                counts[name] = counts.get(name, 0) + 1
        return counts

    # Fallback: assume shallow circuit
    depth = getattr(circuit, "depth", lambda: 5)
    if callable(depth):
        depth = depth()
    n_qubits = getattr(circuit, "num_qubits", 2)
    return {"u": depth * n_qubits}


def _bind_params(circuit: Any, params: NDArray[np.float64]) -> Any:
    """Bind parameter values to a parameterized circuit.

    Returns the original circuit unchanged if it has no free parameters.
    """
    if hasattr(circuit, "parameters") and len(circuit.parameters) > 0:
        param_list = list(circuit.parameters)
        n_bind = min(len(param_list), len(params))
        binding = dict(zip(param_list[:n_bind], params[:n_bind], strict=False))
        return circuit.assign_parameters(binding)
    return circuit
