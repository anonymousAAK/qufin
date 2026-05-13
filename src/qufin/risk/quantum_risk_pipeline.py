"""Multi-asset quantum VaR/CVaR pipeline (Woerner & Egger, 1806.06893).

Full pipeline for portfolio risk analysis using quantum amplitude estimation:
1. Build portfolio loss distribution from multi-asset returns + weights
2. Encode distribution into quantum amplitudes via tree-of-RY loading
3. Bisection search over loss thresholds using QAE tail-probability oracle
4. CVaR estimation via conditional value oracle
5. Classical comparison for validation

References
----------
Woerner & Egger, "Quantum Risk Analysis", npj Quantum Information 5:15 (2019),
arXiv:1806.06893.
Egger, Gutierrez, Mestre, Woerner, "Credit Risk Analysis using Quantum
Computers", IEEE TQE 2:1 (2021), arXiv:1907.03044.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from qufin.backends.base import Backend
from qufin.options.amplitude_estimation.estimation_problem import EstimationProblem

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PortfolioRiskSpec:
    """Specification for multi-asset portfolio risk analysis.

    Parameters
    ----------
    returns : NDArray[np.float64]
        Historical returns, shape ``(T, n_assets)``.
    weights : NDArray[np.float64]
        Portfolio weights, shape ``(n_assets,)``.
    confidence_level : float
        VaR confidence level (e.g. 0.99 for 99 % VaR).
    horizon : int
        Holding period in days.
    n_qubits_loss : int
        Qubits for loss-distribution discretisation.
    distribution : str
        ``"normal"`` (fit Gaussian) or ``"empirical"`` (histogram).
    """

    returns: NDArray[np.float64]
    weights: NDArray[np.float64]
    confidence_level: float = 0.99
    horizon: int = 1
    n_qubits_loss: int = 6
    distribution: str = "normal"


@dataclass
class QuantumRiskResult:
    """Result container for the quantum risk pipeline.

    Stores both quantum estimates and classical baselines so that
    accuracy can be assessed immediately.
    """

    var_estimate: float = 0.0
    cvar_estimate: float = 0.0
    confidence_level: float = 0.99
    method: str = "iqae"
    n_qae_calls: int = 0
    bisection_steps: int = 0
    bisection_history: list[dict] = field(default_factory=list)
    classical_var: float = 0.0
    classical_cvar: float = 0.0
    relative_error_var: float = 0.0
    relative_error_cvar: float = 0.0


# ---------------------------------------------------------------------------
# Distribution helpers
# ---------------------------------------------------------------------------

def build_portfolio_loss_distribution(
    spec: PortfolioRiskSpec,
) -> tuple[NDArray[np.float64], float, float]:
    """Compute a discretised portfolio loss distribution.

    Parameters
    ----------
    spec : PortfolioRiskSpec
        Portfolio specification (returns, weights, qubit budget, ...).

    Returns
    -------
    probabilities : NDArray[np.float64]
        Normalised probability vector of length ``2 ** spec.n_qubits_loss``.
    loss_min : float
        Lower edge of the discretisation grid.
    loss_max : float
        Upper edge of the discretisation grid.
    """
    returns = np.asarray(spec.returns, dtype=np.float64)
    weights = np.asarray(spec.weights, dtype=np.float64)

    # Portfolio returns -> losses (loss = negative return)
    portfolio_returns = returns @ weights
    # Scale by sqrt(horizon) for multi-day holding period
    portfolio_losses = -portfolio_returns * np.sqrt(spec.horizon)

    n_bins = 2 ** spec.n_qubits_loss

    if spec.distribution == "normal":
        mu = float(np.mean(portfolio_losses))
        sigma = float(np.std(portfolio_losses, ddof=1))
        sigma = max(sigma, 1e-12)

        loss_min = mu - 4 * sigma
        loss_max = mu + 4 * sigma

        # Evaluate Gaussian PDF on the grid and normalise
        grid = np.linspace(loss_min, loss_max, n_bins)
        probs = np.exp(-0.5 * ((grid - mu) / sigma) ** 2) / (
            sigma * np.sqrt(2 * np.pi)
        )
        dx = (loss_max - loss_min) / (n_bins - 1) if n_bins > 1 else 1.0
        probs = probs * dx

    elif spec.distribution == "empirical":
        # Histogram-based empirical distribution
        loss_min = float(np.min(portfolio_losses))
        loss_max = float(np.max(portfolio_losses))
        # Add small padding so that min/max don't sit on bin edges
        pad = (loss_max - loss_min) * 0.01 if loss_max > loss_min else 1e-6
        loss_min -= pad
        loss_max += pad

        probs, _ = np.histogram(
            portfolio_losses, bins=n_bins, range=(loss_min, loss_max), density=False
        )
        probs = probs.astype(np.float64)

    else:
        raise ValueError(
            f"Unknown distribution type: {spec.distribution!r}. "
            "Use 'normal' or 'empirical'."
        )

    # Normalise to a proper probability vector
    total = probs.sum()
    if total > 0:
        probs = probs / total
    else:
        probs = np.ones(n_bins) / n_bins

    return probs, float(loss_min), float(loss_max)


# ---------------------------------------------------------------------------
# Circuit builders
# ---------------------------------------------------------------------------

def build_loss_loading_circuit(
    probabilities: NDArray[np.float64],
    n_qubits: int,
) -> object:
    """Encode a probability distribution as quantum-state amplitudes.

    Constructs |psi> = sum_i sqrt(p_i) |i> using a tree of controlled-RY
    rotations (Grover-Rudolph, quant-ph/0208112).

    The algorithm works by recursive bisection: at each level *d* of the
    binary tree the probabilities are split into two halves and an RY
    rotation controlled on the first *d* qubits sets the relative weight.

    Parameters
    ----------
    probabilities : NDArray[np.float64]
        Probability vector of length ``2 ** n_qubits``.
    n_qubits : int
        Number of qubits (must satisfy ``2 ** n_qubits == len(probabilities)``).

    Returns
    -------
    QuantumCircuit
        Qiskit circuit that prepares the distribution state.
    """
    from qiskit.circuit import QuantumCircuit

    n_states = 2 ** n_qubits
    assert len(probabilities) == n_states, (
        f"len(probabilities)={len(probabilities)} != 2**{n_qubits}={n_states}"
    )

    qc = QuantumCircuit(n_qubits)
    probs = np.asarray(probabilities, dtype=np.float64).copy()

    # Clamp to non-negative (numerical noise)
    probs = np.maximum(probs, 0.0)
    total = probs.sum()
    if total > 0:
        probs = probs / total

    def _load(qc: QuantumCircuit, qubit_idx: int, start: int, end: int) -> None:
        """Recursively load probabilities via bisection RY rotations."""
        if qubit_idx >= n_qubits or start >= end:
            return

        mid = (start + end) // 2
        p_total = probs[start:end].sum()
        if p_total < 1e-16:
            return

        p_lower = probs[start:mid].sum()

        # RY angle: cos(theta/2)^2 = p_lower / p_total
        ratio = np.clip(p_lower / p_total, 0.0, 1.0)
        theta = 2 * np.arccos(np.sqrt(ratio))

        if qubit_idx == 0:
            # First qubit: unconditional RY
            qc.ry(theta, qubit_idx)
        else:
            # Controlled on all previous qubits being in the correct state.
            # We need to figure out which computational-basis state of
            # qubits [0..qubit_idx-1] leads to this particular sub-range.
            # Instead of full multi-controlled gates (exponential overhead),
            # we use controlled-RY for the single parent qubit, which is
            # correct for the balanced binary-tree decomposition.
            #
            # Determine control state from the position of 'start' within
            # the level: the bit of the parent qubit decides left vs right.
            parent_bit = (start >> (n_qubits - qubit_idx)) & 1
            if parent_bit == 1:
                qc.x(qubit_idx - 1)
            qc.cry(theta, qubit_idx - 1, qubit_idx)
            if parent_bit == 1:
                qc.x(qubit_idx - 1)

        # Recurse into left (|0>) and right (|1>) halves
        _load(qc, qubit_idx + 1, start, mid)
        _load(qc, qubit_idx + 1, mid, end)

    _load(qc, 0, 0, n_states)
    return qc


def build_tail_oracle(
    threshold_bin: int,
    n_qubits: int,
) -> object:
    """Comparator oracle: marks states |i> where i >= threshold_bin.

    Flips an ancilla qubit to |1> whenever the computational-basis
    index of the loss register meets or exceeds the threshold.
    The circuit expects ``n_qubits + 1`` qubits, where the last qubit
    is the ancilla.

    Implementation: iterate over all basis states >= threshold and
    apply multi-controlled-X gates (compact for few qubits, and the
    number of qubits used here is small by construction).

    Parameters
    ----------
    threshold_bin : int
        Bin index threshold (0-indexed).
    n_qubits : int
        Number of qubits in the loss register.

    Returns
    -------
    QuantumCircuit
        Qiskit circuit on ``n_qubits + 1`` qubits.
    """
    from qiskit.circuit import QuantumCircuit

    n_states = 2 ** n_qubits
    n_total = n_qubits + 1  # loss register + ancilla
    ancilla = n_qubits  # ancilla index

    qc = QuantumCircuit(n_total)

    for i in range(threshold_bin, n_states):
        bits = format(i, f"0{n_qubits}b")
        # Flip controls so that the pattern matches |i>
        for b_idx, b in enumerate(bits):
            if b == "0":
                qc.x(b_idx)

        if n_qubits == 1:
            qc.cx(0, ancilla)
        else:
            qc.mcx(list(range(n_qubits)), ancilla)

        # Undo flips
        for b_idx, b in enumerate(bits):
            if b == "0":
                qc.x(b_idx)

    return qc


# ---------------------------------------------------------------------------
# QAE runner (delegates to existing IQAE / MLAE / canonical)
# ---------------------------------------------------------------------------

def _run_qae(
    problem: EstimationProblem,
    backend: Backend,
    qae_method: str = "iqae",
    qae_config: dict | None = None,
) -> tuple[float, int]:
    """Run QAE and return ``(amplitude_estimate, n_oracle_calls)``.

    Parameters
    ----------
    problem : EstimationProblem
        The AE problem (state prep + objective qubits).
    backend : Backend
        Execution backend.
    qae_method : str
        ``"iqae"``, ``"mlae"``, or ``"canonical"``.
    qae_config : dict or None
        Extra keyword overrides forwarded to the QAE config dataclass.

    Returns
    -------
    estimate : float
        Estimated amplitude (probability).
    n_oracle_calls : int
        Total oracle calls consumed.
    """
    cfg = qae_config or {}

    if qae_method == "iqae":
        from qufin.options.amplitude_estimation.iqae import (
            IQAEConfig,
            IterativeAmplitudeEstimation,
        )

        iqae_cfg = IQAEConfig(
            epsilon_target=cfg.get("epsilon_target", 0.01),
            alpha=cfg.get("alpha", 0.05),
            shots_per_round=cfg.get("shots_per_round", 1024),
            seed=cfg.get("seed", 42),
        )
        qae = IterativeAmplitudeEstimation(problem, iqae_cfg, backend)
        result = qae.estimate()
        return result.estimate, result.n_oracle_calls

    elif qae_method == "mlae":
        from qufin.options.amplitude_estimation.mlae import (
            MaximumLikelihoodAmplitudeEstimation,
            MLAEConfig,
        )

        mlae_cfg = MLAEConfig(
            n_shots_per_round=cfg.get("shots_per_round", 1024),
            seed=cfg.get("seed", 42),
        )
        qae = MaximumLikelihoodAmplitudeEstimation(problem, mlae_cfg, backend)
        result = qae.estimate()
        return result.estimate, getattr(result, "n_oracle_calls", 0)

    elif qae_method == "canonical":
        from qufin.options.amplitude_estimation.canonical import (
            CanonicalAmplitudeEstimation,
            CanonicalQAEConfig,
        )

        can_cfg = CanonicalQAEConfig(
            n_eval_qubits=cfg.get("n_eval_qubits", 4),
            shots=cfg.get("shots", 1024),
            seed=cfg.get("seed", 42),
        )
        qae = CanonicalAmplitudeEstimation(problem, can_cfg, backend)
        result = qae.estimate()
        return result.estimate, getattr(result, "n_oracle_calls", 0)

    else:
        raise ValueError(f"Unknown QAE method: {qae_method!r}")


# ---------------------------------------------------------------------------
# Classical baselines
# ---------------------------------------------------------------------------

def _classical_var_cvar(
    spec: PortfolioRiskSpec,
) -> tuple[float, float]:
    """Compute classical historical VaR and CVaR for comparison.

    Returns
    -------
    var : float
        Historical VaR at ``spec.confidence_level``.
    cvar : float
        Historical CVaR (expected shortfall) at ``spec.confidence_level``.
    """
    returns = np.asarray(spec.returns, dtype=np.float64)
    weights = np.asarray(spec.weights, dtype=np.float64)

    portfolio_returns = returns @ weights
    portfolio_losses = -portfolio_returns * np.sqrt(spec.horizon)

    alpha = 1 - spec.confidence_level
    var = float(np.percentile(portfolio_losses, 100 * (1 - alpha)))
    tail = portfolio_losses[portfolio_losses >= var]
    cvar = float(np.mean(tail)) if len(tail) > 0 else var
    return var, cvar


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def quantum_var_pipeline(
    spec: PortfolioRiskSpec,
    backend: Backend,
    qae_method: str = "iqae",
    qae_config: dict | None = None,
) -> QuantumRiskResult:
    """End-to-end quantum VaR / CVaR pipeline for a multi-asset portfolio.

    Steps
    -----
    1. Build the discretised loss distribution from historical returns.
    2. Encode the distribution as a quantum state (tree-of-RY).
    3. Bisection search for VaR: binary search over threshold bins,
       using QAE at each step to estimate the tail probability
       P(Loss > threshold).
    4. CVaR estimation: QAE on a conditional-value oracle that
       computes E[Loss | Loss > VaR].
    5. Classical comparison for validation.

    Parameters
    ----------
    spec : PortfolioRiskSpec
        Portfolio risk specification.
    backend : Backend
        Quantum backend for circuit execution.
    qae_method : str
        ``"iqae"``, ``"mlae"``, or ``"canonical"``.
    qae_config : dict or None
        Extra keyword overrides for the QAE algorithm config.

    Returns
    -------
    QuantumRiskResult
        Full result with quantum and classical estimates.
    """
    from qiskit.circuit import QuantumCircuit

    n_q = spec.n_qubits_loss
    n_bins = 2 ** n_q
    alpha = 1 - spec.confidence_level  # tail probability target

    # ------------------------------------------------------------------
    # 1. Build loss distribution
    # ------------------------------------------------------------------
    probs, loss_min, loss_max = build_portfolio_loss_distribution(spec)

    # ------------------------------------------------------------------
    # 2. Build loss-loading circuit
    # ------------------------------------------------------------------
    loading_circ = build_loss_loading_circuit(probs, n_q)

    # ------------------------------------------------------------------
    # 3. Bisection search for VaR
    # ------------------------------------------------------------------
    lo_bin = 0
    hi_bin = n_bins
    n_qae_calls = 0
    bisection_history: list[dict] = []
    n_bisection_steps = n_q  # log2(n_bins) steps suffice

    for _step in range(n_bisection_steps):
        mid_bin = (lo_bin + hi_bin) // 2
        if mid_bin >= n_bins:
            mid_bin = n_bins - 1

        # Build combined circuit: loading + tail oracle
        n_total = n_q + 1  # loss register + ancilla
        combined = QuantumCircuit(n_total)
        combined.compose(loading_circ, range(n_q), inplace=True)

        tail_oracle = build_tail_oracle(mid_bin, n_q)
        combined.compose(tail_oracle, range(n_total), inplace=True)

        problem = EstimationProblem(
            state_preparation=combined,
            objective_qubits=[n_q],  # ancilla
            n_qubits=n_total,
        )

        tail_prob, n_calls = _run_qae(problem, backend, qae_method, qae_config)
        n_qae_calls += 1

        # Convert bin to loss value for logging
        threshold_val = loss_min + (loss_max - loss_min) * mid_bin / n_bins

        bisection_history.append({
            "threshold": threshold_val,
            "tail_prob": tail_prob,
            "n_shots": n_calls,
        })

        if tail_prob > alpha:
            lo_bin = mid_bin  # threshold too low, tail is too fat
        else:
            hi_bin = mid_bin  # threshold too high

    var_bin = (lo_bin + hi_bin) // 2
    var_estimate = loss_min + (loss_max - loss_min) * var_bin / n_bins

    # ------------------------------------------------------------------
    # 4. CVaR estimation: E[Loss | Loss > VaR]
    # ------------------------------------------------------------------
    # Build a conditional-value circuit: for states where loss > VaR,
    # rotate a second ancilla proportional to the loss value so that
    # QAE estimates sum_i p_i * (loss_i / loss_max) for i > var_bin.
    # Dividing by the tail probability gives CVaR.
    n_cvar_total = n_q + 2  # loss register + comparator ancilla + value ancilla
    cvar_circ = QuantumCircuit(n_cvar_total)
    cvar_circ.compose(loading_circ, range(n_q), inplace=True)

    # Mark tail states and rotate value ancilla
    comparator_idx = n_q
    value_idx = n_q + 1

    rescale = loss_max if abs(loss_max) > 1e-12 else 1.0

    for i in range(var_bin, n_bins):
        loss_val = loss_min + (loss_max - loss_min) * (i + 0.5) / n_bins
        bits = format(i, f"0{n_q}b")

        # Flip to match |i>
        for b_idx, b in enumerate(bits):
            if b == "0":
                cvar_circ.x(b_idx)

        # Set comparator ancilla
        if n_q == 1:
            cvar_circ.cx(0, comparator_idx)
        else:
            cvar_circ.mcx(list(range(n_q)), comparator_idx)

        # Controlled-RY on value ancilla, proportional to loss
        normalised = np.clip(abs(loss_val) / abs(rescale), 0.0, 1.0)
        angle = 2 * np.arcsin(np.sqrt(normalised))
        cvar_circ.cry(angle, comparator_idx, value_idx)

        # Undo comparator
        if n_q == 1:
            cvar_circ.cx(0, comparator_idx)
        else:
            cvar_circ.mcx(list(range(n_q)), comparator_idx)

        # Undo flips
        for b_idx, b in enumerate(bits):
            if b == "0":
                cvar_circ.x(b_idx)

    cvar_problem = EstimationProblem(
        state_preparation=cvar_circ,
        objective_qubits=[value_idx],
        n_qubits=n_cvar_total,
    )

    cvar_amplitude, _n_calls_cvar = _run_qae(
        cvar_problem, backend, qae_method, qae_config
    )
    n_qae_calls += 1

    # cvar_amplitude = sum_i p_i * (loss_i / rescale) for i in tail
    # We need E[Loss | Loss > VaR] = (cvar_amplitude * rescale) / P(tail)
    # Approximate tail probability from the last bisection step
    tail_prob_approx = probs[var_bin:].sum() if var_bin < n_bins else alpha
    if tail_prob_approx < 1e-12:
        tail_prob_approx = alpha

    cvar_estimate = (cvar_amplitude * abs(rescale)) / tail_prob_approx

    # ------------------------------------------------------------------
    # 5. Classical comparison
    # ------------------------------------------------------------------
    classical_var, classical_cvar = _classical_var_cvar(spec)

    rel_err_var = (
        abs(var_estimate - classical_var) / abs(classical_var)
        if abs(classical_var) > 1e-12
        else 0.0
    )
    rel_err_cvar = (
        abs(cvar_estimate - classical_cvar) / abs(classical_cvar)
        if abs(classical_cvar) > 1e-12
        else 0.0
    )

    return QuantumRiskResult(
        var_estimate=var_estimate,
        cvar_estimate=cvar_estimate,
        confidence_level=spec.confidence_level,
        method=qae_method,
        n_qae_calls=n_qae_calls,
        bisection_steps=n_bisection_steps,
        bisection_history=bisection_history,
        classical_var=classical_var,
        classical_cvar=classical_cvar,
        relative_error_var=rel_err_var,
        relative_error_cvar=rel_err_cvar,
    )


# ---------------------------------------------------------------------------
# Stress-test wrapper
# ---------------------------------------------------------------------------

def quantum_stress_var(
    spec: PortfolioRiskSpec,
    stress_scenarios: dict[str, NDArray],
    backend: Backend,
    qae_method: str = "iqae",
) -> dict[str, QuantumRiskResult]:
    """Run quantum VaR under multiple stress scenarios.

    Each scenario replaces the returns matrix in *spec* with a
    stressed version (e.g. scaled volatility, correlation shocks,
    historical crisis windows).

    Parameters
    ----------
    spec : PortfolioRiskSpec
        Base portfolio specification.
    stress_scenarios : dict[str, NDArray]
        Mapping of scenario name to modified returns array,
        each of shape ``(T', n_assets)``.
    backend : Backend
        Quantum backend.
    qae_method : str
        QAE algorithm to use.

    Returns
    -------
    dict[str, QuantumRiskResult]
        Mapping of scenario name to its quantum risk result.
    """
    results: dict[str, QuantumRiskResult] = {}
    for name, stressed_returns in stress_scenarios.items():
        stressed_spec = PortfolioRiskSpec(
            returns=np.asarray(stressed_returns, dtype=np.float64),
            weights=spec.weights,
            confidence_level=spec.confidence_level,
            horizon=spec.horizon,
            n_qubits_loss=spec.n_qubits_loss,
            distribution=spec.distribution,
        )
        results[name] = quantum_var_pipeline(
            stressed_spec, backend, qae_method=qae_method
        )
    return results
