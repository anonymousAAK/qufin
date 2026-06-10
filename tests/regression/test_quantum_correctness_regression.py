"""Regression tests for quantum numerical-correctness fixes.

These lock in the corrections to a systematic qubit-endianness bug and a
non-invertible state-preparation crash that previously produced silently wrong
quantum risk numbers, wrong option prices, and wrong portfolio asset mappings.
Each test encodes the *correct* expected behaviour with an analytic or
exhaustive ground truth so the bugs cannot regress unnoticed.

Pre-fix symptoms (for context):
* quantum VaR on N(0,1) returned ~2.59 vs the true 1.645 (57% error);
* European QAE crashed on qiskit >= 1.0 (non-invertible ``initialize``) and,
  when it ran, mispriced by ~2.5x;
* QAOA/VQE returned the *modal* bitstring (not the lowest-energy one) and mapped
  capital to bit-reversed assets.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from qufin.backends.qiskit_backend import QiskitAerBackend

pytestmark = pytest.mark.regression


# ---------------------------------------------------------------------------
# Quantum VaR: comparator endianness
# ---------------------------------------------------------------------------


def test_quantum_var_comparator_marks_correct_tail() -> None:
    """The tail comparator must mark exactly the basis states whose value
    exceeds the threshold (in qiskit little-endian order), not their bit
    reversal."""
    from qiskit.quantum_info import Statevector

    from qufin.options.distributions import normal_distribution
    from qufin.risk.quantum_var import _build_tail_probability_problem

    dist = normal_distribution(n_qubits=4, mean=0.0, std=1.0, n_sigma=4.0)
    threshold = 1.0
    expected = {int(i) for i in np.where(dist.values > threshold)[0]}

    problem = _build_tail_probability_problem(dist, threshold)
    sv = Statevector.from_instruction(problem.state_preparation)

    n_q = dist.n_qubits
    marked: set[int] = set()
    for j, amp in enumerate(sv.data):
        if abs(amp) > 1e-9 and ((j >> n_q) & 1) == 1:  # ancilla (qubit n_q) is 1
            marked.add(j & ((1 << n_q) - 1))

    assert marked == expected


def test_quantum_var_matches_normal_quantile() -> None:
    """95% quantum VaR of a standard-normal loss must be near 1.645, and must
    NOT be the bit-reversed, inflated ~2.59 the bug produced."""
    from qufin.options.distributions import normal_distribution
    from qufin.risk.quantum_var import QuantumVaRConfig, quantum_var

    dist = normal_distribution(n_qubits=4, mean=0.0, std=1.0, n_sigma=4.0)
    cfg = QuantumVaRConfig(
        confidence_level=0.95, n_qubits_loss=4, n_bisection_steps=8,
        qae_method="iqae", qae_epsilon=0.02, qae_shots=2048, seed=42,
    )
    res = quantum_var(dist, QiskitAerBackend(seed=42), cfg)

    # True normal 95% quantile is 1.645; discretisation (4 qubits) keeps us
    # within ~0.6. The pre-fix value of ~2.59 is firmly excluded.
    assert abs(res.var_estimate - 1.645) < 0.6
    assert res.var_estimate < 2.0


# ---------------------------------------------------------------------------
# European QAE: invertible state prep + comparator endianness
# ---------------------------------------------------------------------------


def test_european_qae_prices_near_black_scholes() -> None:
    """European call via QAE must build its Grover operator without crashing on
    qiskit >= 1.0 and price within tolerance of Black-Scholes (~10.45)."""
    from qufin.options.amplitude_estimation.european_qae import (
        EuropeanQAESpec,
        build_european_estimation_problem,
    )
    from qufin.options.amplitude_estimation.iqae import (
        IQAEConfig,
        IterativeAmplitudeEstimation,
    )
    from qufin.options.classical.black_scholes import call_price

    spec = EuropeanQAESpec(s0=100, k=100, r=0.05, sigma=0.2, T=1.0, is_call=True, n_qubits=4)
    problem, rescale = build_european_estimation_problem(spec)  # must not raise
    iqae = IterativeAmplitudeEstimation(
        problem, IQAEConfig(epsilon_target=0.01, shots_per_round=2048, seed=42),
        QiskitAerBackend(seed=42),
    )
    price = iqae.estimate().estimate * rescale
    bs = call_price(s=100, k=100, r=0.05, sigma=0.2, T=1.0)

    assert abs(price - bs) < 1.5     # near BS (4-qubit discretisation)
    assert price < 15.0              # excludes the pre-fix ~26


# ---------------------------------------------------------------------------
# QAOA/VQE: lowest-energy selection + variable-order decode
# ---------------------------------------------------------------------------


def _toy_qubo():
    from qufin.portfolio.qubo import PortfolioQUBO

    mu = np.array([0.20, 0.02, 0.18, 0.01])
    cov = np.diag([0.02, 0.05, 0.02, 0.05])
    return PortfolioQUBO(mu=mu, cov=cov, gamma=1.0, cardinality=2, budget_penalty=5.0)


def _exhaustive_optimum(qubo):
    best_s, best_v = None, float("inf")
    for bits in itertools.product("01", repeat=qubo.n_qubits):
        s = "".join(bits)
        v = qubo.evaluate(s)
        if v < best_v:
            best_v, best_s = v, s
    return best_s, best_v


def test_lowest_energy_selection_beats_modal() -> None:
    """Selecting the lowest-energy sampled state (after reversing qiskit keys to
    variable order) must recover the optimum even when it is rare, whereas the
    modal state would be wrong."""
    qubo = _toy_qubo()
    best_s, _ = _exhaustive_optimum(qubo)
    assert best_s == "1010"  # assets 0 and 2

    # qiskit-order counts: the optimum is rare; '0000' is modal.
    counts = {best_s[::-1]: 5, "0000": 900, "1111": 800, "1000": 700}
    picked = min((k[::-1] for k in counts), key=qubo.evaluate)
    assert picked == best_s

    w = qubo.decode_weights(picked)
    assert sorted(k for k, x in enumerate(w) if x > 0) == [0, 2]
    assert w == pytest.approx([0.5, 0.0, 0.5, 0.0])


def test_qaoa_recovers_exhaustive_optimum() -> None:
    """End-to-end QAOA on the toy problem must return the exhaustive optimum
    with capital mapped to the correct assets (regression for both the
    most-frequent and the endianness bugs)."""
    from qufin.portfolio.optimizers.qaoa import QAOAConfig, QAOAPortfolio

    qubo = _toy_qubo()
    _, opt_val = _exhaustive_optimum(qubo)
    cfg = QAOAConfig(p=2, mixer="xy_ring", cardinality=2, maxiter=60, shots=4096, seed=7)
    res = QAOAPortfolio(qubo, cfg, QiskitAerBackend(seed=7)).run()

    assert res.best_objective <= opt_val + 1e-6
    assert sorted(k for k, x in enumerate(res.weights) if x > 0) == [0, 2]
    assert res.feasible


# ---------------------------------------------------------------------------
# QUBO binary decode: bit significance matches the objective build
# ---------------------------------------------------------------------------


def test_binary_decode_significance_matches_build() -> None:
    """decode_weights must read bit b of an asset with significance 2**b, the
    same convention _build_binary uses to build the objective."""
    from qufin.portfolio.qubo import PortfolioQUBO

    # Variable order: within each asset's slice, position b == bit b with
    # significance 2**b (max_level = 2**2 - 1 = 3).
    qubo = PortfolioQUBO(
        mu=np.array([0.05, 0.05]), cov=np.eye(2) * 0.04,
        encoding="binary", bits_per_asset=2,
    )
    # asset0 bits "10" -> level 1 (2**0); asset1 bits "01" -> level 2 (2**1);
    # pre-normalised ratio 1:2.
    w = qubo.decode_weights("1001")
    assert w == pytest.approx([1 / 3, 2 / 3])


def test_pec_mitigate_warns_it_is_approximate() -> None:
    """The mislabeled PEC routine must warn that it is an approximate heuristic,
    not a faithful Temme et al. quasi-probability implementation."""
    from qiskit.circuit import QuantumCircuit

    from qufin.backends.error_mitigation import PECConfig, pec_mitigate
    from qufin.backends.mock import MockBackend

    qc = QuantumCircuit(1)
    qc.h(0)
    with pytest.warns(UserWarning, match="approximate heuristic"):
        pec_mitigate(qc, MockBackend(), PECConfig(n_samples=4))
