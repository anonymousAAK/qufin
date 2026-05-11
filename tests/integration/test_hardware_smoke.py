"""Hardware smoke tests for IBM Quantum backends.

These tests require:
1. A valid IBM Quantum account (qiskit-ibm-runtime configured)
2. The --hardware pytest flag

Run with:
    pytest tests/integration/test_hardware_smoke.py --hardware -v

Skipped by default in CI and local runs without credentials.
"""

from __future__ import annotations

import numpy as np
import pytest


hardware = pytest.mark.skipif(
    "not config.getoption('--hardware', default=False)",
    reason="Hardware tests require --hardware flag",
)


def pytest_addoption(parser):
    """Add --hardware CLI flag to pytest."""
    try:
        parser.addoption("--hardware", action="store_true", default=False,
                         help="Run tests on IBM Quantum hardware")
    except ValueError:
        pass  # Already added


@hardware
class TestIBMHardwareSmoke:
    """Smoke tests on IBM Quantum hardware (real device)."""

    @pytest.fixture(scope="class")
    def ibm_backend(self):
        """Connect to IBM backend. Skips if credentials unavailable."""
        try:
            from qufin.backends.ibm_runtime import IBMRuntimeBackend
            backend = IBMRuntimeBackend(
                backend_name="ibm_brisbane",
                optimization_level=1,
                resilience_level=1,
            )
            return backend
        except Exception as e:
            pytest.skip(f"IBM backend unavailable: {e}")

    def test_bell_state(self, ibm_backend) -> None:
        """Create a Bell state and verify ~50/50 correlated outcomes."""
        from qiskit.circuit import QuantumCircuit

        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])

        result = ibm_backend.run(qc, shots=4096)
        p00 = result.counts.get("00", 0) / 4096
        p11 = result.counts.get("11", 0) / 4096

        # Bell state: should be ~50% |00> + ~50% |11>
        assert p00 > 0.3, f"P(00) = {p00:.3f}, expected ~0.5"
        assert p11 > 0.3, f"P(11) = {p11:.3f}, expected ~0.5"
        assert p00 + p11 > 0.8, f"Correlated fraction = {p00+p11:.3f}"

    def test_qaoa_4_qubit(self, ibm_backend) -> None:
        """Run 4-qubit QAOA portfolio optimization on hardware."""
        from qufin.portfolio.optimizers.qaoa import QAOAConfig, QAOAPortfolio
        from qufin.portfolio.qubo import PortfolioQUBO

        rng = np.random.default_rng(42)
        mu = rng.normal(0, 0.01, 4)
        cov = np.eye(4) * 0.02
        qubo = PortfolioQUBO(mu, cov, gamma=1.0, cardinality=2)

        config = QAOAConfig(p=1, mixer="x", shots=4096, maxiter=10, seed=42)
        qaoa = QAOAPortfolio(qubo, config, ibm_backend)
        result = qaoa.run()

        # Just check it runs without error and returns valid results
        assert result.best_bitstring is not None
        assert len(result.best_bitstring) == 4
        assert result.wall_time_s > 0

    def test_iqae_simple(self, ibm_backend) -> None:
        """Run IQAE for a known amplitude on hardware."""
        from qiskit.circuit import QuantumCircuit
        from qufin.options.amplitude_estimation.estimation_problem import EstimationProblem
        from qufin.options.amplitude_estimation.iqae import IQAEConfig, IterativeAmplitudeEstimation

        qc = QuantumCircuit(1)
        qc.ry(2 * np.arcsin(np.sqrt(0.25)), 0)
        problem = EstimationProblem(state_preparation=qc, objective_qubits=[0])

        config = IQAEConfig(epsilon_target=0.05, shots_per_round=2048)
        qae = IterativeAmplitudeEstimation(problem, config, ibm_backend)
        result = qae.estimate()

        # On real hardware with noise, wider tolerance
        assert 0.05 < result.estimate < 0.50, (
            f"IQAE on hardware: {result.estimate:.4f}, expected ~0.25"
        )

    def test_backend_metadata(self, ibm_backend) -> None:
        """Verify backend reports correct metadata."""
        assert "ibm-runtime" in ibm_backend.backend_id
        assert ibm_backend.num_qubits >= 100  # Eagle/Heron have 100+ qubits
        assert not ibm_backend.is_simulator()
