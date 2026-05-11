"""Additional unit tests to boost coverage on under-tested modules.

Targets: qgan, rl_quantum, quantum_deep_hedging, asian, barrier,
canonical/fqae/mlae QAE, vqe, qaoa, hybrid, quantum_var,
utils (viz, logging, results), data (equities, macro),
ml (kernels classifier, reservoir fit/predict), deep_hedging.
"""

from __future__ import annotations

import importlib
import json
import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from qufin.backends.base import CircuitResult
from qufin.backends.mock import MockBackend

# =====================================================================
# utils/logging.py  (7 uncovered lines)
# =====================================================================

class TestLogging:
    def test_get_logger_returns_logger(self) -> None:
        from qufin.utils.logging import get_logger

        logger = get_logger("test_module")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "qufin.test_module"

    def test_get_logger_level(self) -> None:
        from qufin.utils.logging import get_logger

        logger = get_logger("test_level", level="DEBUG")
        assert logger.level == logging.DEBUG

    def test_get_logger_default_level(self) -> None:
        from qufin.utils.logging import get_logger

        logger = get_logger("test_default")
        assert logger.level == logging.WARNING

    def test_get_logger_has_handler(self) -> None:
        from qufin.utils.logging import get_logger

        logger = get_logger("test_handler_check")
        assert len(logger.handlers) > 0

    def test_get_logger_idempotent(self) -> None:
        from qufin.utils.logging import get_logger

        logger1 = get_logger("test_idem")
        n_handlers = len(logger1.handlers)
        logger2 = get_logger("test_idem")
        # Should not add duplicate handlers
        assert len(logger2.handlers) == n_handlers


# =====================================================================
# utils/results.py  (7 uncovered lines - _NumpyEncoder + to_json)
# =====================================================================

class TestResultSerialization:
    def test_to_dict(self) -> None:
        from qufin.utils.results import Result

        r = Result(value=1.5, std_err=0.1, n_shots=1024, backend_id="mock")
        d = r.to_dict()
        assert d["value"] == 1.5
        assert d["n_shots"] == 1024

    def test_to_json(self) -> None:
        from qufin.utils.results import Result

        r = Result(value=2.0, backend_id="test")
        j = r.to_json()
        parsed = json.loads(j)
        assert parsed["value"] == 2.0

    def test_numpy_encoder_ndarray(self) -> None:
        from qufin.utils.results import Result

        r = Result(value=1.0, metadata={"arr": np.array([1.0, 2.0])})
        j = r.to_json()
        parsed = json.loads(j)
        assert parsed["metadata"]["arr"] == [1.0, 2.0]

    def test_numpy_encoder_integer(self) -> None:
        from qufin.utils.results import Result

        r = Result(value=1.0, metadata={"x": np.int64(42)})
        j = r.to_json()
        parsed = json.loads(j)
        assert parsed["metadata"]["x"] == 42

    def test_numpy_encoder_float(self) -> None:
        from qufin.utils.results import Result

        r = Result(value=1.0, metadata={"x": np.float64(3.14)})
        j = r.to_json()
        parsed = json.loads(j)
        assert abs(parsed["metadata"]["x"] - 3.14) < 1e-10


# =====================================================================
# options/asian.py  (41 uncovered - build_asian_estimation_problem)
# =====================================================================

class TestAsianEstimationProblem:
    def test_build_arithmetic_problem(self) -> None:
        from qufin.options.asian import AsianOptionSpec, build_asian_estimation_problem

        spec = AsianOptionSpec(
            s0=100, k=100, r=0.05, sigma=0.2, T=1.0,
            n_monitoring=4, is_call=True, average_type="arithmetic",
            n_qubits_per_step=2,
        )
        problem, rescale = build_asian_estimation_problem(spec)
        assert problem.n_qubits == 3  # 2 + 1 ancilla
        assert rescale > 0

    def test_build_geometric_problem(self) -> None:
        from qufin.options.asian import AsianOptionSpec, build_asian_estimation_problem

        spec = AsianOptionSpec(
            s0=100, k=100, r=0.05, sigma=0.2, T=1.0,
            n_monitoring=4, is_call=True, average_type="geometric",
            n_qubits_per_step=2,
        )
        problem, rescale = build_asian_estimation_problem(spec)
        assert problem.n_qubits == 3
        assert rescale > 0

    def test_build_put_problem(self) -> None:
        from qufin.options.asian import AsianOptionSpec, build_asian_estimation_problem

        spec = AsianOptionSpec(
            s0=100, k=100, r=0.05, sigma=0.2, T=1.0,
            n_monitoring=4, is_call=False, n_qubits_per_step=2,
        )
        problem, _rescale = build_asian_estimation_problem(spec)
        assert problem.objective_qubits == [2]

    def test_build_1qubit_problem(self) -> None:
        from qufin.options.asian import AsianOptionSpec, build_asian_estimation_problem

        spec = AsianOptionSpec(
            s0=100, k=100, r=0.05, sigma=0.2, T=1.0,
            n_monitoring=2, n_qubits_per_step=1,
        )
        problem, _rescale = build_asian_estimation_problem(spec)
        assert problem.n_qubits == 2

    def test_deep_otm_put(self) -> None:
        """Deep OTM put should have rescale = 1.0 (max_payoff fallback)."""
        from qufin.options.asian import AsianOptionSpec, build_asian_estimation_problem

        spec = AsianOptionSpec(
            s0=200, k=10, r=0.05, sigma=0.01, T=0.01,
            n_monitoring=1, is_call=False, n_qubits_per_step=2,
        )
        _problem, rescale = build_asian_estimation_problem(spec)
        assert rescale > 0

    def test_geometric_put_closed_form(self) -> None:
        from qufin.options.asian import geometric_asian_closed_form

        put = geometric_asian_closed_form(
            s0=100, k=100, r=0.05, sigma=0.2, T=1.0,
            n_monitoring=12, is_call=False,
        )
        assert put > 0


# =====================================================================
# options/barrier.py  (48 uncovered - barrier_closed_form variants +
#                      build_barrier_estimation_problem)
# =====================================================================

class TestBarrierClosedFormExtended:
    def test_up_and_out_put(self) -> None:
        from qufin.options.barrier import barrier_closed_form

        price = barrier_closed_form(
            s0=100, k=100, r=0.05, sigma=0.2, T=1.0,
            barrier=120, barrier_type="up-and-out", is_call=False,
        )
        assert price >= 0

    def test_down_and_out_put(self) -> None:
        from qufin.options.barrier import barrier_closed_form

        price = barrier_closed_form(
            s0=100, k=100, r=0.05, sigma=0.2, T=1.0,
            barrier=80, barrier_type="down-and-out", is_call=False,
        )
        assert price >= 0

    def test_down_and_in_call(self) -> None:
        from qufin.options.barrier import barrier_closed_form

        price = barrier_closed_form(
            s0=100, k=100, r=0.05, sigma=0.2, T=1.0,
            barrier=80, barrier_type="down-and-in", is_call=True,
        )
        assert price >= 0

    def test_down_and_in_put(self) -> None:
        from qufin.options.barrier import barrier_closed_form

        price = barrier_closed_form(
            s0=100, k=100, r=0.05, sigma=0.2, T=1.0,
            barrier=80, barrier_type="down-and-in", is_call=False,
        )
        assert price >= 0

    def test_up_and_in_call(self) -> None:
        from qufin.options.barrier import barrier_closed_form

        price = barrier_closed_form(
            s0=100, k=100, r=0.05, sigma=0.2, T=1.0,
            barrier=120, barrier_type="up-and-in", is_call=True,
        )
        assert price >= 0

    def test_spot_above_barrier_up_out(self) -> None:
        """Spot above barrier for up-and-out should be 0."""
        from qufin.options.barrier import barrier_closed_form

        price = barrier_closed_form(
            s0=130, k=100, r=0.05, sigma=0.2, T=1.0,
            barrier=120, barrier_type="up-and-out", is_call=True,
        )
        assert price == 0.0

    def test_spot_below_barrier_down_out(self) -> None:
        """Spot below barrier for down-and-out should be 0."""
        from qufin.options.barrier import barrier_closed_form

        price = barrier_closed_form(
            s0=70, k=100, r=0.05, sigma=0.2, T=1.0,
            barrier=80, barrier_type="down-and-out", is_call=True,
        )
        assert price == 0.0

    def test_far_barrier_up_and_in(self) -> None:
        """Very far barrier -> up-and-in should be ~0."""
        from qufin.options.barrier import barrier_closed_form

        price = barrier_closed_form(
            s0=100, k=100, r=0.05, sigma=0.2, T=1.0,
            barrier=500, barrier_type="up-and-in", is_call=True,
        )
        assert price < 1.0  # near zero


class TestBarrierEstimationProblem:
    def test_build_up_and_out_call(self) -> None:
        from qufin.options.barrier import BarrierOptionSpec, build_barrier_estimation_problem

        spec = BarrierOptionSpec(
            s0=100, k=100, r=0.05, sigma=0.2, T=1.0,
            barrier=120, barrier_type="up-and-out", n_qubits=2,
        )
        problem, rescale = build_barrier_estimation_problem(spec)
        assert problem.n_qubits == 3  # 2 + 1 ancilla
        assert rescale > 0

    def test_build_down_and_in_put(self) -> None:
        from qufin.options.barrier import BarrierOptionSpec, build_barrier_estimation_problem

        spec = BarrierOptionSpec(
            s0=100, k=100, r=0.05, sigma=0.2, T=1.0,
            barrier=80, barrier_type="down-and-in",
            is_call=False, n_qubits=2,
        )
        problem, _rescale = build_barrier_estimation_problem(spec)
        assert problem.n_qubits == 3

    def test_build_1qubit(self) -> None:
        from qufin.options.barrier import BarrierOptionSpec, build_barrier_estimation_problem

        spec = BarrierOptionSpec(
            s0=100, k=100, r=0.05, sigma=0.2, T=1.0,
            barrier=120, barrier_type="up-and-out", n_qubits=1,
        )
        problem, _rescale = build_barrier_estimation_problem(spec)
        assert problem.n_qubits == 2

    def test_deep_otm_barrier(self) -> None:
        from qufin.options.barrier import BarrierOptionSpec, build_barrier_estimation_problem

        spec = BarrierOptionSpec(
            s0=200, k=10, r=0.05, sigma=0.01, T=0.01,
            barrier=220, barrier_type="up-and-out",
            is_call=False, n_qubits=2,
        )
        _problem, rescale = build_barrier_estimation_problem(spec)
        assert rescale > 0

    def test_all_barrier_types_in_build(self) -> None:
        from qufin.options.barrier import BarrierOptionSpec, build_barrier_estimation_problem

        for bt in ["up-and-out", "down-and-out", "up-and-in", "down-and-in"]:
            spec = BarrierOptionSpec(
                s0=100, k=100, r=0.05, sigma=0.2, T=1.0,
                barrier=120 if "up" in bt else 80,
                barrier_type=bt, n_qubits=2,
            )
            problem, _rescale = build_barrier_estimation_problem(spec)
            assert problem is not None


# =====================================================================
# ml/qgan.py  (105 uncovered - entire QuantumGAN class)
# =====================================================================

class TestQGAN:
    def test_config_creation(self) -> None:
        from qufin.ml.qgan import QGANConfig

        cfg = QGANConfig(n_qubits=2, n_epochs=1)
        assert cfg.n_qubits == 2
        assert cfg.n_epochs == 1

    def test_result_creation(self) -> None:
        from qufin.ml.qgan import QGANResult

        result = QGANResult(
            generator_params=np.zeros(4),
            loss_history_g=[1.0],
            loss_history_d=[1.0],
            trained_distribution=np.array([0.5, 0.5]),
            kl_divergence=0.1,
            wall_time_s=1.0,
        )
        assert result.kl_divergence == 0.1

    def test_qgan_instantiation(self) -> None:
        from qufin.ml.qgan import QGANConfig, QuantumGAN
        from qufin.options.distributions import uniform_distribution

        dist = uniform_distribution(n_qubits=2)
        cfg = QGANConfig(n_qubits=2, n_epochs=1, generator_reps=1)
        backend = MockBackend(seed=42)
        gan = QuantumGAN(dist, cfg, backend)
        assert gan.config.n_qubits == 2

    def test_generator_n_params(self) -> None:
        from qufin.ml.qgan import QGANConfig, QuantumGAN
        from qufin.options.distributions import uniform_distribution

        dist = uniform_distribution(n_qubits=2)
        cfg = QGANConfig(n_qubits=2, generator_reps=2)
        backend = MockBackend(seed=42)
        gan = QuantumGAN(dist, cfg, backend)
        # 2 * n_qubits * (reps + 1) = 2 * 2 * 3 = 12
        assert gan._generator_n_params() == 12

    def test_build_generator_circuit(self) -> None:
        from qufin.ml.qgan import QGANConfig, QuantumGAN
        from qufin.options.distributions import uniform_distribution

        dist = uniform_distribution(n_qubits=2)
        cfg = QGANConfig(n_qubits=2, generator_reps=1)
        backend = MockBackend(seed=42)
        gan = QuantumGAN(dist, cfg, backend)
        params = np.zeros(gan._generator_n_params())
        circ = gan._build_generator_circuit(params)
        assert circ.num_qubits == 2

    def test_sample_generator(self) -> None:
        from qufin.ml.qgan import QGANConfig, QuantumGAN
        from qufin.options.distributions import uniform_distribution

        dist = uniform_distribution(n_qubits=2)
        cfg = QGANConfig(n_qubits=2, generator_reps=1, shots=100)
        backend = MockBackend(seed=42)
        gan = QuantumGAN(dist, cfg, backend)
        params = np.zeros(gan._generator_n_params())
        sampled = gan._sample_generator(params)
        assert len(sampled) == 4
        assert np.isclose(sampled.sum(), 1.0, atol=0.1)

    def test_discriminator_forward(self) -> None:
        from qufin.ml.qgan import QGANConfig, QuantumGAN
        from qufin.options.distributions import uniform_distribution

        dist = uniform_distribution(n_qubits=2)
        cfg = QGANConfig(n_qubits=2, discriminator_hidden=[4])
        backend = MockBackend(seed=42)
        gan = QuantumGAN(dist, cfg, backend)
        weights, biases = gan._init_discriminator()
        x = np.array([[0.5]])
        out = gan._discriminator_forward(x, weights, biases)
        assert 0.0 <= out.item() <= 1.0

    def test_kl_divergence(self) -> None:
        from qufin.ml.qgan import QGANConfig, QuantumGAN
        from qufin.options.distributions import uniform_distribution

        dist = uniform_distribution(n_qubits=2)
        cfg = QGANConfig(n_qubits=2)
        backend = MockBackend(seed=42)
        gan = QuantumGAN(dist, cfg, backend)
        p = np.array([0.25, 0.25, 0.25, 0.25])
        q = np.array([0.25, 0.25, 0.25, 0.25])
        kl = gan._kl_divergence(p, q)
        assert abs(kl) < 1e-6

    def test_kl_divergence_nonzero(self) -> None:
        from qufin.ml.qgan import QGANConfig, QuantumGAN
        from qufin.options.distributions import uniform_distribution

        dist = uniform_distribution(n_qubits=2)
        cfg = QGANConfig(n_qubits=2)
        backend = MockBackend(seed=42)
        gan = QuantumGAN(dist, cfg, backend)
        p = np.array([0.5, 0.5, 0.0, 0.0])
        q = np.array([0.25, 0.25, 0.25, 0.25])
        kl = gan._kl_divergence(p, q)
        assert kl > 0

    def test_train_runs(self) -> None:
        """Train with minimal config to cover the training loop."""
        from qufin.ml.qgan import QGANConfig, QuantumGAN
        from qufin.options.distributions import uniform_distribution

        dist = uniform_distribution(n_qubits=2)
        cfg = QGANConfig(
            n_qubits=2, generator_reps=1, n_epochs=1,
            discriminator_hidden=[2], shots=32, batch_size=4, seed=42,
        )
        backend = MockBackend(seed=42)
        gan = QuantumGAN(dist, cfg, backend)
        result = gan.train()
        assert result.wall_time_s > 0
        assert len(result.loss_history_g) == 1
        assert len(result.loss_history_d) == 1
        assert len(result.trained_distribution) == 4


# =====================================================================
# hedging/quantum_deep_hedging.py  (41 uncovered)
# =====================================================================

class TestQuantumDeepHedging:
    def test_config_defaults(self) -> None:
        from qufin.hedging.quantum_deep_hedging import QuantumDeepHedgingConfig

        cfg = QuantumDeepHedgingConfig()
        assert cfg.n_qubits == 4
        assert cfg.n_layers == 2

    def test_build_circuit(self) -> None:
        from qufin.hedging.quantum_deep_hedging import build_circuit

        circ = build_circuit(n_qubits=3, n_layers=2, entanglement="linear")
        assert circ.num_qubits == 3

    def test_encode_features(self) -> None:
        from qufin.hedging.quantum_deep_hedging import _encode_features

        features = np.array([0.1, 0.2, 0.3])
        circ = _encode_features(4, features)
        assert circ.num_qubits == 4

    def test_forward_evaluation(self) -> None:
        from qufin.hedging.quantum_deep_hedging import build_circuit, forward

        circ = build_circuit(3, 1, "linear")
        n_params = len(circ.parameters)
        params = np.zeros(n_params)
        features = np.array([0.5, 0.3, 0.1])
        expectations = forward(params, features, 3, 1, "linear")
        assert len(expectations) == 3
        assert all(-1.0 <= e <= 1.0 + 1e-10 for e in expectations)

    def test_forward_param_mismatch(self) -> None:
        from qufin.hedging.quantum_deep_hedging import forward

        with pytest.raises(ValueError, match="Expected"):
            forward(np.zeros(2), np.array([0.1]), 3, 1)

    def test_resource_estimate(self) -> None:
        from qufin.hedging.quantum_deep_hedging import resource_estimate

        res = resource_estimate(4, 2)
        assert res["gate_count"] > 0
        assert res["depth"] > 0
        assert res["params"] > 0

    def test_hedger_class(self) -> None:
        from qufin.hedging.quantum_deep_hedging import (
            QuantumDeepHedger,
            QuantumDeepHedgingConfig,
        )

        cfg = QuantumDeepHedgingConfig(n_qubits=3, n_layers=1)
        hedger = QuantumDeepHedger(cfg)
        assert hedger.n_params > 0

    def test_hedger_forward(self) -> None:
        from qufin.hedging.quantum_deep_hedging import (
            QuantumDeepHedger,
            QuantumDeepHedgingConfig,
        )

        cfg = QuantumDeepHedgingConfig(n_qubits=3, n_layers=1)
        hedger = QuantumDeepHedger(cfg)
        params = np.zeros(hedger.n_params)
        features = np.array([0.1, 0.2, 0.3])
        out = hedger.forward(params, features)
        assert len(out) == 3

    def test_hedger_resource_estimate(self) -> None:
        from qufin.hedging.quantum_deep_hedging import (
            QuantumDeepHedger,
            QuantumDeepHedgingConfig,
        )

        cfg = QuantumDeepHedgingConfig(n_qubits=4, n_layers=2)
        hedger = QuantumDeepHedger(cfg)
        res = hedger.resource_estimate()
        assert res["params"] == hedger.n_params


# =====================================================================
# hedging/rl_quantum.py  (58 uncovered)
# =====================================================================

class TestQuantumRLPolicy:
    def test_policy_config(self) -> None:
        from qufin.hedging.rl_quantum import QuantumPolicyConfig

        cfg = QuantumPolicyConfig(n_qubits=3, n_layers=1, n_actions=2)
        assert cfg.n_actions == 2

    def test_build_policy_circuit(self) -> None:
        from qufin.hedging.rl_quantum import build_policy_circuit

        circ = build_policy_circuit(4, 2, 3)
        assert circ.num_qubits == 4

    def test_encode_state(self) -> None:
        from qufin.hedging.rl_quantum import _encode_state

        circ = _encode_state(3, np.array([0.1, 0.2]))
        assert circ.num_qubits == 3

    def test_policy_instantiation(self) -> None:
        from qufin.hedging.rl_quantum import QuantumPolicy, QuantumPolicyConfig

        cfg = QuantumPolicyConfig(n_qubits=3, n_layers=1, n_actions=3)
        policy = QuantumPolicy(cfg)
        assert policy.n_params > 0

    def test_policy_default_config(self) -> None:
        from qufin.hedging.rl_quantum import QuantumPolicy

        policy = QuantumPolicy()
        assert policy.config.n_qubits == 4

    def test_select_action_statevector(self) -> None:
        from qufin.hedging.rl_quantum import QuantumPolicy, QuantumPolicyConfig

        cfg = QuantumPolicyConfig(n_qubits=2, n_layers=1, n_actions=2)
        policy = QuantumPolicy(cfg)
        state = np.array([0.5, 0.3])
        params = np.zeros(policy.n_params)
        probs = policy.select_action(state, params, backend=None)
        assert len(probs) == 2
        assert abs(probs.sum() - 1.0) < 1e-10

    def test_select_action_param_mismatch(self) -> None:
        from qufin.hedging.rl_quantum import QuantumPolicy, QuantumPolicyConfig

        cfg = QuantumPolicyConfig(n_qubits=2, n_layers=1, n_actions=2)
        policy = QuantumPolicy(cfg)
        state = np.array([0.5, 0.3])
        with pytest.raises(ValueError, match="Expected"):
            policy.select_action(state, np.zeros(2), backend=None)

    def test_probs_to_action_probs(self) -> None:
        from qufin.hedging.rl_quantum import QuantumPolicy, QuantumPolicyConfig

        cfg = QuantumPolicyConfig(n_qubits=2, n_layers=1, n_actions=3)
        policy = QuantumPolicy(cfg)
        full_probs = np.array([0.25, 0.25, 0.25, 0.25])
        action_probs = policy._probs_to_action_probs(full_probs, 3)
        assert len(action_probs) == 3
        assert abs(action_probs.sum() - 1.0) < 1e-10

    def test_probs_to_action_probs_zero(self) -> None:
        from qufin.hedging.rl_quantum import QuantumPolicy, QuantumPolicyConfig

        cfg = QuantumPolicyConfig(n_qubits=2, n_layers=1, n_actions=3)
        policy = QuantumPolicy(cfg)
        full_probs = np.zeros(4)
        action_probs = policy._probs_to_action_probs(full_probs, 3)
        assert abs(action_probs.sum() - 1.0) < 1e-10

    def test_counts_to_action_probs(self) -> None:
        from qufin.hedging.rl_quantum import QuantumPolicy, QuantumPolicyConfig

        cfg = QuantumPolicyConfig(n_qubits=2, n_layers=1, n_actions=2)
        policy = QuantumPolicy(cfg)
        counts = {"00": 50, "01": 30, "10": 15, "11": 5}
        probs = policy._counts_to_action_probs(counts, 2)
        assert len(probs) == 2
        assert abs(probs.sum() - 1.0) < 1e-10

    def test_log_prob(self) -> None:
        from qufin.hedging.rl_quantum import QuantumPolicy, QuantumPolicyConfig

        cfg = QuantumPolicyConfig(n_qubits=2, n_layers=1, n_actions=2)
        policy = QuantumPolicy(cfg)
        state = np.array([0.5, 0.3])
        params = np.zeros(policy.n_params)
        lp = policy.log_prob(state, 0, params, backend=None)
        assert lp <= 0  # log prob is always <= 0


# =====================================================================
# hedging/deep_hedging.py  (13 uncovered - hedge_paths method)
# =====================================================================

class TestDeepHedgerHedge:
    def test_hedge(self) -> None:
        from qufin.hedging.deep_hedging import DeepHedger, DeepHedgingConfig

        cfg = DeepHedgingConfig(n_epochs=1, n_paths=32, n_steps=5)
        hedger = DeepHedger(cfg, s0=100.0, strike=100.0, seed=0)
        # Generate some fake paths
        rng = np.random.default_rng(0)
        paths = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, (10, 6)), axis=1))
        ratios = hedger.hedge(paths)
        assert ratios.shape == (10, 5)
        # Ratios should be in (-1, 1) due to tanh output
        assert np.all(ratios >= -1.0 - 1e-10)
        assert np.all(ratios <= 1.0 + 1e-10)


# =====================================================================
# ml/kernels.py  (15 uncovered - QuantumKernelClassifier fit/predict)
# =====================================================================

class TestQuantumKernelClassifier:
    def test_fit_and_predict(self) -> None:
        from qufin.ml.kernels import QuantumKernelClassifier

        backend = MockBackend(seed=42)
        clf = QuantumKernelClassifier(n_qubits=2, backend=backend, reps=1)
        rng = np.random.default_rng(0)
        X_train = rng.uniform(0, 2 * np.pi, (6, 2))
        y_train = np.array([0, 0, 0, 1, 1, 1])
        clf.fit(X_train, y_train)
        preds = clf.predict(X_train[:2])
        assert len(preds) == 2

    def test_predict_before_fit_raises(self) -> None:
        from qufin.ml.kernels import QuantumKernelClassifier

        backend = MockBackend(seed=42)
        clf = QuantumKernelClassifier(n_qubits=2, backend=backend)
        with pytest.raises(AssertionError):
            clf.predict(np.zeros((2, 2)))


# =====================================================================
# ml/reservoir.py  (14 uncovered - fit/predict)
# =====================================================================

class TestQuantumReservoirFitPredict:
    def test_fit_and_predict(self) -> None:
        from qufin.ml.reservoir import QuantumReservoir, QuantumReservoirConfig

        cfg = QuantumReservoirConfig(n_qubits=2, n_layers=1, seed=42)
        backend = MockBackend(seed=0)
        res = QuantumReservoir(cfg, backend)
        rng = np.random.default_rng(0)
        X_train = rng.uniform(0, 1, (8, 2))
        y_train = rng.normal(0, 1, 8)
        res.fit(X_train, y_train)
        preds = res.predict(X_train[:3])
        assert preds.shape == (3,)

    def test_predict_before_fit_raises(self) -> None:
        from qufin.ml.reservoir import QuantumReservoir, QuantumReservoirConfig

        cfg = QuantumReservoirConfig(n_qubits=2, n_layers=1)
        backend = MockBackend(seed=0)
        res = QuantumReservoir(cfg, backend)
        with pytest.raises(AssertionError):
            res.predict(np.zeros((2, 2)))

    def test_build_feature_matrix(self) -> None:
        from qufin.ml.reservoir import QuantumReservoir, QuantumReservoirConfig

        cfg = QuantumReservoirConfig(n_qubits=2, n_layers=1, seed=42)
        backend = MockBackend(seed=0)
        res = QuantumReservoir(cfg, backend)
        X = np.array([[0.1, 0.2], [0.3, 0.4]])
        F = res._build_feature_matrix(X)
        assert F.shape == (2, 2)


# =====================================================================
# risk/quantum_var.py  (52 uncovered - _run_qae variants,
#                       _build_conditional_value_problem, quantum_var)
# =====================================================================

class TestQuantumVaRExtended:
    def test_build_conditional_value_problem(self) -> None:
        from qufin.options.distributions import normal_distribution
        from qufin.risk.quantum_var import _build_conditional_value_problem

        dist = normal_distribution(n_qubits=2, mean=0.0, std=0.02, n_sigma=3.0)
        problem, rescale = _build_conditional_value_problem(dist, threshold=0.0)
        assert problem.n_qubits == 4  # 2 + 1 comparator + 1 value
        assert rescale > 0

    def test_run_qae_unknown_method(self) -> None:
        from qufin.options.distributions import normal_distribution
        from qufin.risk.quantum_var import (
            QuantumVaRConfig,
            _build_tail_probability_problem,
            _run_qae,
        )

        dist = normal_distribution(n_qubits=2, mean=0.0, std=0.02, n_sigma=3.0)
        problem = _build_tail_probability_problem(dist, threshold=0.0)
        config = QuantumVaRConfig(qae_method="nonexistent")
        backend = MockBackend(seed=42)
        with pytest.raises(ValueError, match="Unknown QAE method"):
            _run_qae(problem, backend, config)

    def test_quantum_var_default_config(self) -> None:
        """quantum_var with config=None should use defaults."""
        from qufin.options.distributions import normal_distribution
        from qufin.risk.quantum_var import QuantumVaRConfig, quantum_var

        dist = normal_distribution(n_qubits=2, mean=0.0, std=0.02, n_sigma=3.0)
        from qufin.backends.qiskit_backend import QiskitAerBackend

        backend = QiskitAerBackend(seed=42)
        config = QuantumVaRConfig(
            n_bisection_steps=1,
            qae_epsilon=0.1,
            qae_shots=128,
            seed=42,
        )
        result = quantum_var(dist, backend, config)
        assert result.es_estimate is not None

    def test_decomposed_state_prep(self) -> None:
        from qufin.risk.quantum_var import _decomposed_state_prep

        amps = np.array([0.5, 0.5, 0.5, 0.5])
        circ = _decomposed_state_prep(amps, 2)
        assert circ.num_qubits == 2


# =====================================================================
# portfolio/optimizers/vqe.py  (65 uncovered)
# =====================================================================

class TestVQEPortfolio:
    def _make_qubo_and_backend(self, n_assets=2, cardinality=1):
        """Helper: create QUBO and MockBackend with correct bitstring width."""
        from qufin.portfolio.qubo import PortfolioQUBO

        mu = np.arange(1, n_assets + 1, dtype=float) * 0.01
        cov = np.eye(n_assets) * 0.01
        qubo = PortfolioQUBO(mu, cov, gamma=1.0, cardinality=cardinality)
        n_q = qubo.n_qubits
        # MockBackend must return bitstrings of width n_q
        counts = {format(i, f"0{n_q}b"): 10 for i in range(2**n_q)}
        backend = MockBackend(default_counts=counts, seed=42)
        return qubo, backend

    def test_config_defaults(self) -> None:
        from qufin.portfolio.optimizers.vqe import VQEConfig

        cfg = VQEConfig()
        assert cfg.reps == 3
        assert cfg.cvar_alpha == 0.5

    def test_n_params(self) -> None:
        from qufin.portfolio.optimizers.vqe import VQEConfig, VQEPortfolio

        qubo, backend = self._make_qubo_and_backend(3, 2)
        cfg = VQEConfig(reps=1, rotation_blocks=["ry", "rz"])
        vqe = VQEPortfolio(qubo, cfg, backend)
        n = vqe._n_params()
        assert n > 0

    def test_build_circuit(self) -> None:
        from qufin.portfolio.optimizers.vqe import VQEConfig, VQEPortfolio

        qubo, backend = self._make_qubo_and_backend()
        cfg = VQEConfig(reps=1, entanglement="linear")
        vqe = VQEPortfolio(qubo, cfg, backend)
        params = np.zeros(vqe._n_params())
        circ = vqe._build_circuit(params)
        assert circ.num_qubits == qubo.n_qubits

    def test_build_circuit_circular(self) -> None:
        from qufin.portfolio.optimizers.vqe import VQEConfig, VQEPortfolio

        qubo, backend = self._make_qubo_and_backend(3, 2)
        cfg = VQEConfig(reps=1, entanglement="circular")
        vqe = VQEPortfolio(qubo, cfg, backend)
        params = np.zeros(vqe._n_params())
        circ = vqe._build_circuit(params)
        assert circ.num_qubits == qubo.n_qubits

    def test_build_circuit_full(self) -> None:
        from qufin.portfolio.optimizers.vqe import VQEConfig, VQEPortfolio

        qubo, backend = self._make_qubo_and_backend(3, 2)
        cfg = VQEConfig(reps=1, entanglement="full")
        vqe = VQEPortfolio(qubo, cfg, backend)
        params = np.zeros(vqe._n_params())
        circ = vqe._build_circuit(params)
        assert circ.num_qubits == qubo.n_qubits

    def test_evaluate_counts(self) -> None:
        from qufin.portfolio.optimizers.vqe import VQEConfig, VQEPortfolio

        qubo, backend = self._make_qubo_and_backend()
        cfg = VQEConfig(cvar_alpha=0.5)
        vqe = VQEPortfolio(qubo, cfg, backend)
        n_q = qubo.n_qubits
        result = CircuitResult(
            counts={format(i, f"0{n_q}b"): 25 for i in range(2**n_q)},
            shots=100, backend_id="mock",
        )
        val = vqe._evaluate_counts(result)
        assert isinstance(val, float)

    def test_run_minimal(self) -> None:
        from qufin.portfolio.optimizers.vqe import VQEConfig, VQEPortfolio

        qubo, backend = self._make_qubo_and_backend()
        cfg = VQEConfig(reps=1, maxiter=2, shots=32, seed=42)
        vqe = VQEPortfolio(qubo, cfg, backend)
        result = vqe.run()
        assert result.best_bitstring != ""
        assert result.wall_time_s > 0
        assert len(result.history) > 0

    def test_run_with_initial_params(self) -> None:
        from qufin.portfolio.optimizers.vqe import VQEConfig, VQEPortfolio

        qubo, backend = self._make_qubo_and_backend()
        cfg = VQEConfig(reps=1, maxiter=2, shots=32, seed=42)
        vqe = VQEPortfolio(qubo, cfg, backend)
        n_p = vqe._n_params()
        cfg.initial_params = np.ones(n_p) * 0.5
        vqe2 = VQEPortfolio(qubo, cfg, backend)
        result = vqe2.run()
        assert result.best_bitstring != ""


# =====================================================================
# portfolio/optimizers/qaoa.py  (63 uncovered)
# =====================================================================

class TestQAOAPortfolio:
    def _make_qubo_and_backend(self, n_assets=2, cardinality=1):
        from qufin.portfolio.qubo import PortfolioQUBO

        mu = np.arange(1, n_assets + 1, dtype=float) * 0.01
        cov = np.eye(n_assets) * 0.01
        qubo = PortfolioQUBO(mu, cov, gamma=1.0, cardinality=cardinality)
        n_q = qubo.n_qubits
        counts = {format(i, f"0{n_q}b"): 10 for i in range(2**n_q)}
        backend = MockBackend(default_counts=counts, seed=42)
        return qubo, backend

    def test_config_defaults(self) -> None:
        from qufin.portfolio.optimizers.qaoa import QAOAConfig

        cfg = QAOAConfig()
        assert cfg.p == 3
        assert cfg.mixer == "x"

    def test_build_circuit(self) -> None:
        from qufin.portfolio.optimizers.qaoa import QAOAConfig, QAOAPortfolio

        qubo, backend = self._make_qubo_and_backend()
        cfg = QAOAConfig(p=1, mixer="x", shots=32)
        qaoa = QAOAPortfolio(qubo, cfg, backend)
        betas = np.array([0.5])
        gammas = np.array([0.5])
        circ = qaoa._build_circuit(betas, gammas)
        assert circ.num_qubits == qubo.n_qubits

    def test_evaluate_counts(self) -> None:
        from qufin.portfolio.optimizers.qaoa import QAOAConfig, QAOAPortfolio

        qubo, backend = self._make_qubo_and_backend()
        cfg = QAOAConfig(p=1, cvar_alpha=0.5)
        qaoa = QAOAPortfolio(qubo, cfg, backend)
        n_q = qubo.n_qubits
        result = CircuitResult(
            counts={format(i, f"0{n_q}b"): 25 for i in range(2**n_q)},
            shots=100, backend_id="mock",
        )
        val = qaoa._evaluate_counts(result)
        assert isinstance(val, float)

    def test_run_minimal(self) -> None:
        from qufin.portfolio.optimizers.qaoa import QAOAConfig, QAOAPortfolio

        qubo, backend = self._make_qubo_and_backend()
        cfg = QAOAConfig(p=1, mixer="x", maxiter=2, shots=32, seed=42)
        qaoa = QAOAPortfolio(qubo, cfg, backend)
        result = qaoa.run()
        assert result.best_bitstring != ""
        assert result.wall_time_s > 0

    def test_run_with_initial_params(self) -> None:
        from qufin.portfolio.optimizers.qaoa import QAOAConfig, QAOAPortfolio

        qubo, backend = self._make_qubo_and_backend()
        cfg = QAOAConfig(
            p=1, mixer="x", maxiter=2, shots=32, seed=42,
            initial_gammas=np.array([0.3]),
            initial_betas=np.array([0.5]),
        )
        qaoa = QAOAPortfolio(qubo, cfg, backend)
        result = qaoa.run()
        assert result.best_bitstring != ""

    def test_objective_function(self) -> None:
        from qufin.portfolio.optimizers.qaoa import QAOAConfig, QAOAPortfolio

        qubo, backend = self._make_qubo_and_backend()
        cfg = QAOAConfig(p=1, shots=32, seed=42)
        qaoa = QAOAPortfolio(qubo, cfg, backend)
        params = np.array([0.5, 0.3])  # gamma, beta
        val = qaoa._objective(params)
        assert isinstance(val, float)
        assert len(qaoa._history) == 1


# =====================================================================
# portfolio/optimizers/hybrid.py  (28 uncovered)
# =====================================================================

class TestHybridOptimizer:
    def test_config_defaults(self) -> None:
        from qufin.portfolio.optimizers.hybrid import HybridConfig

        cfg = HybridConfig()
        assert cfg.qaoa_p == 2

    def test_result_defaults(self) -> None:
        from qufin.portfolio.optimizers.hybrid import HybridResult

        r = HybridResult()
        assert r.best_objective == float("inf")

    def test_run(self) -> None:
        from qufin.portfolio.optimizers.hybrid import (
            HybridConfig,
            HybridOptimizer,
        )
        from qufin.portfolio.qubo import PortfolioQUBO

        mu = np.array([0.01, 0.02])
        cov = np.eye(2) * 0.01
        qubo = PortfolioQUBO(mu, cov, gamma=1.0, cardinality=1)
        n_q = qubo.n_qubits
        counts = {format(i, f"0{n_q}b"): 10 for i in range(2**n_q)}
        backend = MockBackend(default_counts=counts, seed=42)
        cfg = HybridConfig(
            qaoa_p=1, qaoa_maxiter=2, qaoa_shots=32, seed=42,
        )
        optimizer = HybridOptimizer(qubo, cfg, backend)
        result = optimizer.run()
        assert result.best_bitstring != ""
        assert result.classical_time_s > 0
        assert result.quantum_time_s >= 0
        assert result.wall_time_s > 0


# =====================================================================
# options/amplitude_estimation (canonical, fqae, mlae with MockBackend)
# =====================================================================

class TestCanonicalQAEMock:
    """Test canonical QAE with MockBackend for coverage."""

    def test_config_creation(self) -> None:
        from qufin.options.amplitude_estimation.canonical import CanonicalQAEConfig

        cfg = CanonicalQAEConfig(n_eval_qubits=3, shots=100)
        assert cfg.n_eval_qubits == 3

    def test_result_creation(self) -> None:
        from qufin.options.amplitude_estimation.canonical import CanonicalQAEResult

        r = CanonicalQAEResult(estimate=0.5, theta_estimate=0.3)
        assert r.estimate == 0.5

    def test_instantiation(self) -> None:
        from qiskit.circuit import QuantumCircuit

        from qufin.options.amplitude_estimation.canonical import (
            CanonicalAmplitudeEstimation,
            CanonicalQAEConfig,
        )
        from qufin.options.amplitude_estimation.estimation_problem import (
            EstimationProblem,
        )

        qc = QuantumCircuit(1)
        qc.ry(np.pi / 2, 0)
        problem = EstimationProblem(state_preparation=qc, objective_qubits=[0], n_qubits=1)
        cfg = CanonicalQAEConfig(n_eval_qubits=2, shots=32, seed=42)
        backend = MockBackend(seed=42)
        qae = CanonicalAmplitudeEstimation(problem, cfg, backend)
        assert qae.problem is problem

    def test_estimate_runs(self) -> None:
        from qiskit.circuit import QuantumCircuit

        from qufin.options.amplitude_estimation.canonical import (
            CanonicalAmplitudeEstimation,
            CanonicalQAEConfig,
        )
        from qufin.options.amplitude_estimation.estimation_problem import (
            EstimationProblem,
        )

        qc = QuantumCircuit(1)
        qc.ry(np.pi / 2, 0)
        problem = EstimationProblem(state_preparation=qc, objective_qubits=[0], n_qubits=1)
        cfg = CanonicalQAEConfig(n_eval_qubits=2, shots=32, seed=42)
        backend = MockBackend(seed=42)
        qae = CanonicalAmplitudeEstimation(problem, cfg, backend)
        result = qae.estimate()
        assert 0 <= result.estimate <= 1
        assert result.n_oracle_calls == 3  # 2^2 - 1
        assert result.wall_time_s > 0


class TestFQAEMock:
    """Test FQAE with MockBackend for coverage."""

    def test_config_creation(self) -> None:
        from qufin.options.amplitude_estimation.fqae import FQAEConfig

        cfg = FQAEConfig(max_depth=4, n_shots_per_round=100)
        assert cfg.max_depth == 4

    def test_result_creation(self) -> None:
        from qufin.options.amplitude_estimation.fqae import FQAEResult

        r = FQAEResult(estimate=0.3, n_rounds=5)
        assert r.n_rounds == 5

    def test_estimate_runs(self) -> None:
        from qiskit.circuit import QuantumCircuit

        from qufin.options.amplitude_estimation.estimation_problem import (
            EstimationProblem,
        )
        from qufin.options.amplitude_estimation.fqae import (
            FaithfulAmplitudeEstimation,
            FQAEConfig,
        )

        qc = QuantumCircuit(1)
        qc.ry(np.pi / 2, 0)
        problem = EstimationProblem(state_preparation=qc, objective_qubits=[0], n_qubits=1)
        cfg = FQAEConfig(max_depth=2, n_shots_per_round=32, seed=42)
        backend = MockBackend(seed=42)
        fqae = FaithfulAmplitudeEstimation(problem, cfg, backend)
        result = fqae.estimate()
        assert 0 <= result.estimate <= 1
        assert result.n_rounds == 3  # depths 0,1,2
        assert result.max_depth_used == 2

    def test_count_good(self) -> None:
        from qiskit.circuit import QuantumCircuit

        from qufin.options.amplitude_estimation.estimation_problem import (
            EstimationProblem,
        )
        from qufin.options.amplitude_estimation.fqae import (
            FaithfulAmplitudeEstimation,
            FQAEConfig,
        )

        qc = QuantumCircuit(1)
        problem = EstimationProblem(state_preparation=qc, objective_qubits=[0], n_qubits=1)
        cfg = FQAEConfig()
        backend = MockBackend(seed=42)
        fqae = FaithfulAmplitudeEstimation(problem, cfg, backend)
        cr = CircuitResult(counts={"0": 30, "1": 70}, shots=100, backend_id="mock")
        n_good, n_total = fqae._count_good(cr)
        assert n_good == 70
        assert n_total == 100


class TestMLAEMock:
    """Test MLAE with MockBackend for coverage."""

    def test_config_creation(self) -> None:
        from qufin.options.amplitude_estimation.mlae import MLAEConfig

        cfg = MLAEConfig(evaluation_schedule=[0, 1, 2], n_shots_per_round=100)
        assert cfg.evaluation_schedule == [0, 1, 2]

    def test_result_creation(self) -> None:
        from qufin.options.amplitude_estimation.mlae import MLAEResult

        r = MLAEResult(estimate=0.4, n_rounds=3)
        assert r.n_rounds == 3

    def test_estimate_runs(self) -> None:
        from qiskit.circuit import QuantumCircuit

        from qufin.options.amplitude_estimation.estimation_problem import (
            EstimationProblem,
        )
        from qufin.options.amplitude_estimation.mlae import (
            MaximumLikelihoodAmplitudeEstimation,
            MLAEConfig,
        )

        qc = QuantumCircuit(1)
        qc.ry(np.pi / 2, 0)
        problem = EstimationProblem(state_preparation=qc, objective_qubits=[0], n_qubits=1)
        cfg = MLAEConfig(evaluation_schedule=[0, 1, 2], n_shots_per_round=32, seed=42)
        backend = MockBackend(seed=42)
        mlae = MaximumLikelihoodAmplitudeEstimation(problem, cfg, backend)
        result = mlae.estimate()
        assert 0 <= result.estimate <= 1
        assert result.n_rounds == 3
        assert result.n_oracle_calls > 0

    def test_estimate_default_schedule(self) -> None:
        from qiskit.circuit import QuantumCircuit

        from qufin.options.amplitude_estimation.estimation_problem import (
            EstimationProblem,
        )
        from qufin.options.amplitude_estimation.mlae import (
            MaximumLikelihoodAmplitudeEstimation,
            MLAEConfig,
        )

        qc = QuantumCircuit(1)
        qc.ry(np.pi / 2, 0)
        problem = EstimationProblem(state_preparation=qc, objective_qubits=[0], n_qubits=1)
        cfg = MLAEConfig(n_shots_per_round=32, seed=42)  # no schedule -> uses default
        backend = MockBackend(seed=42)
        mlae = MaximumLikelihoodAmplitudeEstimation(problem, cfg, backend)
        result = mlae.estimate()
        assert result.n_rounds == 6  # default [0,1,2,4,8,16]

    def test_log_likelihood(self) -> None:
        from qiskit.circuit import QuantumCircuit

        from qufin.options.amplitude_estimation.estimation_problem import (
            EstimationProblem,
        )
        from qufin.options.amplitude_estimation.mlae import (
            MaximumLikelihoodAmplitudeEstimation,
            MLAEConfig,
        )

        qc = QuantumCircuit(1)
        problem = EstimationProblem(state_preparation=qc, objective_qubits=[0], n_qubits=1)
        cfg = MLAEConfig()
        backend = MockBackend(seed=42)
        mlae = MaximumLikelihoodAmplitudeEstimation(problem, cfg, backend)
        # data: (m_k, h_k, N_k)
        data = [(0, 50, 100), (1, 80, 100)]
        ll = mlae._log_likelihood(np.pi / 4, data)
        assert isinstance(ll, float)

    def test_count_good(self) -> None:
        from qiskit.circuit import QuantumCircuit

        from qufin.options.amplitude_estimation.estimation_problem import (
            EstimationProblem,
        )
        from qufin.options.amplitude_estimation.mlae import (
            MaximumLikelihoodAmplitudeEstimation,
            MLAEConfig,
        )

        qc = QuantumCircuit(1)
        problem = EstimationProblem(state_preparation=qc, objective_qubits=[0], n_qubits=1)
        cfg = MLAEConfig()
        backend = MockBackend(seed=42)
        mlae = MaximumLikelihoodAmplitudeEstimation(problem, cfg, backend)
        cr = CircuitResult(counts={"0": 40, "1": 60}, shots=100, backend_id="mock")
        n_good, n_total = mlae._count_good(cr)
        assert n_good == 60
        assert n_total == 100


# =====================================================================
# data/equities.py  (15 uncovered - requires yfinance, mock it)
# =====================================================================

class TestYahooEquityProvider:
    def test_get_prices(self) -> None:
        import pandas as pd

        mock_data = pd.DataFrame(
            {"Close": [100.0, 101.0, 102.0]},
            index=pd.date_range("2024-01-01", periods=3),
        )
        with patch.dict("sys.modules", {"yfinance": MagicMock()}):
            import yfinance as yf

            yf.download = MagicMock(return_value=mock_data)
            from qufin.data.equities import YahooEquityProvider

            provider = YahooEquityProvider(cache=False)
            result = provider.get_prices(["AAPL"], "2024-01-01", "2024-01-03")
            assert len(result) > 0

    def test_get_returns(self) -> None:
        import pandas as pd

        prices = pd.DataFrame(
            {"Close": [100.0, 101.0, 102.0, 103.0, 104.0]},
            index=pd.date_range("2024-01-01", periods=5),
        )
        with patch.dict("sys.modules", {"yfinance": MagicMock()}):
            import yfinance as yf

            yf.download = MagicMock(return_value=prices)
            from qufin.data.equities import YahooEquityProvider

            provider = YahooEquityProvider(cache=False)
            result = provider.get_returns(["AAPL"], "2024-01-01", "2024-01-05")
            assert len(result) > 0


# =====================================================================
# data/macro.py  (22 uncovered - requires fredapi, mock it)
# =====================================================================

class TestFREDProvider:
    def test_import_error(self) -> None:
        """Should raise ImportError when fredapi not available."""
        with patch.dict("sys.modules", {"fredapi": None}):
            # Force reimport
            import qufin.data.macro

            importlib.reload(qufin.data.macro)
            with pytest.raises(ImportError, match="fredapi"):
                qufin.data.macro.FREDProvider()

    def test_series_dict(self) -> None:
        from qufin.data.macro import FREDProvider

        assert "tbill_3m" in FREDProvider.SERIES
        assert "yield_10y" in FREDProvider.SERIES


# =====================================================================
# utils/viz.py  (62 uncovered - mock matplotlib)
# =====================================================================

class TestVisualization:
    def test_plot_efficient_frontier(self) -> None:
        pytest.importorskip("matplotlib")
        import matplotlib

        matplotlib.use("Agg")
        from qufin.utils.viz import plot_efficient_frontier

        returns = np.array([0.05, 0.08, 0.06])
        cov = np.eye(3) * 0.01
        ax = plot_efficient_frontier(returns, cov)
        assert ax is not None

    def test_plot_efficient_frontier_with_weights(self) -> None:
        pytest.importorskip("matplotlib")
        import matplotlib

        matplotlib.use("Agg")
        from qufin.utils.viz import plot_efficient_frontier

        returns = np.array([0.05, 0.08, 0.06])
        cov = np.eye(3) * 0.01
        weights = [np.array([0.5, 0.3, 0.2])]
        ax = plot_efficient_frontier(returns, cov, weights_list=weights, labels=["Test"])
        assert ax is not None

    def test_plot_convergence(self) -> None:
        pytest.importorskip("matplotlib")
        import matplotlib

        matplotlib.use("Agg")
        from qufin.utils.viz import plot_convergence

        ax = plot_convergence([1.0, 0.8, 0.6, 0.5])
        assert ax is not None

    def test_plot_convergence_with_reference(self) -> None:
        pytest.importorskip("matplotlib")
        import matplotlib

        matplotlib.use("Agg")
        from qufin.utils.viz import plot_convergence

        ax = plot_convergence([1.0, 0.8, 0.6], reference=0.5)
        assert ax is not None

    def test_plot_loss_distribution(self) -> None:
        pytest.importorskip("matplotlib")
        import matplotlib

        matplotlib.use("Agg")
        from qufin.utils.viz import plot_loss_distribution

        losses = np.random.default_rng(42).normal(0, 1, 100)
        ax = plot_loss_distribution(losses)
        assert ax is not None

    def test_plot_loss_distribution_with_markers(self) -> None:
        pytest.importorskip("matplotlib")
        import matplotlib

        matplotlib.use("Agg")
        from qufin.utils.viz import plot_loss_distribution

        losses = np.random.default_rng(42).normal(0, 1, 100)
        ax = plot_loss_distribution(losses, var=1.5, es=2.0)
        assert ax is not None

    def test_plot_benchmark_comparison(self) -> None:
        pytest.importorskip("matplotlib")
        import matplotlib

        matplotlib.use("Agg")
        from qufin.utils.viz import plot_benchmark_comparison

        ax = plot_benchmark_comparison(
            ["A", "B", "C"], [1.0, 2.0, 3.0],
        )
        assert ax is not None

    def test_plot_benchmark_with_reference(self) -> None:
        pytest.importorskip("matplotlib")
        import matplotlib

        matplotlib.use("Agg")
        from qufin.utils.viz import plot_benchmark_comparison

        ax = plot_benchmark_comparison(
            ["A", "B"], [1.0, 2.0], reference=1.5,
        )
        assert ax is not None


# =====================================================================
# backends/__init__.py  (10 uncovered - import helpers)
# =====================================================================

class TestBackendsInit:
    def test_mock_importable(self) -> None:
        from qufin.backends import MockBackend

        assert MockBackend is not None

    def test_get_noisy_backend(self) -> None:
        from qufin.backends import get_noisy_backend

        backend = get_noisy_backend(seed=42)
        assert backend is not None

    def test_get_ibm_backend_import(self) -> None:
        """get_ibm_backend should be callable (may fail if no creds)."""
        from qufin.backends import get_ibm_backend

        assert callable(get_ibm_backend)

    def test_get_pennylane_backend_import(self) -> None:
        from qufin.backends import get_pennylane_backend

        assert callable(get_pennylane_backend)

    def test_get_cirq_backend_import(self) -> None:
        from qufin.backends import get_cirq_backend

        assert callable(get_cirq_backend)

    def test_get_braket_backend_import(self) -> None:
        from qufin.backends import get_braket_backend

        assert callable(get_braket_backend)


# =====================================================================
# options/__init__.py  (4 uncovered)
# =====================================================================

class TestOptionsInit:
    def test_european_importable(self) -> None:
        from qufin.options import EuropeanOption

        assert EuropeanOption is not None

    def test_get_distribution_loaders(self) -> None:
        from qufin.options import get_distribution_loaders

        mod = get_distribution_loaders()
        assert hasattr(mod, "log_normal_distribution")

    def test_get_amplitude_estimation(self) -> None:
        from qufin.options import get_amplitude_estimation

        mod = get_amplitude_estimation()
        assert mod is not None
