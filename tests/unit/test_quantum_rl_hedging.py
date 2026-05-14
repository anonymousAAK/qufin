"""Unit tests for qufin.hedging.quantum_rl_hedging module."""

from __future__ import annotations

import numpy as np
import pytest

from qufin.hedging.quantum_rl_hedging import (
    ClassicalHedgingPolicy,
    DynamicsType,
    HedgingEnvironment,
    HedgingEvalResult,
    HestonDynamicsConfig,
    QuantumHedgingPolicy,
    QuantumRLHedger,
    QuantumRLHedgingConfig,
    RewardType,
    TrainingResult,
    _bs_delta,
    _bs_gamma,
    _bs_vega,
    _encode_market_state,
    _parameter_shift_gradient,
    build_vqc_policy,
    compare_policies,
    compute_reward,
    evaluate_policy,
    evaluate_vqc_policy,
    simulate_gbm_paths,
    simulate_heston_paths,
    train_quantum_ppo,
)

# ---------------------------------------------------------------------------
# Config and dataclass tests
# ---------------------------------------------------------------------------

class TestQuantumRLHedgingConfig:
    """Tests for QuantumRLHedgingConfig dataclass."""

    def test_default_creation(self) -> None:
        cfg = QuantumRLHedgingConfig()
        assert cfg.n_qubits == 4
        assert cfg.n_layers == 2
        assert cfg.dynamics == DynamicsType.GBM
        assert cfg.reward_type == RewardType.VARIANCE
        assert cfg.transaction_cost == 0.001
        assert cfg.clip_epsilon == 0.2

    def test_custom_creation(self) -> None:
        cfg = QuantumRLHedgingConfig(
            n_qubits=6, n_layers=3, dynamics=DynamicsType.HESTON,
            reward_type=RewardType.CVAR, n_episodes=10,
        )
        assert cfg.n_qubits == 6
        assert cfg.n_layers == 3
        assert cfg.dynamics == DynamicsType.HESTON
        assert cfg.reward_type == RewardType.CVAR


class TestHestonDynamicsConfig:
    """Tests for HestonDynamicsConfig dataclass."""

    def test_defaults(self) -> None:
        hc = HestonDynamicsConfig()
        assert hc.v0 == 0.04
        assert hc.kappa == 2.0
        assert hc.rho == -0.7


class TestTrainingResult:
    """Tests for TrainingResult dataclass."""

    def test_empty_creation(self) -> None:
        tr = TrainingResult()
        assert tr.episode_rewards == []
        assert tr.episode_losses == []
        assert len(tr.final_params) == 0


class TestEnums:
    """Tests for RewardType and DynamicsType enums."""

    def test_reward_types(self) -> None:
        assert RewardType.VARIANCE.value == "variance"
        assert RewardType.CVAR.value == "cvar"

    def test_dynamics_types(self) -> None:
        assert DynamicsType.GBM.value == "gbm"
        assert DynamicsType.HESTON.value == "heston"


# ---------------------------------------------------------------------------
# Path simulation tests
# ---------------------------------------------------------------------------

class TestSimulateGBMPaths:
    """Tests for GBM path simulation."""

    def test_output_shape(self) -> None:
        rng = np.random.default_rng(42)
        paths = simulate_gbm_paths(100.0, 0.05, 0.2, 1.0, 30, 100, rng)
        assert paths.shape == (100, 31)

    def test_initial_price(self) -> None:
        rng = np.random.default_rng(42)
        paths = simulate_gbm_paths(100.0, 0.05, 0.2, 1.0, 10, 50, rng)
        np.testing.assert_allclose(paths[:, 0], 100.0)

    def test_positive_prices(self) -> None:
        rng = np.random.default_rng(42)
        paths = simulate_gbm_paths(100.0, 0.05, 0.2, 1.0, 30, 200, rng)
        assert np.all(paths > 0)


class TestSimulateHestonPaths:
    """Tests for Heston path simulation."""

    def test_output_shapes(self) -> None:
        rng = np.random.default_rng(42)
        hc = HestonDynamicsConfig()
        spot, var_ = simulate_heston_paths(
            100.0, 0.05, 1.0, 20, 50, hc, rng,
        )
        assert spot.shape == (50, 21)
        assert var_.shape == (50, 21)

    def test_variance_non_negative(self) -> None:
        rng = np.random.default_rng(42)
        hc = HestonDynamicsConfig()
        _, var_ = simulate_heston_paths(
            100.0, 0.05, 1.0, 50, 100, hc, rng,
        )
        assert np.all(var_ >= 0)


# ---------------------------------------------------------------------------
# Greeks tests
# ---------------------------------------------------------------------------

class TestGreeks:
    """Tests for Black-Scholes Greeks helpers."""

    def test_delta_in_range(self) -> None:
        spot = np.array([80.0, 100.0, 120.0])
        d = _bs_delta(spot, 100.0, 0.05, 0.2, 1.0)
        assert np.all(d >= 0) and np.all(d <= 1)

    def test_delta_at_expiry(self) -> None:
        spot = np.array([90.0, 110.0])
        d = _bs_delta(spot, 100.0, 0.05, 0.2, 0.0)
        np.testing.assert_array_equal(d, [0.0, 1.0])

    def test_gamma_non_negative(self) -> None:
        spot = np.array([80.0, 100.0, 120.0])
        g = _bs_gamma(spot, 100.0, 0.05, 0.2, 1.0)
        assert np.all(g >= 0)

    def test_gamma_at_expiry(self) -> None:
        spot = np.array([100.0])
        g = _bs_gamma(spot, 100.0, 0.05, 0.2, 0.0)
        np.testing.assert_array_equal(g, [0.0])

    def test_vega_non_negative(self) -> None:
        spot = np.array([80.0, 100.0, 120.0])
        v = _bs_vega(spot, 100.0, 0.05, 0.2, 1.0)
        assert np.all(v >= 0)

    def test_vega_at_expiry(self) -> None:
        spot = np.array([100.0])
        v = _bs_vega(spot, 100.0, 0.05, 0.2, 0.0)
        np.testing.assert_array_equal(v, [0.0])


# ---------------------------------------------------------------------------
# Reward computation tests
# ---------------------------------------------------------------------------

class TestComputeReward:
    """Tests for compute_reward function."""

    def test_variance_reward_negative(self) -> None:
        pnl = np.array([1.0, -1.0, 0.5, -0.5])
        r = compute_reward(pnl, RewardType.VARIANCE)
        assert r <= 0  # negative variance

    def test_zero_variance_pnl(self) -> None:
        pnl = np.ones(10) * 5.0
        r = compute_reward(pnl, RewardType.VARIANCE)
        assert r == pytest.approx(0.0)

    def test_cvar_reward(self) -> None:
        pnl = np.array([-10, -5, 0, 5, 10, 15, 20, 25, 30, 35])
        r = compute_reward(pnl, RewardType.CVAR, alpha=0.1)
        # Worst 10% = first element = -10
        assert r == pytest.approx(-10.0)


# ---------------------------------------------------------------------------
# VQC policy circuit tests
# ---------------------------------------------------------------------------

class TestBuildVQCPolicy:
    """Tests for VQC policy circuit builder."""

    def test_returns_tuple(self) -> None:
        _enc, ansatz, n_params = build_vqc_policy(4, 2)
        assert n_params > 0
        assert ansatz is not None

    def test_param_count_scales_with_qubits(self) -> None:
        _, _, p4 = build_vqc_policy(4, 2)
        _, _, p6 = build_vqc_policy(6, 2)
        assert p6 > p4


class TestEncodeMarketState:
    """Tests for market state encoding circuit."""

    def test_circuit_has_correct_qubits(self) -> None:
        features = np.array([0.5, 0.01, 10.0, 1.0])
        qc = _encode_market_state(4, features)
        assert qc.num_qubits == 4

    def test_encodes_up_to_4_features(self) -> None:
        features = np.array([0.5, 0.01, 10.0, 1.0, 99.0])
        qc = _encode_market_state(4, features)
        # Should only encode first 4
        assert qc.num_qubits == 4


class TestEvaluateVQCPolicy:
    """Tests for VQC policy evaluation."""

    def test_output_in_unit_interval(self) -> None:
        _, _ansatz, n_params = build_vqc_policy(4, 1)
        params = np.zeros(n_params)
        features = np.array([0.5, 0.01, 10.0, 1.0])
        ratio = evaluate_vqc_policy(params, features, 4, 1)
        assert 0.0 <= ratio <= 1.0

    def test_different_params_give_different_ratios(self) -> None:
        _, _, n_params = build_vqc_policy(4, 1)
        features = np.array([0.5, 0.01, 10.0, 1.0])
        r1 = evaluate_vqc_policy(np.zeros(n_params), features, 4, 1)
        r2 = evaluate_vqc_policy(np.ones(n_params), features, 4, 1)
        # Very unlikely to be exactly equal with different params
        assert r1 != pytest.approx(r2, abs=1e-6)

    def test_wrong_param_count_raises(self) -> None:
        with pytest.raises(ValueError, match="Expected"):
            evaluate_vqc_policy(np.array([1.0]), np.array([0.5]), 4, 1)


# ---------------------------------------------------------------------------
# Environment tests
# ---------------------------------------------------------------------------

class TestHedgingEnvironment:
    """Tests for the hedging environment."""

    def test_simulate_gbm_paths(self) -> None:
        cfg = QuantumRLHedgingConfig(
            n_steps=5, n_paths_per_episode=10, seed=42,
        )
        env = HedgingEnvironment(cfg)
        paths, var_paths = env.simulate_paths()
        assert paths.shape == (10, 6)
        assert var_paths is None

    def test_simulate_heston_paths(self) -> None:
        cfg = QuantumRLHedgingConfig(
            dynamics=DynamicsType.HESTON,
            n_steps=5, n_paths_per_episode=10, seed=42,
        )
        hc = HestonDynamicsConfig()
        env = HedgingEnvironment(cfg, hc)
        paths, var_paths = env.simulate_paths()
        assert paths.shape == (10, 6)
        assert var_paths is not None
        assert var_paths.shape == (10, 6)

    def test_compute_features_shape(self) -> None:
        cfg = QuantumRLHedgingConfig(seed=42)
        env = HedgingEnvironment(cfg)
        spot = np.array([100.0, 105.0, 95.0])
        features = env.compute_features(
            spot, tau=0.5,
            position=np.zeros(3),
            portfolio_value=np.zeros(3),
        )
        assert features.shape == (3, 4)

    def test_run_episode_shapes(self) -> None:
        cfg = QuantumRLHedgingConfig(
            n_steps=5, n_paths_per_episode=10, seed=42,
        )
        env = HedgingEnvironment(cfg)

        def dummy_policy(features: np.ndarray) -> np.ndarray:
            return np.full(features.shape[0], 0.5)

        pnl, costs, ratios = env.run_episode(dummy_policy)
        assert pnl.shape == (10,)
        assert costs.shape == (10,)
        assert ratios.shape == (10, 5)

    def test_transaction_costs_positive(self) -> None:
        cfg = QuantumRLHedgingConfig(
            n_steps=5, n_paths_per_episode=20,
            transaction_cost=0.01, seed=42,
        )
        env = HedgingEnvironment(cfg)

        def varying_policy(features: np.ndarray) -> np.ndarray:
            # Alternate between 0.3 and 0.7 to force trades
            return np.where(features[:, 0] > 0.5, 0.7, 0.3)

        _, costs, _ = env.run_episode(varying_policy)
        assert np.all(costs >= 0)
        assert np.mean(costs) > 0


# ---------------------------------------------------------------------------
# Policy tests
# ---------------------------------------------------------------------------

class TestQuantumHedgingPolicy:
    """Tests for the quantum hedging policy."""

    def test_instantiation(self) -> None:
        cfg = QuantumRLHedgingConfig(n_qubits=4, n_layers=1, seed=42)
        policy = QuantumHedgingPolicy(cfg)
        assert policy.n_params > 0
        assert len(policy.params) == policy.n_params

    def test_call_returns_ratios(self) -> None:
        cfg = QuantumRLHedgingConfig(n_qubits=4, n_layers=1, seed=42)
        policy = QuantumHedgingPolicy(cfg)
        features = np.array([[0.5, 0.01, 10.0, 1.0], [0.6, 0.02, 8.0, 0.9]])
        ratios = policy(features)
        assert ratios.shape == (2,)
        assert np.all(ratios >= 0) and np.all(ratios <= 1)

    def test_get_hedge_ratio_single(self) -> None:
        cfg = QuantumRLHedgingConfig(n_qubits=4, n_layers=1, seed=42)
        policy = QuantumHedgingPolicy(cfg)
        features = np.array([0.5, 0.01, 10.0, 1.0])
        ratio = policy.get_hedge_ratio(features)
        assert 0.0 <= ratio <= 1.0


class TestClassicalHedgingPolicy:
    """Tests for the classical MLP baseline."""

    def test_instantiation(self) -> None:
        policy = ClassicalHedgingPolicy(hidden_dim=16, n_layers=2, seed=42)
        assert len(policy.params) == 3  # 2 hidden + 1 output layer

    def test_call_returns_ratios_in_0_1(self) -> None:
        policy = ClassicalHedgingPolicy(seed=42)
        features = np.random.default_rng(0).standard_normal((10, 4))
        ratios = policy(features)
        assert ratios.shape == (10,)
        assert np.all(ratios >= 0) and np.all(ratios <= 1)


# ---------------------------------------------------------------------------
# Training tests (mocked for speed)
# ---------------------------------------------------------------------------

class TestTrainQuantumPPO:
    """Tests for the PPO training loop."""

    def test_training_returns_result(self) -> None:
        cfg = QuantumRLHedgingConfig(
            n_qubits=4, n_layers=1,
            n_episodes=2, n_paths_per_episode=8, n_steps=3,
            seed=42,
        )
        result = train_quantum_ppo(cfg, max_grad_samples=2)
        assert isinstance(result, TrainingResult)
        assert len(result.episode_rewards) == 2
        assert len(result.episode_losses) == 2
        assert len(result.final_params) > 0

    def test_training_with_cvar_reward(self) -> None:
        cfg = QuantumRLHedgingConfig(
            n_qubits=4, n_layers=1,
            reward_type=RewardType.CVAR,
            n_episodes=1, n_paths_per_episode=8, n_steps=3,
            seed=42,
        )
        result = train_quantum_ppo(cfg, max_grad_samples=2)
        assert len(result.episode_rewards) == 1


# ---------------------------------------------------------------------------
# Evaluation tests
# ---------------------------------------------------------------------------

class TestEvaluatePolicy:
    """Tests for evaluate_policy function."""

    def test_returns_eval_result(self) -> None:
        cfg = QuantumRLHedgingConfig(
            n_steps=3, n_paths_per_episode=20, seed=42,
        )

        def dummy(features: np.ndarray) -> np.ndarray:
            return np.full(features.shape[0], 0.5)

        result = evaluate_policy(dummy, cfg, n_eval_paths=20)
        assert isinstance(result, HedgingEvalResult)
        assert result.pnl_distribution.shape == (20,)
        assert result.hedge_ratios.shape == (20, 3)
        assert result.var_95 is not None
        assert result.cvar_95 is not None

    def test_zero_cost_with_no_transaction_cost(self) -> None:
        cfg = QuantumRLHedgingConfig(
            n_steps=3, n_paths_per_episode=20,
            transaction_cost=0.0, seed=42,
        )

        def const_policy(features: np.ndarray) -> np.ndarray:
            return np.full(features.shape[0], 0.5)

        result = evaluate_policy(const_policy, cfg, n_eval_paths=20)
        # With constant policy, transaction costs after first step are 0
        # but first step has a trade from 0 -> 0.5, which with tc=0 is free
        assert result.total_transaction_costs == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# QuantumRLHedger end-to-end tests
# ---------------------------------------------------------------------------

class TestQuantumRLHedger:
    """Tests for the QuantumRLHedger convenience class."""

    def test_instantiation_default(self) -> None:
        hedger = QuantumRLHedger()
        assert hedger.config is not None
        assert hedger.policy is not None

    def test_instantiation_custom(self) -> None:
        cfg = QuantumRLHedgingConfig(n_qubits=4, n_layers=1, n_episodes=1)
        hedger = QuantumRLHedger(cfg)
        assert hedger.config.n_qubits == 4

    def test_hedge_single_sample(self) -> None:
        cfg = QuantumRLHedgingConfig(n_qubits=4, n_layers=1, seed=42)
        hedger = QuantumRLHedger(cfg)
        features = np.array([0.5, 0.01, 10.0, 1.0])
        ratios = hedger.hedge(features)
        assert ratios.shape == (1,)
        assert 0.0 <= ratios[0] <= 1.0

    def test_hedge_batch(self) -> None:
        cfg = QuantumRLHedgingConfig(n_qubits=4, n_layers=1, seed=42)
        hedger = QuantumRLHedger(cfg)
        features = np.array([
            [0.5, 0.01, 10.0, 1.0],
            [0.6, 0.02, 8.0, 0.9],
        ])
        ratios = hedger.hedge(features)
        assert ratios.shape == (2,)

    def test_train_and_evaluate(self) -> None:
        cfg = QuantumRLHedgingConfig(
            n_qubits=4, n_layers=1,
            n_episodes=1, n_paths_per_episode=8, n_steps=3,
            seed=42,
        )
        hedger = QuantumRLHedger(cfg)
        tr = hedger.train(max_grad_samples=2)
        assert isinstance(tr, TrainingResult)
        assert hedger.training_result is not None

        result = hedger.evaluate(n_eval_paths=8)
        assert isinstance(result, HedgingEvalResult)


class TestCompare:
    """Tests for compare_policies function."""

    def test_compare_returns_both(self) -> None:
        cfg = QuantumRLHedgingConfig(
            n_steps=3, n_paths_per_episode=10, seed=42,
            n_qubits=4, n_layers=1,
        )
        results = compare_policies(cfg, n_eval_paths=10)
        assert "quantum" in results
        assert "classical" in results
        assert isinstance(results["quantum"], HedgingEvalResult)
        assert isinstance(results["classical"], HedgingEvalResult)


# ---------------------------------------------------------------------------
# Parameter shift gradient test
# ---------------------------------------------------------------------------

class TestParameterShiftGradient:
    """Tests for the parameter-shift gradient estimator."""

    def test_gradient_shape(self) -> None:
        _, _, n_params = build_vqc_policy(4, 1)
        params = np.zeros(n_params)
        features = np.array([[0.5, 0.01, 10.0, 1.0]])
        grad = _parameter_shift_gradient(params, features, 4, 1, "linear")
        assert grad.shape == (n_params,)

    def test_gradient_not_all_zero(self) -> None:
        _, _, n_params = build_vqc_policy(4, 1)
        rng = np.random.default_rng(42)
        params = rng.uniform(-np.pi, np.pi, n_params)
        features = rng.standard_normal((2, 4))
        grad = _parameter_shift_gradient(params, features, 4, 1, "linear")
        # With random params, gradient should not be identically zero
        assert np.any(np.abs(grad) > 1e-10)
