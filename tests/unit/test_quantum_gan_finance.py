"""Unit tests for qufin.ml.quantum_gan_finance (HQGAN)."""

from __future__ import annotations

import numpy as np
import pytest

from qufin.backends.mock import MockBackend
from qufin.ml.quantum_gan_finance import (
    HQGAN,
    HQGANConfig,
    HQGANResult,
    StylizedFactsResult,
    _arch_lm_test,
    _chi2_sf,
    _erf,
    _NumpyDiscriminator,
    build_generator_circuit,
    evaluate_stylized_facts,
    generator_n_params,
    privacy_preserving_synthetic,
    train_on_synthetic_validate_on_real,
)

# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

@pytest.fixture
def backend() -> MockBackend:
    """MockBackend with 4-qubit default counts."""
    counts = {f"{i:04b}": 64 for i in range(16)}
    return MockBackend(default_counts=counts, seed=42)


@pytest.fixture
def small_config() -> HQGANConfig:
    """Minimal config for fast tests."""
    return HQGANConfig(
        n_qubits=4,
        generator_reps=1,
        latent_dim=4,
        discriminator_hidden=[8, 4, 2],
        n_epochs=2,
        batch_size=4,
        lr_generator=1e-3,
        lr_discriminator=1e-3,
        n_critic=1,
        gradient_penalty_lambda=1.0,
        shots=64,
        seed=42,
        window_size=5,
        convergence_window=3,
        convergence_threshold=1e-4,
    )


@pytest.fixture
def fake_returns() -> np.ndarray:
    """Synthetic return series for testing."""
    rng = np.random.default_rng(123)
    return rng.normal(0.0005, 0.02, size=200)


# -----------------------------------------------------------------------
# 1. HQGANConfig dataclass
# -----------------------------------------------------------------------

class TestHQGANConfig:
    def test_defaults(self) -> None:
        cfg = HQGANConfig()
        assert cfg.n_qubits == 4
        assert cfg.generator_reps == 3
        assert cfg.discriminator_hidden == [128, 64, 32]
        assert cfg.n_epochs == 200
        assert cfg.gradient_penalty_lambda == 10.0

    def test_custom_values(self) -> None:
        cfg = HQGANConfig(n_qubits=8, n_epochs=50)
        assert cfg.n_qubits == 8
        assert cfg.n_epochs == 50


# -----------------------------------------------------------------------
# 2. StylizedFactsResult dataclass
# -----------------------------------------------------------------------

class TestStylizedFactsResult:
    def test_defaults(self) -> None:
        r = StylizedFactsResult()
        assert r.kurtosis == 0.0
        assert r.fat_tails is False
        assert r.volatility_clustering is False
        assert r.leverage_effect is False
        assert r.autocorrelation_present is False


# -----------------------------------------------------------------------
# 3. Generator circuit builder
# -----------------------------------------------------------------------

class TestGeneratorCircuit:
    def test_n_params(self) -> None:
        assert generator_n_params(4, 1) == 16  # 2*4*(1+1)
        assert generator_n_params(4, 3) == 32  # 2*4*(3+1)
        assert generator_n_params(8, 2) == 48  # 2*8*(2+1)

    def test_build_circuit_returns_qiskit_circuit(self) -> None:
        pytest.importorskip("qiskit")
        params = np.zeros(generator_n_params(4, 1))
        qc = build_generator_circuit(params, n_qubits=4, reps=1)
        assert qc.num_qubits == 4
        assert qc.num_clbits == 4

    def test_build_circuit_with_latent(self) -> None:
        pytest.importorskip("qiskit")
        params = np.zeros(generator_n_params(4, 1))
        latent = np.array([0.1, 0.2, 0.3, 0.4])
        qc = build_generator_circuit(params, n_qubits=4, reps=1, latent=latent)
        assert qc.num_qubits == 4

    def test_circuit_8_qubits(self) -> None:
        pytest.importorskip("qiskit")
        n = 8
        reps = 2
        params = np.random.default_rng(0).uniform(0, 2 * np.pi, generator_n_params(n, reps))
        qc = build_generator_circuit(params, n_qubits=n, reps=reps)
        assert qc.num_qubits == 8


# -----------------------------------------------------------------------
# 4. NumpyDiscriminator
# -----------------------------------------------------------------------

class TestNumpyDiscriminator:
    def test_forward_shape(self) -> None:
        rng = np.random.default_rng(0)
        disc = _NumpyDiscriminator(input_dim=5, hidden=[8, 4, 2], rng=rng)
        x = rng.normal(size=(10, 5))
        out = disc.forward(x)
        assert out.shape == (10, 1)

    def test_forward_1d_input(self) -> None:
        rng = np.random.default_rng(0)
        disc = _NumpyDiscriminator(input_dim=1, hidden=[4], rng=rng)
        x = rng.normal(size=5)  # 5 scalar samples
        out = disc.forward(x)
        assert out.shape == (5, 1)

    def test_get_set_params_roundtrip(self) -> None:
        rng = np.random.default_rng(0)
        disc = _NumpyDiscriminator(input_dim=3, hidden=[4, 2], rng=rng)
        params = disc.get_params()
        # 3 layers: [w0(3,4), b0(4), w1(4,2), b1(2), w2(2,1), b2(1)]
        assert len(params) == 6

        # Modify and set back
        orig = [p.copy() for p in params]
        params[0] = params[0] + 1.0
        disc.set_params(params)
        new_params = disc.get_params()
        np.testing.assert_allclose(new_params[0], orig[0] + 1.0)


# -----------------------------------------------------------------------
# 5. Stylised facts evaluation
# -----------------------------------------------------------------------

class TestEvaluateStylizedFacts:
    def test_short_series_returns_defaults(self) -> None:
        result = evaluate_stylized_facts(np.array([0.01, 0.02]))
        assert result.kurtosis == 0.0

    def test_fat_tails_gaussian(self) -> None:
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.01, 10000)
        result = evaluate_stylized_facts(returns)
        # Gaussian kurtosis ~3, so fat_tails may or may not be True
        assert isinstance(result.kurtosis, float)
        assert isinstance(result.fat_tails, bool)

    def test_fat_tails_t_distribution(self) -> None:
        rng = np.random.default_rng(42)
        # t-distribution with df=3 has excess kurtosis
        returns = rng.standard_t(df=3, size=10000)
        result = evaluate_stylized_facts(returns)
        assert result.kurtosis > 3.0
        assert result.fat_tails is True

    def test_leverage_corr_type(self) -> None:
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.01, 500)
        result = evaluate_stylized_facts(returns)
        assert isinstance(result.leverage_corr, float)

    def test_abs_return_autocorr_type(self) -> None:
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.01, 500)
        result = evaluate_stylized_facts(returns)
        assert isinstance(result.abs_return_autocorr, float)

    def test_constant_series(self) -> None:
        """Constant returns should not crash."""
        returns = np.ones(100) * 0.01
        result = evaluate_stylized_facts(returns)
        assert result.kurtosis == 0.0


# -----------------------------------------------------------------------
# 6. ARCH LM test
# -----------------------------------------------------------------------

class TestArchLMTest:
    def test_short_input(self) -> None:
        stat, pval = _arch_lm_test(np.array([1.0, 2.0]), lags=5)
        assert stat == 0.0
        assert pval == 1.0

    def test_returns_tuple(self) -> None:
        rng = np.random.default_rng(0)
        resid_sq = rng.normal(0, 1, 200) ** 2
        stat, pval = _arch_lm_test(resid_sq, lags=5)
        assert isinstance(stat, float)
        assert 0.0 <= pval <= 1.0


# -----------------------------------------------------------------------
# 7. Chi-squared SF & erf
# -----------------------------------------------------------------------

class TestStatHelpers:
    def test_chi2_sf_zero(self) -> None:
        assert _chi2_sf(0.0, 5) == 1.0

    def test_chi2_sf_large(self) -> None:
        # Very large x should give p ~ 0
        p = _chi2_sf(1000.0, 5)
        assert p < 0.01

    def test_erf_zero(self) -> None:
        assert abs(_erf(0.0)) < 1e-10

    def test_erf_large(self) -> None:
        assert abs(_erf(5.0) - 1.0) < 1e-5


# -----------------------------------------------------------------------
# 8. Wasserstein loss helpers
# -----------------------------------------------------------------------

class TestWassersteinLoss:
    def test_critic_loss(self) -> None:
        real = np.array([1.0, 2.0, 3.0])
        fake = np.array([0.5, 1.0, 1.5])
        loss = HQGAN.wasserstein_loss_critic(real, fake)
        expected = np.mean(fake) - np.mean(real)
        assert abs(loss - expected) < 1e-10

    def test_generator_loss(self) -> None:
        fake = np.array([1.0, 2.0, 3.0])
        loss = HQGAN.wasserstein_loss_generator(fake)
        assert abs(loss - (-2.0)) < 1e-10

    def test_gradient_penalty_nonnegative(self) -> None:
        rng = np.random.default_rng(42)
        disc = _NumpyDiscriminator(input_dim=5, hidden=[4], rng=rng)
        real = rng.normal(size=(8, 5))
        fake = rng.normal(size=(8, 5))
        gp = HQGAN.gradient_penalty(disc, real, fake, rng)
        assert gp >= 0.0


# -----------------------------------------------------------------------
# 9. HQGAN construction
# -----------------------------------------------------------------------

class TestHQGANConstruction:
    def test_init(self, small_config: HQGANConfig, backend: MockBackend) -> None:
        hqgan = HQGAN(small_config, backend)
        assert hqgan.config is small_config
        assert hqgan._n_params == generator_n_params(
            small_config.n_qubits, small_config.generator_reps
        )

    def test_sample_generator_shape(
        self, small_config: HQGANConfig, backend: MockBackend
    ) -> None:
        hqgan = HQGAN(small_config, backend)
        params = np.zeros(hqgan._n_params)
        samples = hqgan._sample_generator(params, 50)
        assert len(samples) == 50
        assert np.all(samples >= 0.0)
        assert np.all(samples <= 1.0)


# -----------------------------------------------------------------------
# 10. HQGAN training (short run)
# -----------------------------------------------------------------------

class TestHQGANTraining:
    def test_train_returns_result(
        self,
        small_config: HQGANConfig,
        backend: MockBackend,
        fake_returns: np.ndarray,
    ) -> None:
        hqgan = HQGAN(small_config, backend)
        result = hqgan.train(fake_returns)

        assert isinstance(result, HQGANResult)
        assert len(result.loss_history_g) > 0
        assert len(result.loss_history_d) > 0
        assert len(result.wasserstein_estimates) > 0
        assert len(result.synthetic_data) == len(fake_returns)
        assert isinstance(result.stylized_facts, StylizedFactsResult)
        assert result.wall_time_s > 0

    def test_generate_denormalises(
        self, small_config: HQGANConfig, backend: MockBackend
    ) -> None:
        hqgan = HQGAN(small_config, backend)
        params = np.zeros(hqgan._n_params)
        data = hqgan.generate(params, 100, real_min=-0.05, real_max=0.05)
        assert len(data) == 100
        # Should be in the denormalised range (approximately)
        assert data.min() >= -0.1
        assert data.max() <= 0.15


# -----------------------------------------------------------------------
# 11. Privacy-preserving synthetic helper
# -----------------------------------------------------------------------

class TestPrivacyPreservingSynthetic:
    def test_returns_tuple(
        self,
        small_config: HQGANConfig,
        backend: MockBackend,
        fake_returns: np.ndarray,
    ) -> None:
        synthetic, result = privacy_preserving_synthetic(
            fake_returns, small_config, backend, n_synthetic=50
        )
        assert len(synthetic) == 50
        assert isinstance(result, HQGANResult)

    def test_default_n_synthetic(
        self,
        small_config: HQGANConfig,
        backend: MockBackend,
        fake_returns: np.ndarray,
    ) -> None:
        synthetic, _ = privacy_preserving_synthetic(
            fake_returns, small_config, backend
        )
        assert len(synthetic) == len(fake_returns)


# -----------------------------------------------------------------------
# 12. Train-on-synthetic, validate-on-real
# -----------------------------------------------------------------------

class TestTrainOnSyntheticValidateOnReal:
    def test_returns_metrics(self, fake_returns: np.ndarray) -> None:
        rng = np.random.default_rng(0)
        synthetic = rng.normal(0.0005, 0.02, size=200)
        metrics = train_on_synthetic_validate_on_real(fake_returns, synthetic)

        assert "train_mse" in metrics
        assert "test_mse" in metrics
        assert "train_mae" in metrics
        assert "test_mae" in metrics
        assert "synthetic_mean" in metrics
        assert "real_mean" in metrics
        assert metrics["train_mse"] >= 0
        assert metrics["test_mse"] >= 0

    def test_short_synthetic(self) -> None:
        """Very short series should not crash."""
        real = np.array([0.01, 0.02, 0.03])
        syn = np.array([0.01, 0.02])
        metrics = train_on_synthetic_validate_on_real(real, syn)
        # With train_fraction=0.8, n_train=max(int(2*0.8),2)=2 -> only 2 points
        assert isinstance(metrics["synthetic_std"], float)

    def test_single_point_real(self) -> None:
        """Single real data point should produce NaN test metrics."""
        real = np.array([0.01])
        syn = np.random.default_rng(0).normal(0, 0.01, 50)
        metrics = train_on_synthetic_validate_on_real(real, syn)
        assert np.isnan(metrics["test_mse"])


# -----------------------------------------------------------------------
# 13. HQGANResult dataclass
# -----------------------------------------------------------------------

class TestHQGANResult:
    def test_fields(self) -> None:
        r = HQGANResult(
            generator_params=np.zeros(10),
            loss_history_g=[1.0, 0.5],
            loss_history_d=[2.0, 1.0],
            wasserstein_estimates=[0.5, 0.3],
            synthetic_data=np.zeros(50),
            stylized_facts=StylizedFactsResult(),
            wall_time_s=1.23,
            converged=False,
        )
        assert len(r.generator_params) == 10
        assert r.converged is False
        assert r.wall_time_s == 1.23
