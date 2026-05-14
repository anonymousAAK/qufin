"""Unit tests for qufin.options.implied_vol_surface."""

from __future__ import annotations

import numpy as np
import pytest

from qufin.backends.mock import MockBackend
from qufin.options.implied_vol_surface import (
    IVSurfaceData,
    QuantumIVSurface,
    QuantumIVSurfaceConfig,
    SABRModel,
    SurfaceMetrics,
    SVIModel,
    evaluate_surface,
    generate_synthetic_iv_surface,
)

# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

@pytest.fixture
def backend() -> MockBackend:
    return MockBackend(seed=42)


@pytest.fixture
def synth_data() -> IVSurfaceData:
    """Small synthetic IV surface for fast tests."""
    return generate_synthetic_iv_surface(
        n_strikes=6, n_expiries=4, seed=42, noise=0.002,
    )


@pytest.fixture
def train_test_split(synth_data: IVSurfaceData):
    """Split synthetic data 80/20."""
    n = len(synth_data.strikes)
    rng = np.random.default_rng(99)
    idx = rng.permutation(n)
    n_train = int(0.8 * n)
    train_idx = idx[:n_train]
    test_idx = idx[n_train:]
    train = IVSurfaceData(
        strikes=synth_data.strikes[train_idx],
        expiries=synth_data.expiries[train_idx],
        ivs=synth_data.ivs[train_idx],
        spot=synth_data.spot,
    )
    test = IVSurfaceData(
        strikes=synth_data.strikes[test_idx],
        expiries=synth_data.expiries[test_idx],
        ivs=synth_data.ivs[test_idx],
        spot=synth_data.spot,
    )
    return train, test


# -----------------------------------------------------------------------
# IVSurfaceData
# -----------------------------------------------------------------------

class TestIVSurfaceData:
    def test_creation(self) -> None:
        data = IVSurfaceData(
            strikes=np.array([90, 100, 110]),
            expiries=np.array([0.5, 0.5, 0.5]),
            ivs=np.array([0.25, 0.20, 0.22]),
            spot=100.0,
        )
        assert len(data.strikes) == 3
        assert data.spot == 100.0

    def test_moneyness(self) -> None:
        data = IVSurfaceData(
            strikes=np.array([100.0]),
            expiries=np.array([1.0]),
            ivs=np.array([0.2]),
            spot=100.0,
        )
        np.testing.assert_allclose(data.moneyness, [0.0], atol=1e-12)

    def test_features_shape(self) -> None:
        data = IVSurfaceData(
            strikes=np.array([90, 100, 110]),
            expiries=np.array([0.5, 1.0, 1.5]),
            ivs=np.array([0.25, 0.20, 0.22]),
            spot=100.0,
        )
        assert data.features.shape == (3, 2)

    def test_mismatched_lengths_raises(self) -> None:
        with pytest.raises(ValueError, match="equal length"):
            IVSurfaceData(
                strikes=np.array([90, 100]),
                expiries=np.array([0.5]),
                ivs=np.array([0.2, 0.25]),
            )


# -----------------------------------------------------------------------
# evaluate_surface
# -----------------------------------------------------------------------

class TestEvaluateSurface:
    def test_perfect_prediction(self) -> None:
        y = np.array([0.2, 0.25, 0.3])
        metrics = evaluate_surface(y, y)
        assert metrics.rmse == pytest.approx(0.0, abs=1e-12)
        assert metrics.mae == pytest.approx(0.0, abs=1e-12)
        assert metrics.n_samples == 3

    def test_known_error(self) -> None:
        y_true = np.array([0.2, 0.3])
        y_pred = np.array([0.21, 0.29])
        metrics = evaluate_surface(y_true, y_pred)
        assert metrics.rmse == pytest.approx(0.01, abs=1e-10)
        assert metrics.mae == pytest.approx(0.01, abs=1e-10)
        assert metrics.max_error == pytest.approx(0.01, abs=1e-10)


# -----------------------------------------------------------------------
# Synthetic data generation
# -----------------------------------------------------------------------

class TestSyntheticData:
    def test_shape(self) -> None:
        data = generate_synthetic_iv_surface(n_strikes=5, n_expiries=3)
        assert len(data.strikes) == 15
        assert len(data.expiries) == 15
        assert len(data.ivs) == 15

    def test_positive_ivs(self) -> None:
        data = generate_synthetic_iv_surface(n_strikes=10, n_expiries=5, seed=0)
        assert np.all(data.ivs > 0)

    def test_reproducibility(self) -> None:
        d1 = generate_synthetic_iv_surface(seed=123)
        d2 = generate_synthetic_iv_surface(seed=123)
        np.testing.assert_array_equal(d1.ivs, d2.ivs)

    def test_skew_present(self) -> None:
        """OTM puts (low strikes) should have higher IV than OTM calls."""
        data = generate_synthetic_iv_surface(
            n_strikes=10, n_expiries=1, skew=-0.15, smile=0.0,
            noise=0.0, seed=0,
        )
        # Pick one expiry slice
        low_k = data.ivs[data.strikes < data.spot]
        high_k = data.ivs[data.strikes > data.spot]
        assert np.mean(low_k) > np.mean(high_k)


# -----------------------------------------------------------------------
# SABR model
# -----------------------------------------------------------------------

class TestSABRModel:
    def test_calibrate_and_predict(self, synth_data: IVSurfaceData) -> None:
        sabr = SABRModel(beta=0.5)
        sabr.calibrate(synth_data)
        preds = sabr.predict(
            synth_data.strikes, synth_data.expiries, synth_data.spot
        )
        assert preds.shape == synth_data.ivs.shape
        assert np.all(np.isfinite(preds))

    def test_params_accessible(self, synth_data: IVSurfaceData) -> None:
        sabr = SABRModel(beta=0.5)
        sabr.calibrate(synth_data)
        params = sabr.params
        assert "alpha" in params
        assert "rho" in params
        assert "nu" in params
        assert "beta" in params
        assert -1 < params["rho"] < 1

    def test_uncalibrated_raises(self) -> None:
        sabr = SABRModel()
        with pytest.raises(RuntimeError, match="calibrate"):
            sabr.predict(np.array([100.0]), np.array([1.0]), 100.0)

    def test_atm_accuracy(self) -> None:
        """SABR should be accurate near ATM."""
        data = generate_synthetic_iv_surface(
            n_strikes=10, n_expiries=3, noise=0.0, seed=7,
        )
        sabr = SABRModel(beta=0.5)
        sabr.calibrate(data)
        preds = sabr.predict(data.strikes, data.expiries, data.spot)
        atm_mask = np.abs(data.strikes - data.spot) < 5.0
        atm_errors = np.abs(preds[atm_mask] - data.ivs[atm_mask])
        assert np.mean(atm_errors) < 0.05  # reasonable ATM fit


# -----------------------------------------------------------------------
# SVI model
# -----------------------------------------------------------------------

class TestSVIModel:
    def test_calibrate_and_predict(self, synth_data: IVSurfaceData) -> None:
        svi = SVIModel()
        svi.calibrate(synth_data)
        preds = svi.predict(
            synth_data.strikes, synth_data.expiries, synth_data.spot
        )
        assert preds.shape == synth_data.ivs.shape
        assert np.all(np.isfinite(preds))
        assert np.all(preds > 0)

    def test_params_shape(self, synth_data: IVSurfaceData) -> None:
        svi = SVIModel()
        svi.calibrate(synth_data)
        params = svi.params
        assert params.shape == (5,)

    def test_uncalibrated_raises(self) -> None:
        svi = SVIModel()
        with pytest.raises(RuntimeError, match="calibrate"):
            svi.predict(np.array([100.0]), np.array([1.0]), 100.0)

    def test_fit_quality(self) -> None:
        """SVI should fit synthetic data well (no noise)."""
        data = generate_synthetic_iv_surface(
            n_strikes=8, n_expiries=4, noise=0.0, seed=11,
        )
        svi = SVIModel()
        svi.calibrate(data)
        preds = svi.predict(data.strikes, data.expiries, data.spot)
        metrics = evaluate_surface(data.ivs, preds)
        assert metrics.rmse < 0.05


# -----------------------------------------------------------------------
# QSVM regression
# -----------------------------------------------------------------------

class TestQSVMRegression:
    def test_fit_and_predict(self, backend: MockBackend) -> None:
        data = generate_synthetic_iv_surface(
            n_strikes=4, n_expiries=2, noise=0.0, seed=42,
        )
        cfg = QuantumIVSurfaceConfig(n_qubits=2, method="qsvm", reps=1)
        model = QuantumIVSurface(cfg, backend)
        model.fit(data)
        preds = model.predict(data.strikes, data.expiries, data.spot)
        assert preds.shape == data.ivs.shape
        assert np.all(np.isfinite(preds))

    def test_evaluate_returns_metrics(self, backend: MockBackend) -> None:
        data = generate_synthetic_iv_surface(
            n_strikes=4, n_expiries=2, noise=0.0, seed=42,
        )
        cfg = QuantumIVSurfaceConfig(n_qubits=2, method="qsvm", reps=1)
        model = QuantumIVSurface(cfg, backend)
        model.fit(data)
        metrics = model.evaluate(data)
        assert isinstance(metrics, SurfaceMetrics)
        assert metrics.rmse >= 0.0
        assert metrics.n_samples == len(data.ivs)

    def test_requires_backend(self) -> None:
        cfg = QuantumIVSurfaceConfig(method="qsvm")
        with pytest.raises(ValueError, match="Backend required"):
            QuantumIVSurface(cfg, backend=None)


# -----------------------------------------------------------------------
# VQC regression
# -----------------------------------------------------------------------

class TestVQCRegression:
    def test_fit_and_predict(self, backend: MockBackend) -> None:
        data = generate_synthetic_iv_surface(
            n_strikes=3, n_expiries=2, noise=0.0, seed=42,
        )
        cfg = QuantumIVSurfaceConfig(
            n_qubits=2, method="vqc", n_layers=2, maxiter=50, seed=42,
        )
        model = QuantumIVSurface(cfg, backend)
        model.fit(data)
        preds = model.predict(data.strikes, data.expiries, data.spot)
        assert preds.shape == data.ivs.shape
        assert np.all(np.isfinite(preds))

    def test_requires_backend(self) -> None:
        cfg = QuantumIVSurfaceConfig(method="vqc")
        with pytest.raises(ValueError, match="Backend required"):
            QuantumIVSurface(cfg, backend=None)


# -----------------------------------------------------------------------
# QSVM vs VQC comparison
# -----------------------------------------------------------------------

class TestQuantumComparison:
    def test_both_methods_produce_output(self, backend: MockBackend) -> None:
        """Both QSVM and VQC should produce finite predictions."""
        data = generate_synthetic_iv_surface(
            n_strikes=3, n_expiries=2, noise=0.0, seed=42,
        )
        cfg_qsvm = QuantumIVSurfaceConfig(n_qubits=2, method="qsvm", reps=1)
        cfg_vqc = QuantumIVSurfaceConfig(
            n_qubits=2, method="vqc", n_layers=2, maxiter=30, seed=42,
        )
        model_qsvm = QuantumIVSurface(cfg_qsvm, backend)
        model_vqc = QuantumIVSurface(cfg_vqc, backend)

        model_qsvm.fit(data)
        model_vqc.fit(data)

        preds_qsvm = model_qsvm.predict(data.strikes, data.expiries, data.spot)
        preds_vqc = model_vqc.predict(data.strikes, data.expiries, data.spot)

        assert np.all(np.isfinite(preds_qsvm))
        assert np.all(np.isfinite(preds_vqc))
        # They should not be identical (different methods)
        assert not np.allclose(preds_qsvm, preds_vqc, atol=1e-10)


# -----------------------------------------------------------------------
# Classical baseline comparison
# -----------------------------------------------------------------------

class TestClassicalComparison:
    def test_sabr_vs_svi(self) -> None:
        """Both SABR and SVI should fit reasonably on synthetic data."""
        data = generate_synthetic_iv_surface(
            n_strikes=8, n_expiries=4, noise=0.0, seed=55,
        )
        sabr = SABRModel(beta=0.5)
        sabr.calibrate(data)
        preds_sabr = sabr.predict(data.strikes, data.expiries, data.spot)

        svi = SVIModel()
        svi.calibrate(data)
        preds_svi = svi.predict(data.strikes, data.expiries, data.spot)

        metrics_sabr = evaluate_surface(data.ivs, preds_sabr)
        metrics_svi = evaluate_surface(data.ivs, preds_svi)

        # Both should have finite, positive predictions
        assert np.all(np.isfinite(preds_sabr))
        assert np.all(np.isfinite(preds_svi))
        # Both should achieve some reasonable fit
        assert metrics_sabr.rmse < 0.1
        assert metrics_svi.rmse < 0.1


# -----------------------------------------------------------------------
# Out-of-sample prediction
# -----------------------------------------------------------------------

class TestOutOfSample:
    def test_svi_oos(self, train_test_split) -> None:
        """SVI out-of-sample should have finite, positive predictions."""
        train, test = train_test_split
        svi = SVIModel()
        svi.calibrate(train)
        preds = svi.predict(test.strikes, test.expiries, test.spot)
        assert np.all(np.isfinite(preds))
        assert np.all(preds > 0)
        metrics = evaluate_surface(test.ivs, preds)
        assert metrics.rmse < 0.15

    def test_sabr_oos(self, train_test_split) -> None:
        """SABR out-of-sample should have finite predictions."""
        train, test = train_test_split
        sabr = SABRModel(beta=0.5)
        sabr.calibrate(train)
        preds = sabr.predict(test.strikes, test.expiries, test.spot)
        assert np.all(np.isfinite(preds))

    def test_qsvm_oos(self, backend: MockBackend, train_test_split) -> None:
        """QSVM out-of-sample predictions should be finite."""
        train, test = train_test_split
        cfg = QuantumIVSurfaceConfig(n_qubits=2, method="qsvm", reps=1)
        model = QuantumIVSurface(cfg, backend)
        model.fit(train)
        preds = model.predict(test.strikes, test.expiries, test.spot)
        assert np.all(np.isfinite(preds))
        assert preds.shape == test.ivs.shape


# -----------------------------------------------------------------------
# Edge cases: ATM and deep OTM
# -----------------------------------------------------------------------

class TestEdgeCases:
    def test_atm_prediction(self, backend: MockBackend) -> None:
        """Model should handle ATM options (K == S)."""
        data = generate_synthetic_iv_surface(
            n_strikes=6, n_expiries=3, noise=0.0, seed=42,
        )
        cfg = QuantumIVSurfaceConfig(n_qubits=2, method="qsvm", reps=1)
        model = QuantumIVSurface(cfg, backend)
        model.fit(data)
        # Predict at ATM
        atm_strikes = np.array([100.0, 100.0])
        atm_expiries = np.array([0.5, 1.0])
        preds = model.predict(atm_strikes, atm_expiries, data.spot)
        assert np.all(np.isfinite(preds))
        assert preds.shape == (2,)

    def test_deep_otm_prediction(self, backend: MockBackend) -> None:
        """Model should handle deep OTM options."""
        data = generate_synthetic_iv_surface(
            n_strikes=6, n_expiries=3, noise=0.0, seed=42,
        )
        cfg = QuantumIVSurfaceConfig(n_qubits=2, method="qsvm", reps=1)
        model = QuantumIVSurface(cfg, backend)
        model.fit(data)
        # Deep OTM strikes
        otm_strikes = np.array([70.0, 140.0])
        otm_expiries = np.array([0.5, 0.5])
        preds = model.predict(otm_strikes, otm_expiries, data.spot)
        assert np.all(np.isfinite(preds))

    def test_single_expiry_slice(self) -> None:
        """SVI should work with a single expiry."""
        data = generate_synthetic_iv_surface(
            n_strikes=8, n_expiries=1, noise=0.0, seed=42,
        )
        svi = SVIModel()
        svi.calibrate(data)
        preds = svi.predict(data.strikes, data.expiries, data.spot)
        assert np.all(np.isfinite(preds))
        assert np.all(preds > 0)

    def test_sabr_atm_formula(self) -> None:
        """SABR ATM formula should return positive vol."""
        from qufin.options.implied_vol_surface import _sabr_iv

        iv = _sabr_iv(f=100.0, k=100.0, T=1.0, alpha=0.3, beta=0.5, rho=-0.3, nu=0.4)
        assert iv > 0
        assert np.isfinite(iv)

    def test_all_method_fits_everything(self, backend: MockBackend) -> None:
        """Method 'all' should fit all sub-models."""
        data = generate_synthetic_iv_surface(
            n_strikes=3, n_expiries=2, noise=0.0, seed=42,
        )
        cfg = QuantumIVSurfaceConfig(
            n_qubits=2, method="all", reps=1, n_layers=2, maxiter=20, seed=42,
        )
        model = QuantumIVSurface(cfg, backend)
        model.fit(data)
        # All methods should produce predictions
        for method in ("qsvm", "vqc", "sabr", "svi"):
            preds = model.predict(
                data.strikes, data.expiries, data.spot, method=method,
            )
            assert np.all(np.isfinite(preds)), f"{method} produced non-finite"

    def test_unknown_method_raises(self, backend: MockBackend) -> None:
        data = generate_synthetic_iv_surface(n_strikes=3, n_expiries=2, seed=42)
        cfg = QuantumIVSurfaceConfig(n_qubits=2, method="qsvm", reps=1)
        model = QuantumIVSurface(cfg, backend)
        model.fit(data)
        with pytest.raises(ValueError, match="Unknown method"):
            model.predict(data.strikes, data.expiries, data.spot, method="bogus")
