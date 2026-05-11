"""Unit tests for credit risk models."""

from __future__ import annotations

import numpy as np
import pytest

from qufin.risk.credit.gaussian_copula import (
    CreditPortfolio,
    gaussian_copula_mc,
    vasicek_analytical,
)
from qufin.risk.credit.nig_copula import (
    NIGParams,
    nig_copula_mc,
    nig_pdf,
)
from qufin.risk.credit.egger import (
    EggerConfig,
    egger_classical_reference,
    egger_expected_loss,
    build_expected_loss_problem,
)


@pytest.fixture
def small_portfolio() -> CreditPortfolio:
    return CreditPortfolio(
        n_obligors=3,
        default_probs=np.array([0.05, 0.03, 0.07]),
        correlations=np.array([0.2, 0.2, 0.2]),
        exposures=np.array([100.0, 200.0, 150.0]),
        recovery_rates=np.array([0.4, 0.3, 0.5]),
    )


@pytest.fixture
def homogeneous_portfolio() -> CreditPortfolio:
    """Homogeneous portfolio for Vasicek comparison."""
    n = 5
    return CreditPortfolio(
        n_obligors=n,
        default_probs=np.full(n, 0.05),
        correlations=np.full(n, 0.2),
        exposures=np.full(n, 100.0),
    )


class TestCreditPortfolio:
    def test_lgd(self, small_portfolio: CreditPortfolio) -> None:
        lgd = small_portfolio.lgd
        assert lgd[0] == pytest.approx(60.0)   # 100 * (1-0.4)
        assert lgd[1] == pytest.approx(140.0)  # 200 * (1-0.3)
        assert lgd[2] == pytest.approx(75.0)   # 150 * (1-0.5)

    def test_default_thresholds(self, small_portfolio: CreditPortfolio) -> None:
        thresholds = small_portfolio.default_thresholds
        assert len(thresholds) == 3
        # Threshold for PD=0.05 should be about -1.645
        assert thresholds[0] == pytest.approx(-1.6449, abs=0.01)

    def test_scalar_correlation(self) -> None:
        port = CreditPortfolio(
            n_obligors=3,
            default_probs=np.array([0.05, 0.05, 0.05]),
            correlations=np.float64(0.3),
            exposures=np.array([100.0, 100.0, 100.0]),
        )
        assert len(port.correlations) == 3
        assert all(port.correlations == 0.3)


class TestGaussianCopulaMC:
    def test_expected_loss_positive(self, small_portfolio: CreditPortfolio) -> None:
        result = gaussian_copula_mc(small_portfolio, n_simulations=50_000, seed=42)
        assert result.expected_loss > 0

    def test_var_exceeds_el(self, small_portfolio: CreditPortfolio) -> None:
        result = gaussian_copula_mc(small_portfolio, n_simulations=50_000, seed=42)
        assert result.var_99 >= result.expected_loss

    def test_es_exceeds_var(self, small_portfolio: CreditPortfolio) -> None:
        result = gaussian_copula_mc(small_portfolio, n_simulations=50_000, seed=42)
        assert result.es_99 >= result.var_99

    def test_deterministic(self, small_portfolio: CreditPortfolio) -> None:
        r1 = gaussian_copula_mc(small_portfolio, n_simulations=10_000, seed=42)
        r2 = gaussian_copula_mc(small_portfolio, n_simulations=10_000, seed=42)
        assert r1.expected_loss == r2.expected_loss

    def test_el_close_to_analytical(self, small_portfolio: CreditPortfolio) -> None:
        """MC expected loss should approximate sum(PD * LGD)."""
        result = gaussian_copula_mc(small_portfolio, n_simulations=200_000, seed=42)
        analytical_el = float(np.sum(small_portfolio.default_probs * small_portfolio.lgd))
        assert result.expected_loss == pytest.approx(analytical_el, rel=0.15)


class TestVasicekAnalytical:
    def test_basic(self) -> None:
        result = vasicek_analytical(pd=0.05, rho=0.2, lgd=1.0, confidence=0.99)
        assert result["expected_loss"] == pytest.approx(0.05)
        assert result["var"] > result["expected_loss"]
        assert result["conditional_pd"] > 0

    def test_higher_correlation_higher_var(self) -> None:
        r_low = vasicek_analytical(pd=0.05, rho=0.1)
        r_high = vasicek_analytical(pd=0.05, rho=0.5)
        assert r_high["var"] > r_low["var"]


class TestNIGCopula:
    def test_nig_params(self) -> None:
        params = NIGParams(alpha=1.5, beta=0.0, mu=0.0, delta=1.0)
        assert params.gamma == pytest.approx(1.5)
        assert params.skewness == pytest.approx(0.0)

    def test_nig_pdf(self) -> None:
        params = NIGParams(alpha=1.5, beta=0.0, mu=0.0, delta=1.0)
        x = np.linspace(-3, 3, 100)
        pdf = nig_pdf(x, params)
        assert all(pdf >= 0)
        dx = x[1] - x[0]
        assert float(np.sum(pdf) * dx) == pytest.approx(1.0, abs=0.05)

    def test_invalid_params(self) -> None:
        with pytest.raises(ValueError):
            NIGParams(alpha=-1.0)
        with pytest.raises(ValueError):
            NIGParams(alpha=1.0, beta=1.5)

    def test_nig_copula_mc_basic(self) -> None:
        result = nig_copula_mc(
            n_obligors=3,
            default_probs=np.array([0.05, 0.03, 0.07]),
            nig_params=NIGParams(),
            exposures=np.array([100.0, 200.0, 150.0]),
            n_simulations=20_000,
            seed=42,
        )
        assert result["expected_loss"] > 0
        assert result["var"] >= result["expected_loss"]


class TestEgger:
    def test_classical_reference(self, small_portfolio: CreditPortfolio) -> None:
        el = egger_classical_reference(small_portfolio)
        expected = float(np.sum(small_portfolio.lgd * small_portfolio.default_probs))
        assert el == pytest.approx(expected)

    def test_build_problem(self, small_portfolio: CreditPortfolio) -> None:
        config = EggerConfig(n_qubits_z=2, seed=42)
        problem, rescale = build_expected_loss_problem(small_portfolio, config)
        # n_z + n_obligors + 1 payoff ancilla
        assert problem.n_qubits == 2 + 3 + 1
        assert rescale > 0

    def test_egger_qae(self, small_portfolio: CreditPortfolio) -> None:
        """Test Egger QAE expected loss (lightweight config)."""
        from qufin.backends.qiskit_backend import QiskitAerBackend

        backend = QiskitAerBackend()
        config = EggerConfig(
            n_qubits_z=2,
            qae_method="iqae",
            qae_epsilon=0.1,
            qae_shots=256,
            seed=42,
        )
        result = egger_expected_loss(small_portfolio, backend, config)
        assert result.expected_loss >= 0
        assert result.n_qubits_total == 6
        assert len(result.conditional_pds) == 3
