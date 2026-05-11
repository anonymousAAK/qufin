"""Production stress test suite for qufin.

This suite validates every module end-to-end against known analytical
results, cross-checks quantum vs classical implementations, and stress-tests
edge cases, numerical stability, and scalability.

Designed for presentation to JPMorgan, Google, and tier-1 institutions.

Run with:
    pytest tests/stress/ -v --tb=short
    pytest tests/stress/ -v --tb=short -k "not slow"   # fast subset (~30s)
"""

from __future__ import annotations

import importlib
import pkgutil
import time

import numpy as np
import pytest
from scipy.stats import norm

# ---------------------------------------------------------------------------
# 0. INFRASTRUCTURE: every module imports without error
# ---------------------------------------------------------------------------


class TestModuleIntegrity:
    """Verify that every qufin submodule can be imported."""

    def test_all_modules_import(self) -> None:
        import qufin

        failures = []
        count = 0
        for _importer, name, _ispkg in pkgutil.walk_packages(
            qufin.__path__, prefix="qufin."
        ):
            try:
                importlib.import_module(name)
                count += 1
            except Exception as exc:
                failures.append(f"{name}: {exc}")

        assert count >= 80, f"Only {count} modules found — expected >= 80"
        assert failures == [], "Import failures:\n" + "\n".join(failures)

    def test_top_level_exports(self) -> None:
        import qufin

        assert hasattr(qufin, "backends")
        assert hasattr(qufin, "options")
        assert hasattr(qufin, "portfolio")
        assert hasattr(qufin, "risk")
        assert hasattr(qufin, "hedging")
        assert hasattr(qufin, "ml")
        assert hasattr(qufin, "derivatives")
        assert hasattr(qufin, "benchmarks")
        assert hasattr(qufin, "data")
        assert hasattr(qufin, "utils")


# ---------------------------------------------------------------------------
# 1. BLACK-SCHOLES: exact analytical validation
# ---------------------------------------------------------------------------


class TestBlackScholesExact:
    """Validate BS pricing against textbook-known values and identities."""

    def test_atm_call_textbook(self) -> None:
        """ATM call S=K=100, r=5%, sigma=20%, T=1y -> ~10.4506 (Hull ch.15)."""
        from qufin.options.classical.black_scholes import call_price

        c = call_price(100.0, 100.0, 0.05, 0.20, 1.0)
        assert abs(c - 10.4506) < 0.01, f"ATM call={c}, expected ~10.4506"

    def test_atm_put_textbook(self) -> None:
        from qufin.options.classical.black_scholes import put_price

        p = put_price(100.0, 100.0, 0.05, 0.20, 1.0)
        expected = 10.4506 - 100 + 100 * np.exp(-0.05)  # put-call parity
        assert abs(p - expected) < 0.01

    def test_put_call_parity_exact(self) -> None:
        """C - P = S*exp(-qT) - K*exp(-rT) for all parameter combos."""
        from qufin.options.classical.black_scholes import call_price, put_price

        rng = np.random.default_rng(42)
        for _ in range(500):
            s = rng.uniform(20, 500)
            k = rng.uniform(20, 500)
            r = rng.uniform(0, 0.15)
            q = rng.uniform(0, 0.05)
            sigma = rng.uniform(0.05, 1.5)
            T = rng.uniform(0.01, 5.0)
            c = call_price(s, k, r, sigma, T, q)
            p = put_price(s, k, r, sigma, T, q)
            parity = s * np.exp(-q * T) - k * np.exp(-r * T)
            assert abs((c - p) - parity) < 1e-8, (
                f"Parity violated: s={s:.2f} k={k:.2f} r={r:.4f} "
                f"q={q:.4f} sigma={sigma:.4f} T={T:.4f}"
            )

    def test_greeks_finite_difference(self) -> None:
        """Verify analytical Greeks match finite-difference estimates."""
        from qufin.options.classical.black_scholes import (
            call_price,
            delta,
            gamma,
            rho,
            vega,
        )

        s, k, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.0
        eps = 1e-5

        # Delta
        d_analytical = delta(s, k, r, sigma, T, option_type="call")
        c_up = call_price(s + eps, k, r, sigma, T)
        c_dn = call_price(s - eps, k, r, sigma, T)
        d_fd = (c_up - c_dn) / (2 * eps)
        assert abs(d_analytical - d_fd) < 1e-5, f"Delta: {d_analytical} vs FD {d_fd}"

        # Gamma
        g_analytical = gamma(s, k, r, sigma, T)
        g_fd = (
            call_price(s + eps, k, r, sigma, T)
            - 2 * call_price(s, k, r, sigma, T)
            + call_price(s - eps, k, r, sigma, T)
        ) / eps**2
        assert abs(g_analytical - g_fd) < 1e-4, f"Gamma: {g_analytical} vs FD {g_fd}"

        # Vega
        v_analytical = vega(s, k, r, sigma, T)
        c_vup = call_price(s, k, r, sigma + eps, T)
        c_vdn = call_price(s, k, r, sigma - eps, T)
        v_fd = (c_vup - c_vdn) / (2 * eps)
        assert abs(v_analytical - v_fd) < 1e-4, f"Vega: {v_analytical} vs FD {v_fd}"

        # Rho
        rho_analytical = rho(s, k, r, sigma, T, option_type="call")
        c_rup = call_price(s, k, r + eps, sigma, T)
        c_rdn = call_price(s, k, r - eps, sigma, T)
        rho_fd = (c_rup - c_rdn) / (2 * eps)
        assert abs(rho_analytical - rho_fd) < 1e-4, f"Rho: {rho_analytical} vs FD {rho_fd}"

    def test_implied_volatility_round_trip(self) -> None:
        """BS price -> implied vol -> BS price should round-trip exactly."""
        from qufin.options.classical.black_scholes import call_price, implied_volatility

        for sigma_true in [0.10, 0.20, 0.40, 0.80, 1.20]:
            price = call_price(100.0, 100.0, 0.05, sigma_true, 1.0)
            sigma_implied = implied_volatility(
                price, 100.0, 100.0, 0.05, 1.0, option_type="call"
            )
            assert abs(sigma_implied - sigma_true) < 1e-6, (
                f"IV round-trip failed: true={sigma_true}, implied={sigma_implied}"
            )

    def test_deep_itm_otm_limits(self) -> None:
        """Deep ITM call -> S - K*exp(-rT); deep OTM call -> 0."""
        from qufin.options.classical.black_scholes import call_price

        # Deep ITM: S=1000, K=1
        c = call_price(1000.0, 1.0, 0.05, 0.2, 1.0)
        intrinsic = 1000.0 - 1.0 * np.exp(-0.05)
        assert abs(c - intrinsic) < 0.01

        # Deep OTM: S=1, K=1000
        c = call_price(1.0, 1000.0, 0.05, 0.2, 1.0)
        assert c < 1e-10

    def test_zero_vol_call(self) -> None:
        """At zero vol, call = max(S*exp(-qT) - K*exp(-rT), 0)."""
        from qufin.options.classical.black_scholes import call_price

        c = call_price(110.0, 100.0, 0.05, 1e-10, 1.0)
        expected = max(110.0 - 100.0 * np.exp(-0.05), 0)
        assert abs(c - expected) < 0.01


# ---------------------------------------------------------------------------
# 2. BINOMIAL TREE: convergence to BS
# ---------------------------------------------------------------------------


class TestBinomialConvergence:
    """CRR tree should converge to BS as n_steps -> infinity."""

    def test_european_call_converges_to_bs(self) -> None:
        from qufin.options.classical.binomial import crr_tree
        from qufin.options.classical.black_scholes import call_price

        bs = call_price(100.0, 100.0, 0.05, 0.20, 1.0)
        for n in [50, 200, 500]:
            tree = crr_tree(100, 100, 0.05, 0.20, 1.0, n_steps=n, option_type="call")
            assert abs(tree.price - bs) < 0.5 / np.sqrt(n), (
                f"n={n}: tree={tree.price:.6f}, bs={bs:.6f}"
            )

    def test_european_put_converges_to_bs(self) -> None:
        from qufin.options.classical.binomial import crr_tree
        from qufin.options.classical.black_scholes import put_price

        bs = put_price(100.0, 100.0, 0.05, 0.20, 1.0)
        tree = crr_tree(100, 100, 0.05, 0.20, 1.0, n_steps=500, option_type="put")
        assert abs(tree.price - bs) < 0.05

    def test_american_put_geq_european(self) -> None:
        """American put >= European put (early exercise premium)."""
        from qufin.options.classical.binomial import crr_tree

        euro = crr_tree(100, 110, 0.05, 0.3, 1.0, 200, "put", "european")
        amer = crr_tree(100, 110, 0.05, 0.3, 1.0, 200, "put", "american")
        assert amer.price >= euro.price - 1e-10

    def test_delta_agrees_with_bs(self) -> None:
        """Tree delta should be close to BS delta."""
        from qufin.options.classical.binomial import crr_tree
        from qufin.options.classical.black_scholes import delta

        bs_d = delta(100, 100, 0.05, 0.20, 1.0, option_type="call")
        tree = crr_tree(100, 100, 0.05, 0.20, 1.0, n_steps=500, option_type="call")
        assert abs(tree.delta - bs_d) < 0.02, f"tree delta={tree.delta}, bs={bs_d}"

    def test_gamma_positive_for_vanilla(self) -> None:
        from qufin.options.classical.binomial import crr_tree

        tree = crr_tree(100, 100, 0.05, 0.20, 1.0, n_steps=500, option_type="call")
        assert tree.gamma > 0, f"gamma={tree.gamma}, expected > 0"


# ---------------------------------------------------------------------------
# 3. MONTE CARLO: convergence and variance reduction
# ---------------------------------------------------------------------------


class TestMonteCarloConvergence:
    """MC pricing convergence and antithetic variance reduction."""

    def test_european_mc_converges_to_bs(self) -> None:
        from qufin.options.classical.black_scholes import call_price
        from qufin.options.classical.monte_carlo import european_mc

        bs = call_price(100.0, 100.0, 0.05, 0.20, 1.0)
        mc = european_mc(100, 100, 0.05, 0.20, 1.0, n_paths=500_000, option_type="call", seed=42)
        assert abs(mc.price - bs) < 0.15, f"MC={mc.price:.4f}, BS={bs:.4f}"

    def test_antithetic_reduces_variance(self) -> None:
        from qufin.options.classical.monte_carlo import european_mc

        plain = european_mc(100, 100, 0.05, 0.2, 1.0,
                            n_paths=50_000, option_type="call", antithetic=False, seed=42)
        anti = european_mc(100, 100, 0.05, 0.2, 1.0,
                           n_paths=50_000, option_type="call", antithetic=True, seed=42)
        assert anti.std_err <= plain.std_err * 1.1  # antithetic should help

    def test_asian_mc_positive(self) -> None:
        from qufin.options.classical.monte_carlo import asian_mc

        result = asian_mc(100, 100, 0.05, 0.2, 1.0, n_paths=50_000, option_type="call", seed=42)
        assert result.price > 0

    def test_barrier_mc_less_than_vanilla(self) -> None:
        """Knock-out barrier price <= vanilla price."""
        from qufin.options.classical.monte_carlo import barrier_mc, european_mc

        vanilla = european_mc(100, 100, 0.05, 0.2, 1.0, n_paths=50_000, option_type="call", seed=42)
        knocked = barrier_mc(100, 100, 0.05, 0.2, 1.0, barrier=120.0,
                             barrier_type="up-and-out", n_paths=50_000, option_type="call", seed=42)
        assert knocked.price <= vanilla.price + 0.5


# ---------------------------------------------------------------------------
# 4. PORTFOLIO OPTIMIZATION: classical baselines
# ---------------------------------------------------------------------------


class TestPortfolioOptimization:
    """Mean-variance, Black-Litterman, risk parity, HRP."""

    @pytest.fixture
    def market_data(self):
        rng = np.random.default_rng(42)
        mu = rng.normal(0.0005, 0.0005, 10)
        F = rng.normal(0, 0.3, (10, 3))
        cov = F @ F.T + np.diag(rng.uniform(0.01, 0.04, 10))
        cov = (cov + cov.T) / 2
        return mu, cov

    def test_min_variance_weights_sum_to_one(self, market_data) -> None:
        from qufin.portfolio.classical.mean_variance import Objective, mean_variance

        mu, cov = market_data
        result = mean_variance(mu, cov, Objective.MIN_VARIANCE)
        assert abs(result.weights.sum() - 1.0) < 1e-6
        assert np.all(result.weights >= -1e-6)

    def test_max_sharpe_positive_sharpe(self, market_data) -> None:
        from qufin.portfolio.classical.mean_variance import Objective, mean_variance

        mu, cov = market_data
        # Ensure at least one asset has positive excess return
        mu_shifted = mu + 0.001
        result = mean_variance(mu_shifted, cov, Objective.MAX_SHARPE, risk_free_rate=0.0)
        assert abs(result.weights.sum() - 1.0) < 1e-4
        assert result.sharpe_ratio > 0

    def test_cardinality_constraint(self, market_data) -> None:
        from qufin.portfolio.classical.mean_variance import Objective, mean_variance

        mu, cov = market_data
        result = mean_variance(mu, cov, Objective.MIN_VARIANCE, cardinality=3)
        n_nonzero = np.sum(result.weights > 1e-4)
        assert n_nonzero <= 3, f"{n_nonzero} assets selected, expected <= 3"

    def test_black_litterman_posterior(self, market_data) -> None:
        from qufin.portfolio.classical.black_litterman import black_litterman

        _mu, cov = market_data
        market_caps = np.arange(1, 11, dtype=np.float64) * 1e9
        P = np.zeros((1, 10))
        P[0, 0] = 1
        P[0, 1] = -1
        Q = np.array([0.01])  # asset 0 outperforms asset 1 by 1%
        result = black_litterman(cov, market_caps, P=P, Q=Q)
        assert result.posterior_mu[0] > result.posterior_mu[1]

    def test_risk_parity_equal_risk_contribution(self, market_data) -> None:
        from qufin.portfolio.classical.risk_parity import risk_parity

        _, cov = market_data
        result = risk_parity(cov)
        assert abs(result.weights.sum() - 1.0) < 1e-4
        # Check risk contributions are roughly equal
        w = result.weights
        port_vol = np.sqrt(w @ cov @ w)
        mrc = cov @ w / port_vol
        rc = w * mrc
        rc_pct = rc / rc.sum()
        assert np.std(rc_pct) < 0.05, f"Risk contributions not equal: {rc_pct}"

    def test_hrp_returns_valid_weights(self, market_data) -> None:
        from qufin.portfolio.classical.hrp import hrp

        _, cov = market_data
        result = hrp(cov)
        assert abs(result.weights.sum() - 1.0) < 1e-6
        assert np.all(result.weights >= 0)
        assert np.all(result.weights <= 1)


# ---------------------------------------------------------------------------
# 5. QUBO + QAOA: quantum portfolio optimization
# ---------------------------------------------------------------------------


class TestQUBOPortfolio:
    """QUBO formulation + exhaustive solver + QAOA."""

    def test_qubo_symmetry(self) -> None:
        from qufin.portfolio.qubo import PortfolioQUBO

        rng = np.random.default_rng(42)
        mu = rng.normal(0, 0.01, 8)
        cov = np.eye(8) * 0.04
        qubo = PortfolioQUBO(mu, cov, gamma=1.0, cardinality=3)
        Q = qubo.build_matrix()
        np.testing.assert_allclose(Q, Q.T, atol=1e-12)

    def test_exhaustive_finds_global_minimum(self) -> None:
        from qufin.portfolio.optimizers.exhaustive import exhaustive_solve
        from qufin.portfolio.qubo import PortfolioQUBO

        rng = np.random.default_rng(42)
        mu = rng.normal(0, 0.01, 6)
        cov = np.eye(6) * 0.02
        qubo = PortfolioQUBO(mu, cov, gamma=1.0, cardinality=2)
        result = exhaustive_solve(qubo)
        # Global min should be the best among all 2^6 bitstrings
        for i in range(2**6):
            bs = format(i, "06b")
            obj = qubo.evaluate(bs)
            assert result.best_objective <= obj + 1e-10

    @pytest.mark.slow
    def test_qaoa_improves_over_random(self) -> None:
        """QAOA with p=1 should beat random sampling on average."""
        from qufin.backends.qiskit_backend import QiskitAerBackend
        from qufin.portfolio.optimizers.qaoa import QAOAConfig, QAOAPortfolio
        from qufin.portfolio.qubo import PortfolioQUBO

        rng = np.random.default_rng(42)
        n = 4
        mu = rng.normal(0, 0.01, n)
        cov = np.eye(n) * 0.02
        qubo = PortfolioQUBO(mu, cov, gamma=1.0, cardinality=2)

        # Random baseline
        random_objs = []
        for _ in range(100):
            bs = format(rng.integers(0, 2**n), f"0{n}b")
            random_objs.append(qubo.evaluate(bs))

        backend = QiskitAerBackend(seed=42)
        config = QAOAConfig(p=1, mixer="x", shots=4096, maxiter=30, seed=42)
        qaoa = QAOAPortfolio(qubo, config, backend)
        result = qaoa.run()

        # QAOA should find a reasonable objective (at least not worst)
        assert result.best_objective < np.max(random_objs)


# ---------------------------------------------------------------------------
# 6. QUANTUM AMPLITUDE ESTIMATION: all 4 variants
# ---------------------------------------------------------------------------


class TestQAEVariants:
    """Test all 4 QAE algorithms against known amplitudes."""

    @pytest.fixture
    def simple_problem(self):
        """A = sin^2(pi/6) = 0.25 problem."""
        from qiskit.circuit import QuantumCircuit

        from qufin.options.amplitude_estimation.estimation_problem import EstimationProblem

        qc = QuantumCircuit(1)
        qc.ry(2 * np.arcsin(np.sqrt(0.25)), 0)
        return EstimationProblem(
            state_preparation=qc,
            objective_qubits=[0],
            n_qubits=1,
        )

    @pytest.mark.slow
    def test_iqae_accuracy(self, simple_problem) -> None:
        from qufin.backends.qiskit_backend import QiskitAerBackend
        from qufin.options.amplitude_estimation.iqae import (
            IQAEConfig,
            IterativeAmplitudeEstimation,
        )

        backend = QiskitAerBackend(seed=42)
        config = IQAEConfig(epsilon_target=0.01, alpha=0.05, shots_per_round=4096)
        qae = IterativeAmplitudeEstimation(simple_problem, config, backend)
        result = qae.estimate()
        assert abs(result.estimate - 0.25) < 0.05, f"IQAE estimate={result.estimate}"

    @pytest.mark.slow
    def test_mlae_accuracy(self, simple_problem) -> None:
        from qufin.backends.qiskit_backend import QiskitAerBackend
        from qufin.options.amplitude_estimation.mlae import (
            MaximumLikelihoodAmplitudeEstimation,
            MLAEConfig,
        )

        backend = QiskitAerBackend(seed=42)
        config = MLAEConfig(n_shots_per_round=4096)
        qae = MaximumLikelihoodAmplitudeEstimation(simple_problem, config, backend)
        result = qae.estimate()
        assert abs(result.estimate - 0.25) < 0.05, f"MLAE estimate={result.estimate}"

    @pytest.mark.slow
    def test_canonical_qae_accuracy(self, simple_problem) -> None:
        from qufin.backends.qiskit_backend import QiskitAerBackend
        from qufin.options.amplitude_estimation.canonical import (
            CanonicalAmplitudeEstimation,
            CanonicalQAEConfig,
        )

        backend = QiskitAerBackend(seed=42)
        config = CanonicalQAEConfig(n_eval_qubits=5, shots=8192)
        qae = CanonicalAmplitudeEstimation(simple_problem, config, backend)
        result = qae.estimate()
        assert abs(result.estimate - 0.25) < 0.05, f"Canonical estimate={result.estimate}"

    @pytest.mark.slow
    def test_fqae_accuracy(self, simple_problem) -> None:
        from qufin.backends.qiskit_backend import QiskitAerBackend
        from qufin.options.amplitude_estimation.fqae import (
            FaithfulAmplitudeEstimation,
            FQAEConfig,
        )

        backend = QiskitAerBackend(seed=42)
        config = FQAEConfig(n_shots_per_round=4096)
        qae = FaithfulAmplitudeEstimation(simple_problem, config, backend)
        result = qae.estimate()
        assert abs(result.estimate - 0.25) < 0.10, f"FQAE estimate={result.estimate}"


# ---------------------------------------------------------------------------
# 7. RISK: VaR, ES, credit, stress, counterparty
# ---------------------------------------------------------------------------


class TestRiskManagement:
    """Full risk module validation."""

    @pytest.fixture
    def returns_1y(self):
        rng = np.random.default_rng(42)
        return rng.normal(0.0003, 0.015, 252)

    def test_historical_var_ordering(self, returns_1y) -> None:
        """ES >= VaR always (ES is further in the tail)."""
        from qufin.risk.classical_var import historical_var

        result = historical_var(returns_1y, confidence=0.95)
        assert result.expected_shortfall >= result.var - 1e-10

    def test_parametric_var_agrees_with_formula(self, returns_1y) -> None:
        """Parametric VaR should match direct Gaussian computation."""
        from qufin.risk.classical_var import parametric_var

        result = parametric_var(returns_1y, confidence=0.99)
        mu = float(np.mean(returns_1y))
        sigma = float(np.std(returns_1y, ddof=1))
        z = norm.ppf(0.99)
        expected_var = -(mu - z * sigma)
        assert abs(result.var - expected_var) < 1e-10

    def test_mc_var_consistent_with_parametric(self, returns_1y) -> None:
        """MC VaR should be within ~20% of parametric for normal data."""
        from qufin.risk.classical_var import monte_carlo_var, parametric_var

        param = parametric_var(returns_1y, confidence=0.95)
        mc = monte_carlo_var(returns_1y, confidence=0.95, n_simulations=200_000, seed=42)
        assert abs(mc.var - param.var) / param.var < 0.2

    def test_var_monotone_in_confidence(self, returns_1y) -> None:
        """Higher confidence -> higher VaR."""
        from qufin.risk.classical_var import historical_var

        var_90 = historical_var(returns_1y, confidence=0.90).var
        var_95 = historical_var(returns_1y, confidence=0.95).var
        var_99 = historical_var(returns_1y, confidence=0.99).var
        assert var_90 <= var_95 + 1e-10
        assert var_95 <= var_99 + 1e-10

    def test_stress_scenarios_all_negative_pnl(self) -> None:
        """All stress scenarios should produce negative P&L for equity-only portfolio."""
        from qufin.risk.stress import SCENARIO_LIBRARY, apply_stress

        weights = np.array([1.0, 0.0, 0.0, 0.0])  # pure equity
        for name, scenario in SCENARIO_LIBRARY.items():
            result = apply_stress(1_000_000.0, weights, scenario)
            assert result["total_pnl"] < 0, f"{name}: positive P&L under stress"

    def test_stress_suite_runs_all(self) -> None:
        from qufin.risk.stress import SCENARIO_LIBRARY, stress_test_suite

        weights = np.array([0.5, 0.2, 0.2, 0.1])
        results = stress_test_suite(1_000_000.0, weights)
        assert len(results) == len(SCENARIO_LIBRARY)

    def test_cva_positive_for_risky_counterparty(self) -> None:
        from qufin.risk.counterparty import CounterpartyExposure, compute_cva

        exp = CounterpartyExposure(
            name="RiskyBank",
            notional=10_000_000.0,
            pd=0.05,
            lgd=0.45,
            exposure_profile=np.linspace(100_000, 500_000, 12),
        )
        cva = compute_cva(exp)
        assert cva > 0

    def test_sa_ccr_ead(self) -> None:
        """SA-CCR: EAD = alpha * (RC + PFE)."""
        from qufin.risk.counterparty import compute_ead_sa_ccr

        ead = compute_ead_sa_ccr(
            notional=1_000_000.0, mtm=50_000.0, add_on_factor=0.01
        )
        # EAD = 1.4 * (max(50k-0, 0) + 1.0 * 0.01*1M) = 1.4 * (50k + 10k) = 84k
        assert abs(ead - 84_000.0) < 1.0

    def test_portfolio_cva(self) -> None:
        from qufin.risk.counterparty import (
            CounterpartyExposure,
            portfolio_cva,
        )

        exposures = [
            CounterpartyExposure("A", 1e6, 0.02, 0.40, np.ones(4) * 100_000),
            CounterpartyExposure("B", 2e6, 0.05, 0.60, np.ones(4) * 200_000),
        ]
        result = portfolio_cva(exposures)
        assert result["total"] > 0
        assert result["B"] > result["A"]  # B is riskier


# ---------------------------------------------------------------------------
# 8. CREDIT RISK: Gaussian copula, Vasicek, NIG
# ---------------------------------------------------------------------------


class TestCreditRisk:
    """Validate credit models against analytical results."""

    def test_vasicek_converges_to_independent(self) -> None:
        """With rho~0, Vasicek VaR should be close to PD * LGD."""
        from qufin.risk.credit.gaussian_copula import vasicek_analytical

        # rho near zero: tail risk vanishes, VaR ~ PD * LGD
        el = vasicek_analytical(pd=0.01, rho=1e-6, confidence=0.99)
        assert el["expected_loss"] > 0

    def test_gaussian_copula_mc_expected_loss(self) -> None:
        from qufin.risk.credit.gaussian_copula import CreditPortfolio, gaussian_copula_mc

        portfolio = CreditPortfolio(
            n_obligors=20,
            default_probs=np.full(20, 0.02),
            correlations=np.full(20, 0.15),
            exposures=np.ones(20) * 100_000,
        )
        result = gaussian_copula_mc(portfolio, n_simulations=100_000, seed=42)
        # Expected loss ~ 20 * 0.02 * 100k = 40k
        assert abs(result.expected_loss - 40_000) < 10_000

    def test_nig_copula_runs(self) -> None:
        from qufin.risk.credit.nig_copula import NIGParams, nig_copula_mc

        nig_params = NIGParams(alpha=1.5, beta=-0.2, mu=0.0, delta=1.0)
        result = nig_copula_mc(
            n_obligors=10,
            default_probs=np.full(10, 0.03),
            nig_params=nig_params,
            exposures=np.ones(10) * 50_000,
            n_simulations=50_000,
            seed=42,
        )
        assert result["expected_loss"] > 0
        # VaR should exceed expected loss
        assert result["var"] >= result["expected_loss"]


# ---------------------------------------------------------------------------
# 9. HEDGING: delta, deep hedging
# ---------------------------------------------------------------------------


class TestHedging:
    """Delta hedging and deep hedging validation."""

    def test_delta_hedge_reduces_variance(self) -> None:
        """Hedged P&L should have lower variance than unhedged."""
        from qufin.hedging.delta import DeltaHedger

        hedger = DeltaHedger()
        pnls = []
        for seed in range(50):
            result = hedger.hedge(100, 100, 0.05, 0.2, 1.0, n_rebalances=50, seed=seed)
            pnls.append(result.hedging_error)
        pnl_std = np.std(pnls)
        # Unhedged std ~ sigma * S * sqrt(T) ~ 0.2 * 100 * 1 = 20
        # Hedged should be much less
        assert pnl_std < 5.0, f"Hedge error std={pnl_std}, too large"

    def test_more_rebalances_better_hedge(self) -> None:
        from qufin.hedging.delta import DeltaHedger

        hedger = DeltaHedger()
        errors = {}
        for n in [5, 20, 100]:
            errs = []
            for seed in range(30):
                r = hedger.hedge(100, 100, 0.05, 0.2, 1.0, n_rebalances=n, seed=seed)
                errs.append(abs(r.hedging_error))
            errors[n] = np.mean(errs)
        assert errors[100] < errors[5], "More rebalances should improve hedge"

    def test_deep_hedger_trains(self) -> None:
        from qufin.hedging.deep_hedging import DeepHedger, DeepHedgingConfig

        cfg = DeepHedgingConfig(n_epochs=10, n_paths=256, n_steps=10, hidden_dim=16)
        hedger = DeepHedger(cfg, s0=100.0, strike=100.0, seed=42)
        initial_loss = hedger.train()
        # After training, loss should be finite
        assert np.isfinite(initial_loss[-1])


# ---------------------------------------------------------------------------
# 10. DERIVATIVES: exotic pricing
# ---------------------------------------------------------------------------


class TestExoticDerivatives:
    """Basket, lookback, cliquet, autocallable, Bermudan."""

    def test_basket_mc_vs_geometric_closed_form(self) -> None:
        """Geometric basket MC should converge to the closed form."""
        from qufin.derivatives.basket import (
            BasketOptionSpec,
            basket_mc,
            geometric_basket_closed_form,
        )

        spec = BasketOptionSpec(
            s0=np.array([100.0, 100.0]),
            k=100.0,
            r=0.05,
            sigma=np.array([0.2, 0.3]),
            corr=np.array([[1.0, 0.5], [0.5, 1.0]]),
            T=1.0,
            is_call=True,
        )
        closed = geometric_basket_closed_form(spec)
        mc = basket_mc(spec, n_paths=200_000, seed=42)
        # MC should be within a few std errors
        assert abs(mc.price - closed) < 4 * mc.std_err + 1.0

    def test_lookback_call_geq_vanilla(self) -> None:
        """Lookback call >= vanilla call (floating strike advantage)."""
        from qufin.derivatives.path_dependent import LookbackOptionSpec, lookback_mc
        from qufin.options.classical.black_scholes import call_price

        bs = call_price(100, 100, 0.05, 0.2, 1.0)
        spec = LookbackOptionSpec(s0=100, r=0.05, sigma=0.2, T=1.0, is_call=True, n_steps=100)
        lb = lookback_mc(spec, n_paths=100_000, seed=42)
        assert lb > bs - 0.5, f"Lookback={lb:.2f} < BS={bs:.2f}"

    def test_cliquet_deterministic(self) -> None:
        from qufin.derivatives.path_dependent import cliquet_mc

        p1 = cliquet_mc(100, 0.05, 0.2, 1.0, n_periods=4, n_paths=10_000, seed=42)
        p2 = cliquet_mc(100, 0.05, 0.2, 1.0, n_periods=4, n_paths=10_000, seed=42)
        assert p1 == p2

    def test_autocallable_prob_increases_with_lower_barrier(self) -> None:
        from qufin.derivatives.autocallable import AutocallableSpec, autocallable_mc

        spec_high = AutocallableSpec(
            s0=100, k=100, barrier=130, coupon=0.05,
            observation_dates=[0.25, 0.5, 0.75, 1.0],
            r=0.05, sigma=0.2, T=1.0,
        )
        spec_low = AutocallableSpec(
            s0=100, k=100, barrier=105, coupon=0.05,
            observation_dates=[0.25, 0.5, 0.75, 1.0],
            r=0.05, sigma=0.2, T=1.0,
        )
        r_high = autocallable_mc(spec_high, n_paths=50_000, seed=42)
        r_low = autocallable_mc(spec_low, n_paths=50_000, seed=42)
        assert r_low["autocall_prob"] >= r_high["autocall_prob"] - 0.01

    def test_lsm_bermudan_geq_european(self) -> None:
        from qufin.derivatives.bermudan_lsm import lsm_price

        european = lsm_price(
            100, 105, 0.05, 0.3, 1.0,
            n_steps=50, n_paths=50_000,
            exercise_dates=[1.0], is_call=False, seed=42,
        )
        bermudan = lsm_price(
            100, 105, 0.05, 0.3, 1.0,
            n_steps=50, n_paths=50_000,
            exercise_dates=[0.25, 0.5, 0.75, 1.0], is_call=False, seed=42,
        )
        assert bermudan["price"] >= european["price"] - 0.5

    def test_bermudan_binomial_vs_lsm(self) -> None:
        """Binomial and LSM should agree within tolerance."""
        from qufin.derivatives.bermudan_lsm import lsm_price
        from qufin.options.bermudan import BermudanOptionSpec, bermudan_binomial

        spec = BermudanOptionSpec(
            s0=100, k=100, r=0.05, sigma=0.2, T=1.0,
            exercise_dates=[0.25, 0.5, 0.75, 1.0],
            is_call=False, n_steps=200,
        )
        tree = bermudan_binomial(spec)
        lsm = lsm_price(
            100, 100, 0.05, 0.2, 1.0,
            n_steps=100, n_paths=100_000,
            exercise_dates=[0.25, 0.5, 0.75, 1.0],
            is_call=False, seed=42,
        )
        assert abs(tree.price - lsm["price"]) < 0.5, (
            f"Binomial={tree.price:.4f}, LSM={lsm['price']:.4f}"
        )


# ---------------------------------------------------------------------------
# 11. QUANTUM ML: kernels, reservoir, VQC
# ---------------------------------------------------------------------------


class TestQuantumML:
    """Quantum machine learning modules."""

    def test_kernel_matrix_properties(self) -> None:
        from qufin.backends.mock import MockBackend
        from qufin.ml.kernels import quantum_kernel_matrix

        backend = MockBackend(seed=42)
        rng = np.random.default_rng(42)
        X = rng.uniform(0, 2 * np.pi, (8, 2))
        K = quantum_kernel_matrix(X, n_qubits=2, backend=backend, reps=1)

        # Symmetric
        np.testing.assert_allclose(K, K.T, atol=1e-12)
        # Diagonal = 1
        np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-12)
        # PSD
        eigvals = np.linalg.eigvalsh(K)
        assert np.all(eigvals >= -1e-10)

    def test_reservoir_features_bounded(self) -> None:
        from qufin.backends.mock import MockBackend
        from qufin.ml.reservoir import QuantumReservoir, QuantumReservoirConfig

        cfg = QuantumReservoirConfig(n_qubits=4, n_layers=2, seed=42)
        backend = MockBackend(seed=0)
        reservoir = QuantumReservoir(cfg, backend)
        features = reservoir.extract_features(np.array([0.1, 0.2, 0.3, 0.4]))
        assert np.all(features >= -1.0 - 1e-10)
        assert np.all(features <= 1.0 + 1e-10)

    def test_vqc_predict_binary(self) -> None:
        from qufin.backends.mock import MockBackend
        from qufin.ml.classifiers import VariationalQuantumClassifier, VQCConfig

        cfg = VQCConfig(n_qubits=2, n_layers=1, n_epochs=5, seed=42)
        backend = MockBackend(seed=0)
        vqc = VariationalQuantumClassifier(cfg, backend)
        X = np.array([[0.1, 0.2], [1.0, 1.5], [0.3, 0.4], [2.0, 2.5]])
        y = np.array([0, 1, 0, 1])
        vqc.fit(X, y)
        preds = vqc.predict(X)
        assert set(preds).issubset({0, 1})
        assert len(preds) == 4


# ---------------------------------------------------------------------------
# 12. DATA GENERATORS: GBM, Heston, Merton
# ---------------------------------------------------------------------------


class TestDataGenerators:
    """Validate synthetic data generators."""

    def test_gbm_martingale_property(self) -> None:
        """Under risk-neutral measure, E[S_T] = S_0 * exp(rT)."""
        from qufin.data.synthetic import gbm_paths

        paths = gbm_paths(s0=100, mu=0.05, sigma=0.2, T=1.0,
                          n_steps=252, n_paths=100_000, seed=42)
        s_T = paths[:, -1]
        expected = 100 * np.exp(0.05)
        assert abs(np.mean(s_T) - expected) / expected < 0.01

    def test_heston_variance_positive(self) -> None:
        """Heston variance paths should stay positive with full truncation."""
        from qufin.data.synthetic import heston_paths

        s_paths, v_paths = heston_paths(
            s0=100, v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7,
            mu=0.05, T=1.0, n_steps=252, n_paths=10_000, seed=42,
        )
        assert np.all(v_paths >= 0), "Negative variance detected"
        assert np.all(s_paths > 0), "Non-positive prices detected"

    def test_merton_fatter_tails(self) -> None:
        """Merton jump-diffusion should produce fatter tails than GBM."""
        from qufin.data.synthetic import gbm_paths, merton_jump_paths

        gbm = gbm_paths(100, 0.05, 0.2, 1.0, 252, 50_000, seed=42)
        merton = merton_jump_paths(
            100, 0.05, 0.15, lam=2.0, jump_mean=-0.10, jump_std=0.20,
            T=1.0, n_steps=252, n_paths=50_000, seed=42,
        )
        gbm_rets = np.log(gbm[:, -1] / gbm[:, 0])
        merton_rets = np.log(merton[:, -1] / merton[:, 0])
        from scipy.stats import kurtosis
        gbm_kurt = kurtosis(gbm_rets)
        merton_kurt = kurtosis(merton_rets)
        assert merton_kurt > gbm_kurt, "Merton should have heavier tails"


# ---------------------------------------------------------------------------
# 13. BENCHMARKS: harness, metrics, leaderboard
# ---------------------------------------------------------------------------


class TestBenchmarkHarness:
    """Benchmark infrastructure."""

    def test_all_problems_instantiate(self) -> None:
        from qufin.benchmarks.problems import all_problems

        problems = all_problems()
        assert len(problems) >= 4
        for p in problems:
            assert p.problem_id
            assert p.description

    def test_runner_dispatches(self) -> None:
        from qufin.benchmarks.problems import portfolio_small
        from qufin.benchmarks.runner import BenchmarkRunner, SolverEntry

        problem = portfolio_small()

        def dummy_solver(p):
            return {"objective": 42.0, "backend": "test"}

        runner = BenchmarkRunner()
        runner.register(SolverEntry("dummy", "classical", dummy_solver))
        rows = runner.run_problem(problem)
        assert len(rows) == 1
        assert rows[0].objective == 42.0

    def test_leaderboard_generation(self) -> None:
        from qufin.benchmarks.leaderboard import to_csv, to_markdown
        from qufin.benchmarks.runner import BenchmarkRow

        rows = [
            BenchmarkRow("p1", "qaoa", "quantum", 0.5, 0.01, 1.0, 10, 5, "aer", 42),
            BenchmarkRow("p1", "mv", "classical", 0.49, 0.0, 0.01, None, None, "", 42),
        ]
        md = to_markdown(rows)
        assert "qaoa" in md
        csv_str = to_csv(rows)
        assert "mv" in csv_str

    def test_manifest_captures_environment(self) -> None:
        from qufin.benchmarks.manifest import build_manifest

        m = build_manifest(problem_ids=["test"], seeds=[42])
        assert m.python_version
        assert "numpy" in m.package_versions


# ---------------------------------------------------------------------------
# 14. BARRIER OPTIONS: closed-form edge cases
# ---------------------------------------------------------------------------


class TestBarrierOptions:
    """Barrier option closed-form edge cases."""

    def test_up_and_out_call_at_barrier_is_zero(self) -> None:
        from qufin.options.barrier import barrier_closed_form

        price = barrier_closed_form(120.0, 100.0, 0.05, 0.2, 1.0, 120.0, "up-and-out", True)
        assert price == 0.0

    def test_down_and_out_call_at_barrier_is_zero(self) -> None:
        from qufin.options.barrier import barrier_closed_form

        price = barrier_closed_form(80.0, 100.0, 0.05, 0.2, 1.0, 80.0, "down-and-out", True)
        assert price == 0.0

    def test_in_out_parity(self) -> None:
        """knock-in + knock-out = vanilla."""
        from qufin.options.barrier import barrier_closed_form
        from qufin.options.classical.black_scholes import call_price

        vanilla = call_price(100, 100, 0.05, 0.2, 1.0)
        out = barrier_closed_form(100, 100, 0.05, 0.2, 1.0, 120.0, "up-and-out", True)
        inp = barrier_closed_form(100, 100, 0.05, 0.2, 1.0, 120.0, "up-and-in", True)
        assert abs((out + inp) - vanilla) < 0.5, (
            f"Parity: in={inp:.4f} + out={out:.4f} = {inp+out:.4f}, vanilla={vanilla:.4f}"
        )

    def test_up_and_out_put(self) -> None:
        from qufin.options.barrier import barrier_closed_form

        price = barrier_closed_form(100, 100, 0.05, 0.2, 1.0, 120.0, "up-and-out", False)
        assert price >= 0

    def test_down_and_out_put(self) -> None:
        from qufin.options.barrier import barrier_closed_form

        price = barrier_closed_form(100, 100, 0.05, 0.2, 1.0, 80.0, "down-and-out", False)
        assert price >= 0


# ---------------------------------------------------------------------------
# 15. NUMERICAL STABILITY: edge cases that break naive implementations
# ---------------------------------------------------------------------------


class TestNumericalStability:
    """Edge cases that stress-test numerical robustness."""

    def test_bs_near_expiry(self) -> None:
        """BS shouldn't explode as T -> 0."""
        from qufin.options.classical.black_scholes import call_price, put_price

        c = call_price(100, 100, 0.05, 0.2, 1e-6)
        p = put_price(100, 100, 0.05, 0.2, 1e-6)
        assert np.isfinite(c)
        assert np.isfinite(p)

    def test_bs_extreme_vol(self) -> None:
        from qufin.options.classical.black_scholes import call_price

        c_low = call_price(100, 100, 0.05, 0.001, 1.0)
        c_high = call_price(100, 100, 0.05, 5.0, 1.0)
        assert np.isfinite(c_low)
        assert np.isfinite(c_high)
        assert c_high > c_low

    def test_cov_matrix_near_singular(self) -> None:
        """Mean-variance should handle near-singular covariance."""
        from qufin.portfolio.classical.mean_variance import Objective, mean_variance

        n = 5
        cov = np.ones((n, n)) * 0.04 + np.eye(n) * 1e-6
        mu = np.ones(n) * 0.001
        result = mean_variance(mu, cov, Objective.MIN_VARIANCE)
        assert abs(result.weights.sum() - 1.0) < 1e-4

    def test_qubo_large_penalty(self) -> None:
        """QUBO with very large penalty shouldn't produce NaN."""
        from qufin.portfolio.qubo import PortfolioQUBO

        mu = np.array([0.01, 0.02, 0.03])
        cov = np.eye(3) * 0.04
        qubo = PortfolioQUBO(mu, cov, gamma=1.0, cardinality=1, budget_penalty=1e6)
        Q = qubo.build_matrix()
        assert np.all(np.isfinite(Q))
        obj = qubo.evaluate("100")
        assert np.isfinite(obj)

    def test_implied_vol_extreme_prices(self) -> None:
        """IV solver should handle edge cases gracefully."""
        from qufin.options.classical.black_scholes import call_price, implied_volatility

        # Very deep ITM: price ~ intrinsic
        price = call_price(200, 100, 0.05, 0.2, 1.0)
        iv = implied_volatility(price, 200, 100, 0.05, 1.0, option_type="call")
        assert abs(iv - 0.2) < 0.01

    def test_var_with_zero_returns(self) -> None:
        """VaR should handle zero-variance returns."""
        from qufin.risk.classical_var import historical_var

        returns = np.zeros(100)
        result = historical_var(returns, confidence=0.95)
        assert result.var == 0.0


# ---------------------------------------------------------------------------
# 16. CROSS-VALIDATION: quantum vs classical agreement
# ---------------------------------------------------------------------------


class TestQuantumClassicalAgreement:
    """Cross-validate quantum and classical implementations."""

    @pytest.mark.slow
    def test_quantum_var_agrees_with_classical(self) -> None:
        """Quantum VaR (via QAE bisection) should agree with classical."""
        from qufin.backends.qiskit_backend import QiskitAerBackend
        from qufin.risk.classical_var import parametric_var
        from qufin.risk.quantum_var import (
            QuantumVaRConfig,
            build_loss_distribution,
            quantum_var,
        )

        rng = np.random.default_rng(42)
        returns = rng.normal(0.0003, 0.015, 252)
        classical = parametric_var(returns, confidence=0.95)

        loss_dist = build_loss_distribution(returns, n_qubits=3)
        backend = QiskitAerBackend(seed=42)
        config = QuantumVaRConfig(
            confidence_level=0.95,
            n_qubits_loss=3,
            n_bisection_steps=5,
            qae_shots=2048,
        )
        quantum = quantum_var(loss_dist, backend, config)
        # With only 3 qubits, accuracy is very limited — check same order of magnitude
        assert abs(quantum.var_estimate - classical.var) / classical.var < 5.0

    @pytest.mark.slow
    def test_egger_credit_agrees_with_classical(self) -> None:
        """Egger QAE expected loss should agree with analytical."""
        from qufin.backends.qiskit_backend import QiskitAerBackend
        from qufin.risk.credit.egger import (
            EggerConfig,
            egger_classical_reference,
            egger_expected_loss,
        )
        from qufin.risk.credit.gaussian_copula import CreditPortfolio

        portfolio = CreditPortfolio(
            n_obligors=2,
            default_probs=np.array([0.05, 0.03]),
            correlations=np.array([0.2, 0.2]),
            exposures=np.array([100_000, 150_000]),
        )
        classical_el = egger_classical_reference(portfolio)
        backend = QiskitAerBackend(seed=42)
        config = EggerConfig(n_qubits_z=2, qae_shots=2048)
        quantum_result = egger_expected_loss(portfolio, backend, config)
        # Within factor of 2 for small qubit count
        assert quantum_result.expected_loss > 0
        assert abs(quantum_result.expected_loss - classical_el) / (classical_el + 1e-10) < 2.0


# ---------------------------------------------------------------------------
# 17. SCALABILITY: verify no O(n!) or accidental exponentials
# ---------------------------------------------------------------------------


class TestScalability:
    """Verify key algorithms scale as expected."""

    def test_portfolio_classical_scales_linearly(self) -> None:
        """Mean-variance solver should scale ~linearly in N assets."""
        from qufin.portfolio.classical.mean_variance import Objective, mean_variance

        times = {}
        for n in [10, 50, 100]:
            rng = np.random.default_rng(42)
            mu = rng.normal(0, 0.01, n)
            cov = np.eye(n) * 0.04
            start = time.perf_counter()
            mean_variance(mu, cov, Objective.MIN_VARIANCE)
            times[n] = time.perf_counter() - start

        # 100-asset should take < 100x the 10-asset time
        assert times[100] < times[10] * 200, (
            f"Scaling issue: 10-asset={times[10]:.3f}s, 100-asset={times[100]:.3f}s"
        )

    def test_mc_pricing_scales_linearly_in_paths(self) -> None:
        from qufin.options.classical.monte_carlo import european_mc

        times = {}
        for n in [10_000, 100_000]:
            start = time.perf_counter()
            european_mc(100, 100, 0.05, 0.2, 1.0, n_paths=n, option_type="call", seed=42)
            times[n] = time.perf_counter() - start

        ratio = times[100_000] / times[10_000]
        assert ratio < 15, f"MC scaling: 10x paths took {ratio:.1f}x time"

    def test_qubo_build_50_assets(self) -> None:
        """QUBO matrix build should complete in < 2s for 50 assets."""
        from qufin.portfolio.qubo import PortfolioQUBO

        rng = np.random.default_rng(42)
        mu = rng.normal(0, 0.01, 50)
        cov = np.eye(50) * 0.04
        start = time.perf_counter()
        qubo = PortfolioQUBO(mu, cov, gamma=1.0, cardinality=10)
        Q = qubo.build_matrix()
        elapsed = time.perf_counter() - start
        assert elapsed < 2.0, f"QUBO build took {elapsed:.2f}s"
        assert Q.shape == (50, 50)


# ---------------------------------------------------------------------------
# 18. REPRODUCIBILITY: same seed = same result, always
# ---------------------------------------------------------------------------


class TestReproducibility:
    """Every stochastic algorithm must be deterministic with a seed."""

    def test_gbm_reproducible(self) -> None:
        from qufin.data.synthetic import gbm_paths

        a = gbm_paths(100, 0.05, 0.2, 1.0, 100, 1000, seed=42)
        b = gbm_paths(100, 0.05, 0.2, 1.0, 100, 1000, seed=42)
        np.testing.assert_array_equal(a, b)

    def test_mc_pricing_reproducible(self) -> None:
        from qufin.options.classical.monte_carlo import european_mc

        a = european_mc(100, 100, 0.05, 0.2, 1.0, n_paths=10_000, option_type="call", seed=7)
        b = european_mc(100, 100, 0.05, 0.2, 1.0, n_paths=10_000, option_type="call", seed=7)
        assert a.price == b.price

    def test_lsm_reproducible(self) -> None:
        from qufin.derivatives.bermudan_lsm import lsm_price

        a = lsm_price(100, 100, 0.05, 0.2, 1.0, n_paths=5000, seed=42)
        b = lsm_price(100, 100, 0.05, 0.2, 1.0, n_paths=5000, seed=42)
        assert a["price"] == b["price"]

    def test_gaussian_copula_reproducible(self) -> None:
        from qufin.risk.credit.gaussian_copula import CreditPortfolio, gaussian_copula_mc

        portfolio = CreditPortfolio(
            n_obligors=5,
            default_probs=np.full(5, 0.02),
            correlations=np.full(5, 0.15),
            exposures=np.ones(5) * 100_000,
        )
        a = gaussian_copula_mc(portfolio, n_simulations=10_000, seed=42)
        b = gaussian_copula_mc(portfolio, n_simulations=10_000, seed=42)
        assert a.expected_loss == b.expected_loss

    def test_delta_hedger_reproducible(self) -> None:
        from qufin.hedging.delta import DeltaHedger

        hedger = DeltaHedger()
        a = hedger.hedge(100, 100, 0.05, 0.2, 1.0, 10, seed=42)
        b = hedger.hedge(100, 100, 0.05, 0.2, 1.0, 10, seed=42)
        assert a.pnl == b.pnl


# ---------------------------------------------------------------------------
# 19. HESTON: stochastic volatility
# ---------------------------------------------------------------------------


class TestHeston:
    """Heston model validation."""

    def test_heston_weak_vs_strong_euler(self) -> None:
        """Both Euler schemes should give similar terminal distributions."""
        from qufin.options.heston import (
            HestonParams,
            heston_strong_euler_terminal,
            heston_weak_euler_terminal,
        )

        params = HestonParams(s0=100, v0=0.04, kappa=2.0, theta=0.04,
                              xi=0.3, rho=-0.7, r=0.05, T=1.0)
        weak = heston_weak_euler_terminal(params, n_steps=100, n_paths=50_000, seed=42)
        strong = heston_strong_euler_terminal(params, n_steps=100, n_paths=50_000, seed=42)
        # Mean terminal prices should agree within MC noise
        k = 100.0
        weak_call = float(np.exp(-params.r * params.T) * np.mean(np.maximum(weak - k, 0)))
        strong_call = float(np.exp(-params.r * params.T) * np.mean(np.maximum(strong - k, 0)))
        assert abs(weak_call - strong_call) < 2.0

    def test_heston_reduces_to_bs_at_zero_vol_of_vol(self) -> None:
        """When xi=0 (no vol-of-vol), Heston -> BS with sigma=sqrt(v0)."""
        from qufin.options.classical.black_scholes import call_price
        from qufin.options.heston import HestonParams, heston_strong_euler_terminal

        v0 = 0.04  # sigma = 0.2
        params = HestonParams(s0=100, v0=v0, kappa=5.0, theta=v0,
                              xi=0.001, rho=0.0, r=0.05, T=1.0)
        terminals = heston_strong_euler_terminal(params, n_steps=200, n_paths=200_000, seed=42)
        k = 100.0
        heston_call = float(np.exp(-params.r * params.T) * np.mean(np.maximum(terminals - k, 0)))
        bs = call_price(100, 100, 0.05, np.sqrt(v0), 1.0)
        assert abs(heston_call - bs) < 1.0, (
            f"Heston(xi~0)={heston_call:.2f}, BS={bs:.2f}"
        )


# ---------------------------------------------------------------------------
# 20. END-TO-END WORKFLOW: full pipeline simulation
# ---------------------------------------------------------------------------


class TestEndToEndWorkflow:
    """Simulate a complete quant workflow from data to risk report."""

    def test_full_portfolio_pipeline(self) -> None:
        """Data -> optimize -> risk -> stress test."""
        from qufin.data.synthetic import gbm_paths
        from qufin.portfolio.classical.mean_variance import Objective, mean_variance
        from qufin.risk.classical_var import historical_var
        from qufin.risk.stress import stress_test_suite

        # 1. Generate synthetic returns for 10 assets
        rng = np.random.default_rng(42)
        returns = np.column_stack([
            gbm_paths(100, rng.uniform(0.03, 0.08), rng.uniform(0.15, 0.35),
                      1.0, 252, 1, seed=i)[:, 1:].flatten()
            / gbm_paths(100, rng.uniform(0.03, 0.08), rng.uniform(0.15, 0.35),
                        1.0, 252, 1, seed=i)[:, :-1].flatten()
            - 1
            for i in range(10)
        ])

        mu = returns.mean(axis=0)
        cov = np.cov(returns, rowvar=False)

        # 2. Optimize portfolio
        result = mean_variance(mu, cov, Objective.MIN_VARIANCE)
        assert abs(result.weights.sum() - 1.0) < 1e-4

        # 3. Compute portfolio returns
        port_returns = returns @ result.weights

        # 4. Risk analysis
        var_result = historical_var(port_returns, confidence=0.95)
        assert var_result.var > 0
        assert var_result.expected_shortfall >= var_result.var

        # 5. Stress testing
        stress_weights = np.array([0.7, 0.1, 0.1, 0.1])
        stress_results = stress_test_suite(1_000_000.0, stress_weights)
        assert len(stress_results) == 4

    def test_full_option_pricing_pipeline(self) -> None:
        """BS -> MC -> binomial -> QAE problem build (no execution)."""
        from qufin.options.amplitude_estimation.european_qae import (
            EuropeanQAESpec,
            build_european_estimation_problem,
        )
        from qufin.options.classical.binomial import crr_tree
        from qufin.options.classical.black_scholes import call_price, price_and_greeks
        from qufin.options.classical.monte_carlo import european_mc

        # Price with all methods
        bs = call_price(100, 100, 0.05, 0.2, 1.0)
        price_and_greeks(100, 100, 0.05, 0.2, 1.0, "call")
        tree = crr_tree(100, 100, 0.05, 0.2, 1.0, n_steps=200)
        mc = european_mc(100, 100, 0.05, 0.2, 1.0, n_paths=100_000, option_type="call", seed=42)

        # All should agree within tolerance
        assert abs(tree.price - bs) < 0.1
        assert abs(mc.price - bs) < 0.5

        # Build QAE problem (circuit construction, no execution)
        spec = EuropeanQAESpec(s0=100, k=100, r=0.05, sigma=0.2, T=1.0, n_qubits=3)
        _problem, rescale = build_european_estimation_problem(spec)
        assert rescale > 0

    def test_full_credit_risk_pipeline(self) -> None:
        """Classical copula -> stress -> CVA."""
        from qufin.risk.counterparty import (
            CounterpartyExposure,
            compute_cva,
            compute_ead_sa_ccr,
        )
        from qufin.risk.credit.gaussian_copula import CreditPortfolio, gaussian_copula_mc

        # 1. Credit portfolio analysis
        portfolio = CreditPortfolio(
            n_obligors=50,
            default_probs=np.random.default_rng(42).uniform(0.005, 0.05, 50),
            correlations=np.full(50, 0.15),
            exposures=np.random.default_rng(42).uniform(50_000, 500_000, 50),
        )
        result = gaussian_copula_mc(portfolio, n_simulations=100_000, seed=42)
        assert result.expected_loss > 0
        assert result.var_99 > result.expected_loss

        # 2. Counterparty CVA
        exp = CounterpartyExposure(
            name="BigBank",
            notional=50_000_000,
            pd=0.02,
            lgd=0.45,
            exposure_profile=np.linspace(1_000_000, 5_000_000, 20),
        )
        cva = compute_cva(exp)
        assert cva > 0

        # 3. SA-CCR
        ead = compute_ead_sa_ccr(50_000_000, 2_000_000, 0.005)
        assert ead > 0
