"""Phase 4.1 — Classical baseline comparisons.

Benchmarks qufin's classical implementations against raw numpy/scipy/cvxpy
for pricing, portfolio optimization, and risk. Outputs JSON + Markdown tables.

Usage:
    python benchmarks/classical_comparison.py [--output results/classical.json]
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class BenchmarkEntry:
    category: str
    method: str
    n: int
    wall_time_s: float
    result: float
    reference: float | None = None
    rel_error: float | None = None
    extra: dict = field(default_factory=dict)


def _timer(fn, *args, repeats: int = 5, **kwargs):
    """Run fn multiple times and return (median_time, result)."""
    times = []
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        times.append(time.perf_counter() - t0)
    return float(np.median(times)), result


# ---------------------------------------------------------------------------
# 1. Black-Scholes pricing
# ---------------------------------------------------------------------------

def bench_bs_pricing() -> list[BenchmarkEntry]:
    from qufin.options.european import EuropeanOption

    entries = []
    opt = EuropeanOption(s0=100, k=100, r=0.05, sigma=0.20, T=1.0, is_call=True)

    # qufin BS
    t, price = _timer(opt.bs_price, repeats=100)
    entries.append(BenchmarkEntry(
        category="BS Pricing", method="qufin.EuropeanOption.bs_price",
        n=1, wall_time_s=t, result=price,
    ))

    # Raw numpy (analytical)
    from scipy.stats import norm

    def bs_numpy(s, k, r, sigma, T):
        d1 = (np.log(s / k) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return s * norm.cdf(d1) - k * np.exp(-r * T) * norm.cdf(d2)

    t2, price2 = _timer(bs_numpy, 100, 100, 0.05, 0.20, 1.0, repeats=100)
    entries.append(BenchmarkEntry(
        category="BS Pricing", method="numpy_analytical",
        n=1, wall_time_s=t2, result=price2, reference=price,
        rel_error=abs(price2 - price) / max(abs(price), 1e-12),
    ))

    # Vectorized: price 10,000 options
    s_vec = np.random.default_rng(42).uniform(80, 120, 10_000)

    def qufin_vec():
        return np.array([
            EuropeanOption(s0=s, k=100, r=0.05, sigma=0.2, T=1.0).bs_price()
            for s in s_vec
        ])

    def numpy_vec():
        return bs_numpy(s_vec, 100, 0.05, 0.20, 1.0)

    t3, _ = _timer(qufin_vec, repeats=3)
    t4, _ = _timer(numpy_vec, repeats=3)
    entries.append(BenchmarkEntry(
        category="BS Pricing", method="qufin_vectorized_10k",
        n=10_000, wall_time_s=t3, result=0,
    ))
    entries.append(BenchmarkEntry(
        category="BS Pricing", method="numpy_vectorized_10k",
        n=10_000, wall_time_s=t4, result=0,
    ))

    return entries


# ---------------------------------------------------------------------------
# 2. Greeks computation
# ---------------------------------------------------------------------------

def bench_greeks() -> list[BenchmarkEntry]:
    from qufin.options.european import EuropeanOption

    entries = []
    opt = EuropeanOption(s0=100, k=100, r=0.05, sigma=0.20, T=1.0, is_call=True)

    for greek_name in ["bs_delta", "bs_gamma", "bs_vega", "bs_theta"]:
        fn = getattr(opt, greek_name)
        t, val = _timer(fn, repeats=100)
        entries.append(BenchmarkEntry(
            category="Greeks", method=f"qufin.{greek_name}",
            n=1, wall_time_s=t, result=val,
        ))

    return entries


# ---------------------------------------------------------------------------
# 3. Monte Carlo pricing
# ---------------------------------------------------------------------------

def bench_mc_pricing() -> list[BenchmarkEntry]:
    from qufin.options.classical.monte_carlo import european_mc

    entries = []

    for n_paths in [10_000, 100_000, 1_000_000]:
        t, res = _timer(
            european_mc,
            s=100, k=100, r=0.05, sigma=0.20, T=1.0,
            n_paths=n_paths, option_type="call", seed=42,
            repeats=3,
        )
        price = res.price if hasattr(res, 'price') else float(res)
        entries.append(BenchmarkEntry(
            category="MC Pricing", method=f"qufin.european_mc",
            n=n_paths, wall_time_s=t, result=price,
            reference=10.4506,  # BS reference
            rel_error=abs(price - 10.4506) / 10.4506,
        ))

    # Numpy-only MC baseline
    def numpy_mc(n_paths):
        rng = np.random.default_rng(42)
        z = rng.standard_normal(n_paths)
        st = 100 * np.exp((0.05 - 0.5 * 0.04) * 1.0 + 0.20 * np.sqrt(1.0) * z)
        payoff = np.maximum(st - 100, 0)
        return float(np.exp(-0.05) * np.mean(payoff))

    for n_paths in [10_000, 100_000, 1_000_000]:
        t, price = _timer(numpy_mc, n_paths, repeats=3)
        entries.append(BenchmarkEntry(
            category="MC Pricing", method="numpy_mc_baseline",
            n=n_paths, wall_time_s=t, result=price,
            reference=10.4506,
            rel_error=abs(price - 10.4506) / 10.4506,
        ))

    return entries


# ---------------------------------------------------------------------------
# 4. Portfolio optimization
# ---------------------------------------------------------------------------

def bench_portfolio() -> list[BenchmarkEntry]:
    from qufin.portfolio.classical.mean_variance import mean_variance, Objective
    from qufin.portfolio.classical.risk_parity import risk_parity
    from qufin.portfolio.classical.hrp import hrp

    entries = []

    for n_assets in [10, 30, 50, 100, 200]:
        rng = np.random.default_rng(42)
        # Factor model for realistic covariance
        factors = rng.normal(0, 0.01, (504, 3))
        loadings = rng.normal(0, 1, (3, n_assets))
        idio = rng.normal(0, 0.005, (504, n_assets))
        returns = factors @ loadings + idio
        mu = np.mean(returns, axis=0)
        cov = np.cov(returns, rowvar=False)

        # Mean-variance (min variance)
        t, res = _timer(mean_variance, mu, cov, objective=Objective.MIN_VARIANCE, repeats=3)
        entries.append(BenchmarkEntry(
            category="Portfolio Optimization", method="qufin.mean_variance",
            n=n_assets, wall_time_s=t, result=res.volatility,
        ))

        # Risk parity
        t, res = _timer(risk_parity, cov, repeats=3)
        entries.append(BenchmarkEntry(
            category="Portfolio Optimization", method="qufin.risk_parity",
            n=n_assets, wall_time_s=t, result=res.portfolio_volatility,
        ))

        # HRP
        t, res = _timer(hrp, returns, repeats=3)
        entries.append(BenchmarkEntry(
            category="Portfolio Optimization", method="qufin.hrp",
            n=n_assets, wall_time_s=t, result=float(np.std(returns @ res.weights)),
        ))

        # Raw scipy baseline for min-variance
        from scipy.optimize import minimize

        def scipy_minvar():
            n = len(mu)
            x0 = np.ones(n) / n
            res2 = minimize(
                lambda w: w @ cov @ w,
                x0, method="SLSQP",
                bounds=[(0, 1)] * n,
                constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1},
            )
            return np.sqrt(res2.fun)

        t, vol = _timer(scipy_minvar, repeats=3)
        entries.append(BenchmarkEntry(
            category="Portfolio Optimization", method="scipy.minimize_SLSQP",
            n=n_assets, wall_time_s=t, result=vol,
        ))

    return entries


# ---------------------------------------------------------------------------
# 5. Cardinality-constrained portfolio
# ---------------------------------------------------------------------------

def bench_cardinality() -> list[BenchmarkEntry]:
    from qufin.portfolio.classical.mean_variance import mean_variance, Objective

    entries = []

    for n_assets, k in [(15, 5), (25, 8), (50, 15)]:
        rng = np.random.default_rng(42)
        factors = rng.normal(0, 0.01, (504, 3))
        loadings = rng.normal(0, 1, (3, n_assets))
        returns = factors @ loadings + rng.normal(0, 0.005, (504, n_assets))
        mu = np.mean(returns, axis=0)
        cov = np.cov(returns, rowvar=False)

        t, res = _timer(
            mean_variance, mu, cov,
            objective=Objective.MIN_VARIANCE, cardinality=k,
            repeats=3,
        )
        n_nonzero = int(np.sum(np.abs(res.weights) > 1e-8))
        entries.append(BenchmarkEntry(
            category="Cardinality Portfolio", method="qufin.mean_variance+cardinality",
            n=n_assets, wall_time_s=t, result=res.volatility,
            extra={"K": k, "actual_cardinality": n_nonzero},
        ))

    return entries


# ---------------------------------------------------------------------------
# 6. VaR computation
# ---------------------------------------------------------------------------

def bench_var() -> list[BenchmarkEntry]:
    from qufin.risk.classical_var import historical_var, parametric_var, monte_carlo_var

    entries = []
    rng = np.random.default_rng(42)

    for n_obs in [252, 1260, 5040]:
        returns = rng.normal(0.0003, 0.015, n_obs)

        t, res = _timer(historical_var, returns, repeats=10)
        entries.append(BenchmarkEntry(
            category="VaR", method="qufin.historical_var",
            n=n_obs, wall_time_s=t, result=res.var,
        ))

        t, res = _timer(parametric_var, returns, repeats=10)
        entries.append(BenchmarkEntry(
            category="VaR", method="qufin.parametric_var",
            n=n_obs, wall_time_s=t, result=res.var,
        ))

        t, res = _timer(monte_carlo_var, returns, n_simulations=100_000, repeats=3)
        entries.append(BenchmarkEntry(
            category="VaR", method="qufin.monte_carlo_var",
            n=n_obs, wall_time_s=t, result=res.var,
        ))

    return entries


# ---------------------------------------------------------------------------
# 7. Credit risk (Vasicek)
# ---------------------------------------------------------------------------

def bench_credit() -> list[BenchmarkEntry]:
    from qufin.risk.credit.gaussian_copula import vasicek_analytical

    entries = []

    for n_obligors in [100, 1000, 10_000]:
        t, res = _timer(
            vasicek_analytical,
            pd=0.02, rho=0.15, lgd=0.45, confidence=0.999,
            repeats=10,
        )
        entries.append(BenchmarkEntry(
            category="Credit Risk", method="qufin.vasicek_analytical",
            n=n_obligors, wall_time_s=t, result=res["expected_loss"],
            extra={"var_99.9": res.get("var", 0)},
        ))

    return entries


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all() -> list[BenchmarkEntry]:
    all_entries = []
    benchmarks = [
        ("BS Pricing", bench_bs_pricing),
        ("Greeks", bench_greeks),
        ("MC Pricing", bench_mc_pricing),
        ("Portfolio Optimization", bench_portfolio),
        ("Cardinality Portfolio", bench_cardinality),
        ("VaR", bench_var),
        ("Credit Risk", bench_credit),
    ]

    for name, fn in benchmarks:
        print(f"  Running {name}...")
        try:
            entries = fn()
            all_entries.extend(entries)
            print(f"    {len(entries)} entries collected")
        except Exception as e:
            print(f"    FAILED: {e}")

    return all_entries


def to_markdown(entries: list[BenchmarkEntry]) -> str:
    lines = ["# Classical Benchmark Results\n"]

    categories = sorted(set(e.category for e in entries))
    for cat in categories:
        lines.append(f"\n## {cat}\n")
        lines.append("| Method | N | Time (s) | Result | Rel Error |")
        lines.append("|--------|---|----------|--------|-----------|")
        for e in entries:
            if e.category != cat:
                continue
            err_str = f"{e.rel_error:.2e}" if e.rel_error is not None else "—"
            lines.append(
                f"| {e.method} | {e.n:,} | {e.wall_time_s:.6f} | "
                f"{e.result:.6f} | {err_str} |"
            )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Run classical benchmarks")
    parser.add_argument("--output", type=str, default="benchmarks/results/classical.json")
    args = parser.parse_args()

    print("=" * 60)
    print("qufin Classical Benchmark Suite")
    print("=" * 60)

    entries = run_all()

    # Save JSON
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump([asdict(e) for e in entries], f, indent=2)
    print(f"\nJSON saved to {out_path}")

    # Save Markdown
    md_path = out_path.with_suffix(".md")
    md_path.write_text(to_markdown(entries))
    print(f"Markdown saved to {md_path}")

    # Summary
    print(f"\nTotal entries: {len(entries)}")
    print(f"Categories: {sorted(set(e.category for e in entries))}")


if __name__ == "__main__":
    main()
