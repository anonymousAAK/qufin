"""Generate synthetic price paths (GBM and Heston stochastic volatility).

Run:
    python examples/synthetic_market_data.py

These generators feed qufin's Monte-Carlo pricers, backtests, and qGAN
training without needing a live market-data connection.
"""

from __future__ import annotations

import numpy as np

from qufin.data.synthetic import gbm_paths, heston_paths


def main() -> None:
    # Geometric Brownian motion: returns paths of shape (n_paths, n_steps + 1).
    gbm = gbm_paths(s0=100, mu=0.08, sigma=0.2, T=1.0, n_steps=252, n_paths=10_000)
    terminal = np.asarray(gbm)[:, -1]
    print("GBM  paths:", np.asarray(gbm).shape)
    print(f"  E[S_T]   = {terminal.mean():.2f}  (analytic {100 * np.exp(0.08):.2f})")
    print(f"  std[S_T] = {terminal.std():.2f}")

    # Heston stochastic volatility: returns (prices, variances).
    prices, variances = heston_paths(
        s0=100, v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7,
        mu=0.08, T=1.0, n_steps=252, n_paths=10_000, seed=42,
    )
    print("\nHeston paths:", np.asarray(prices).shape)
    print(f"  E[S_T]    = {np.asarray(prices)[:, -1].mean():.2f}")
    print(f"  E[v_T]    = {np.asarray(variances)[:, -1].mean():.4f}  (long-run theta=0.04)")


if __name__ == "__main__":
    main()
