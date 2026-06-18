"""Classical option pricing with Black-Scholes: price and Greeks.

Run:
    python examples/option_pricing_classical.py

This is the classical baseline that qufin's quantum amplitude-estimation
pricers (see ``qufin.options.amplitude_estimation``) are compared against.
"""

from __future__ import annotations

from qufin.options.classical.black_scholes import (
    call_price,
    delta,
    gamma,
    put_price,
    rho,
    theta,
    vega,
)


def main() -> None:
    s, k, sigma, r, t = 100.0, 105.0, 0.2, 0.05, 1.0

    print(f"European options on S={s}, K={k}, sigma={sigma}, r={r}, T={t}y\n")
    print(f"  Call price : {call_price(s=s, k=k, sigma=sigma, r=r, T=t):.4f}")
    print(f"  Put price  : {put_price(s=s, k=k, sigma=sigma, r=r, T=t):.4f}")
    print("\nGreeks (call):")
    print(f"  delta : {delta(s=s, k=k, sigma=sigma, r=r, T=t):+.4f}")
    print(f"  gamma : {gamma(s=s, k=k, sigma=sigma, r=r, T=t):+.4f}")
    print(f"  vega  : {vega(s=s, k=k, sigma=sigma, r=r, T=t):+.4f}")
    print(f"  theta : {theta(s=s, k=k, sigma=sigma, r=r, T=t):+.4f}")
    print(f"  rho   : {rho(s=s, k=k, sigma=sigma, r=r, T=t):+.4f}")


if __name__ == "__main__":
    main()
