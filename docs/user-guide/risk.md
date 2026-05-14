# Risk Management

qufin provides comprehensive risk analytics including VaR, CVaR, stress testing, and counterparty risk.

## Value at Risk (VaR)

Three estimation methods, all returning `VaRResult` with dollar amounts:

```python
from qufin.risk.classical_var import historical_var, parametric_var, monte_carlo_var

returns = ...  # (T,) array of portfolio returns

# Historical (empirical percentile)
result = historical_var(returns, confidence=0.95, portfolio_value=10_000_000)
print(f"95% VaR: ${result.var_dollar:,.0f}")
print(f"ES:      ${result.es_dollar:,.0f}")

# Parametric (Gaussian assumption)
result = parametric_var(returns, confidence=0.99)

# Monte Carlo (simulated)
result = monte_carlo_var(returns, confidence=0.99, n_simulations=1_000_000, horizon=10)
```

### Portfolio VaR

Convenience function that combines weights with asset returns:

```python
from qufin.risk.classical_var import portfolio_var

result = portfolio_var(
    asset_returns,  # (T, N) matrix
    weights,        # (N,) vector
    confidence=0.95,
    method="historical",  # or "parametric", "monte_carlo"
)
```

## Conditional VaR (CVaR / Expected Shortfall)

```python
from qufin.risk.cvar import cvar_from_samples, portfolio_cvar, CVaRObjective

# Simple CVaR
cvar = cvar_from_samples(portfolio_returns, alpha=0.05)

# Portfolio CVaR
cvar = portfolio_cvar(asset_returns, weights, alpha=0.05)

# CVaR as optimization objective (for QAOA/VQE)
obj = CVaRObjective(alpha=0.5)
value = obj.evaluate(costs, counts)
```

### Ascending CVaR Schedule

Progressively shift from tail to mean during optimization:

```python
from qufin.risk.cvar import AscendingCVaR

schedule = AscendingCVaR(alpha_start=0.1, alpha_end=1.0, n_steps=100)
for step in range(100):
    objective = schedule.get_objective()
    # ... use in optimizer ...
    schedule.step()
```

## Stress Testing

Pre-built scenarios based on historical crises:

```python
from qufin.risk.stress import stress_test_suite, SCENARIO_LIBRARY

# Run all scenarios
results = stress_test_suite(
    portfolio_value=50_000_000,
    weights=[0.6, 0.2, 0.1, 0.1],  # equity, rates, vol, spreads
)

for name, res in results.items():
    print(f"{name}: ${res['total_pnl']:,.0f} ({res['pct_loss']:.1%})")
```

**Built-in scenarios:**

| Scenario | Equity Shock | Rates Shock | Vol Shock |
|----------|-------------|-------------|-----------|
| Black Monday 1987 | -22.6% | -50 bps | +150% |
| GFC 2008 | -38.0% | -200 bps | +200% |
| COVID 2020 | -34.0% | -150 bps | +400% |
| Rate Hiking 2022 | -19.0% | +300 bps | +50% |

Custom scenarios:

```python
from qufin.risk.stress import StressScenario, apply_stress

custom = StressScenario(
    name="Tail Event",
    date="custom",
    equity_shock=-0.50,
    rates_shock=-300,
    vol_shock=5.0,
    spread_shock=500,
)
result = apply_stress(portfolio_value, weights, custom)
```

## Counterparty Risk (CVA)

```python
from qufin.risk.counterparty import (
    CounterpartyExposure, compute_cva, compute_ead_sa_ccr, portfolio_cva,
)

exposure = CounterpartyExposure(
    name="JPMorgan IRS",
    notional=100_000_000,
    pd=0.001,
    lgd=0.45,
    exposure_profile=np.array([...]),  # time series of expected exposure
)

cva = compute_cva(exposure, risk_free_rate=0.03)
ead = compute_ead_sa_ccr(notional=1e8, mtm=5e6, add_on_factor=0.01)
```

## Credit Risk

### Gaussian Copula (Vasicek)

```python
from qufin.risk.credit.gaussian_copula import vasicek_analytical

result = vasicek_analytical(pd=0.02, rho=0.15, lgd=0.45, confidence=0.999)
print(f"Expected Loss: {result['expected_loss']:.4f}")
print(f"VaR 99.9%:     {result['var']:.4f}")
```

## Quantum Stress Testing

Run quantum-enhanced stress tests with predefined crisis scenarios:

```python
from qufin.risk.quantum_stress import QuantumStressTester, classical_stress_test

tester = QuantumStressTester(backend=backend)
results = tester.run_scenarios(portfolio_weights, portfolio_value=1_000_000)

# Classical alternative
results = classical_stress_test(portfolio_weights, portfolio_value=1_000_000)
```

**Predefined scenarios:** GFC 2008, COVID 2020, Rate Hike 2022.
