# Hedging

qufin provides classical and quantum hedging strategies for derivatives risk management.

## Delta Hedging

The simplest hedging strategy: maintain a position in the underlying that offsets the option's delta.

```python
from qufin.hedging.delta import DeltaHedger
from qufin.options.classical.black_scholes import bs_greeks

# Hedge a short call position
hedger = DeltaHedger(rebalance_freq="daily")
greeks = bs_greeks(s=100, k=105, sigma=0.2, r=0.05, T=1.0)
hedge_ratio = greeks["delta"]  # Buy this many shares per option sold
```

### Discrete Delta Hedging Backtest

```python
import numpy as np
from qufin.hedging.delta import delta_hedge_backtest
from qufin.data.synthetic import gbm_paths

# Simulate underlying price paths
paths = gbm_paths(s0=100, mu=0.08, sigma=0.2, T=1.0, n_steps=252, n_paths=1000)

# Backtest delta hedging with transaction costs
results = delta_hedge_backtest(
    paths=paths,
    strike=105,
    sigma=0.2,
    r=0.05,
    T=1.0,
    rebalance_freq=1,  # daily
    tx_cost=0.001,     # 10 bps per trade
)
print(f"Mean P&L: {results['mean_pnl']:.4f}")
print(f"P&L Std:  {results['std_pnl']:.4f}")
```

## Deep Hedging

Neural network-based hedging that learns optimal strategies directly from data, accounting for transaction costs and market frictions.

!!! note "Requires PyTorch"
    Install with `pip install "qufin[ml]"` to enable deep hedging.

```python
from qufin.hedging.deep_hedging import DeepHedger

hedger = DeepHedger(
    n_layers=3,
    hidden_dim=64,
    n_steps=30,        # hedging steps
    tx_cost=0.001,
    risk_measure="cvar",  # optimize for CVaR of P&L
    alpha=0.95,
)

# Train on simulated paths
hedger.fit(paths_train, strikes=[100, 105, 110], epochs=200)

# Evaluate
pnl = hedger.evaluate(paths_test, strike=105)
print(f"CVaR(95%): {np.percentile(pnl, 5):.4f}")
```

## Quantum Deep Hedging

Replaces the classical neural network with a variational quantum circuit (VQC) as the policy network. Research has shown VQC-based hedging can match classical performance with fewer trainable parameters.

```python
from qufin.hedging.quantum_deep_hedging import QuantumDeepHedger
from qufin.backends.qiskit_backend import QiskitAerBackend

backend = QiskitAerBackend(shots=1024)
hedger = QuantumDeepHedger(
    n_qubits=4,
    n_layers=3,
    backend=backend,
    tx_cost=0.001,
)

hedger.fit(paths_train, strike=105, epochs=100)
```

## RL-Quantum Hedging

Reinforcement learning agent with a quantum policy network for dynamic hedging decisions.

```python
from qufin.hedging.rl_quantum import QuantumRLHedger

agent = QuantumRLHedger(
    n_qubits=4,
    n_layers=2,
    learning_rate=0.001,
    gamma=0.99,         # discount factor
    risk_aversion=0.5,
)

# Train with PPO
agent.train(env="heston", n_episodes=5000, n_steps=252)
```

## Comparison

| Strategy | Strengths | Limitations |
|---|---|---|
| Delta hedging | Simple, model-based, no training | Assumes BS dynamics, ignores tx costs |
| Deep hedging | Learns from data, handles frictions | Requires training data, black box |
| Quantum deep hedging | Fewer parameters, potential speedup | NISQ noise, limited qubits |
| RL-quantum hedging | Adaptive, handles complex dynamics | Training instability, sample inefficient |
