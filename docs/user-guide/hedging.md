# Hedging

qufin provides classical and quantum hedging strategies for derivatives risk management.

## Delta Hedging

The simplest hedging strategy: maintain a position in the underlying that offsets the option's delta.

```python
from qufin.hedging.delta import DeltaHedger
from qufin.options.classical.black_scholes import price_and_greeks

# Current hedge ratio for a call position
greeks = price_and_greeks(s=100, k=105, sigma=0.2, r=0.05, T=1.0, option_type="call")
hedge_ratio = greeks.delta  # Buy this many shares per option sold

# Single-path discrete delta-hedging simulation
hedger = DeltaHedger(is_call=True)
result = hedger.hedge(spot=100, strike=105, r=0.05, sigma=0.2, T=1.0, n_rebalances=252)
print(f"P&L: {result.pnl:.4f}  hedging error: {result.hedging_error:.4f}")
```

### Discrete Delta Hedging Backtest

```python
import numpy as np
from qufin.hedging.delta import DeltaHedger

# Aggregate P&L over many independent hedging paths.
hedger = DeltaHedger(is_call=True)
pnls = [
    hedger.hedge(spot=100, strike=105, r=0.05, sigma=0.2, T=1.0,
                 n_rebalances=252, seed=s).pnl
    for s in range(1000)
]
pnls = np.array(pnls)
print(f"Mean P&L: {pnls.mean():.4f}")
print(f"P&L Std:  {pnls.std():.4f}")
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

backend = QiskitAerBackend(method="automatic", seed=42)  # shots are passed to run()
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
import numpy as np
from qufin.hedging.rl_quantum import QuantumPolicy, QuantumPolicyConfig

# A variational quantum circuit acts as a discrete-action policy network.
policy = QuantumPolicy(QuantumPolicyConfig(n_qubits=4, n_layers=2, n_actions=3))

# Initialise trainable parameters and evaluate action probabilities for a state.
params = np.random.default_rng(42).uniform(0, 2 * np.pi, policy.n_params)
state = np.array([0.1, -0.2, 0.05, 0.0])  # e.g. [moneyness, delta, time, inventory]
action_probs = policy.select_action(state, params)
print("Action probabilities:", action_probs.round(4))
```

## Comparison

| Strategy | Strengths | Limitations |
|---|---|---|
| Delta hedging | Simple, model-based, no training | Assumes BS dynamics, ignores tx costs |
| Deep hedging | Learns from data, handles frictions | Requires training data, black box |
| Quantum deep hedging | Fewer parameters, potential speedup | NISQ noise, limited qubits |
| RL-quantum hedging | Adaptive, handles complex dynamics | Training instability, sample inefficient |
