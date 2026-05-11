# QUBO Formulation

## Overview

The QUBO (Quadratic Unconstrained Binary Optimization) is the bridge between financial optimization problems and quantum hardware. qufin provides a flexible QUBO builder with multiple constraint types and encodings.

## Markowitz to QUBO

### Unconstrained Problem

The Markowitz objective:

$$\min_w \; \gamma \, w^\top \Sigma \, w - \mu^\top w$$

With binary variables $x_i \in \{0, 1\}$ (asset selected or not):

$$\min_x \; x^\top Q \, x$$

where:

$$Q_{ii} = \gamma \Sigma_{ii} - \mu_i, \qquad Q_{ij} = \gamma \Sigma_{ij} \; (i \neq j)$$

### Cardinality Constraint

Select exactly $K$ out of $N$ assets. Added as a quadratic penalty:

$$Q \mathrel{+}= \lambda \left(\sum_i x_i - K\right)^2 = \lambda \left(\sum_{i,j} x_i x_j - 2K \sum_i x_i + K^2\right)$$

The penalty weight $\lambda$ should be large enough to make infeasible solutions suboptimal. qufin's default: `budget_penalty = 1e4`.

### Sector Constraints

Limit the number of assets per sector $s$:

$$Q \mathrel{+}= \lambda_s \left(\sum_{i \in s} x_i - C_s\right)^2$$

where $C_s$ is the sector capacity.

### Turnover Constraint

Penalize deviation from previous allocation $x^{prev}$:

$$Q_{ii} \mathrel{+}= \tau (1 - 2 x_i^{prev})$$

This makes it cheaper to hold current positions and expensive to change.

## Encodings

### One-Hot Encoding

Each asset gets one qubit: $x_i = 1$ means asset $i$ is selected.

- **Qubits**: $N$ (one per asset)
- **Weights**: Equal weight $1/K$ for selected assets
- **Best for**: Selection problems (which assets to hold)

### Binary Encoding

Each asset gets $b$ qubits encoding its weight level:

$$w_i = \sum_{j=0}^{b-1} 2^j \, x_{ib+j} \Big/ \left(2^b - 1\right)$$

- **Qubits**: $N \times b$
- **Weights**: Granularity of $1/(2^b - 1)$
- **Best for**: Weight allocation (how much of each asset)

### Qubit Cost Comparison

```python
from qufin.portfolio.encodings import qubit_cost_table

table = qubit_cost_table([10, 20, 50, 100])
# Shows qubits needed for one-hot, binary (3-bit), unary (4-level)
```

| Assets | One-Hot | Binary (3-bit) | Unary (4-level) |
|--------|---------|----------------|-----------------|
| 10 | 10 | 30 | 40 |
| 20 | 20 | 60 | 80 |
| 50 | 50 | 150 | 200 |
| 100 | 100 | 300 | 400 |

## QUBO to Ising

For quantum execution, the QUBO is mapped to an Ising Hamiltonian via $x_i = (1 - Z_i) / 2$:

$$H = \sum_{i < j} J_{ij} Z_i Z_j + \sum_i h_i Z_i + c$$

This is handled automatically by `PortfolioQUBO.build_matrix()`.

## Usage

```python
from qufin.portfolio.qubo import PortfolioQUBO

qubo = PortfolioQUBO(
    mu=mu,                    # Expected returns (N,)
    cov=cov,                  # Covariance matrix (N, N)
    gamma=1.0,                # Risk aversion
    cardinality=5,            # Select 5 assets
    sector_map={0: [0,1,2], 1: [3,4,5], 2: [6,7,8,9]},
    sector_caps={0: 2, 1: 2, 2: 2},
    turnover_penalty=0.01,
    transaction_cost=0.005,
    previous_weights=prev_w,
    budget_penalty=1e4,
    encoding="one_hot",
)

Q = qubo.build_matrix()               # (n_qubits, n_qubits)
obj = qubo.evaluate("0110100000")      # Evaluate a specific bitstring
w = qubo.decode_weights("0110100000")  # Get portfolio weights
feas = qubo.feasibility_check("0110100000")  # Check constraints
```

## References

- Barkoutsos et al. (2020). "Improving Variational Quantum Optimization using CVaR." [Quantum 4, 256](https://quantum-journal.org/papers/q-2020-04-20-256/)
- Mugel et al. (2022). "Dynamic Portfolio Optimization with Real Datasets Using Multi-Period Quantum Annealing." [IEEE Access](https://ieeexplore.ieee.org/document/9726728)
- Brandhofer et al. (2023). "Benchmarking the Performance of Portfolio Optimization with QAOA." [Quantum Sci. Technol. 8](https://iopscience.iop.org/article/10.1088/2058-9565/acb4b0)
