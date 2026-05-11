# QAOA for Portfolio Optimization

## Background

The Quantum Approximate Optimization Algorithm (QAOA) [Farhi et al., 2014] is a variational quantum algorithm for combinatorial optimization. It alternates between a **problem unitary** $e^{-i\gamma C}$ and a **mixer unitary** $e^{-i\beta B}$ for $p$ layers.

## Mapping Portfolios to QAOA

### Step 1: Markowitz to QUBO

The Markowitz portfolio optimization problem:

$$\min_w \gamma \, w^\top \Sigma \, w - \mu^\top w$$

subject to $\|w\|_0 \leq K$ (cardinality constraint)

is mapped to a QUBO (Quadratic Unconstrained Binary Optimization):

$$\min_x \, x^\top Q \, x$$

where $x \in \{0,1\}^n$ indicates which assets are selected.

### Step 2: QUBO to Ising

The QUBO is converted to an Ising Hamiltonian via $x_i = (1 - z_i)/2$:

$$C = \sum_{i<j} J_{ij} Z_i Z_j + \sum_i h_i Z_i + \text{const}$$

### Step 3: QAOA Circuit

The QAOA circuit applies $p$ layers:

$$|\psi(\vec{\gamma}, \vec{\beta})\rangle = \prod_{l=1}^{p} e^{-i\beta_l B} \, e^{-i\gamma_l C} \, |s\rangle$$

where $|s\rangle$ is the initial state.

## Constraint Handling

### Cardinality Constraint

The constraint "select exactly $K$ of $N$ assets" is enforced via:

1. **Penalty method**: Add $\lambda(\sum_i x_i - K)^2$ to the QUBO objective
2. **Hamming-weight-preserving mixer**: Use XY mixer instead of X mixer
3. **Dicke state initialization**: Start in $|D_n^K\rangle$ (uniform superposition over Hamming-weight-$K$ states)

qufin supports all three approaches.

### XY Mixer

The XY mixer preserves the Hamming weight of the state:

$$B_{XY} = \sum_{\langle i,j \rangle} (X_i X_j + Y_i Y_j)$$

This is implemented as SWAP-like interactions between neighboring qubits.

**Ring topology** (`xy_ring`): Nearest-neighbor on a ring — $O(n)$ depth per layer.

**Full topology** (`xy_full`): All-to-all connections — $O(n^2)$ depth but better mixing.

### Dicke State Initialization

The Dicke state $|D_n^K\rangle$ is prepared using the Bartschi-Eidenbenz Split-and-Cyclic-Shift (SCS) algorithm:

$$|D_n^K\rangle = \frac{1}{\sqrt{\binom{n}{K}}} \sum_{|x|=K} |x\rangle$$

This ensures the initial state satisfies the cardinality constraint.

## CVaR Objective

Instead of optimizing the expected value, qufin supports **Conditional Value at Risk** as the QAOA objective [Barkoutsos et al., 2020]:

$$\text{CVaR}_\alpha = \mathbb{E}[C \,|\, C \leq F^{-1}(\alpha)]$$

Setting $\alpha < 1$ focuses optimization on the tail (best solutions), improving convergence.

## Implementation Choices

| Choice | qufin Default | Rationale |
|--------|--------------|-----------|
| Encoding | One-hot | Direct binary selection, no auxiliary qubits |
| Mixer | XY-ring | Preserves cardinality, $O(n)$ depth |
| Initial state | Dicke $\|D_n^K\rangle$ | Feasible from the start |
| CVaR $\alpha$ | 1.0 (mean) | CVaR < 1 optional for harder problems |
| Optimizer | COBYLA | Gradient-free, noise-robust |
| Shots | 8192 | Good statistics for CVaR estimation |

## References

- Farhi, Goldstone, Gutmann (2014). "A Quantum Approximate Optimization Algorithm." [arXiv:1411.4028](https://arxiv.org/abs/1411.4028)
- Barkoutsos et al. (2020). "Improving Variational Quantum Optimization using CVaR." [Quantum 4, 256](https://quantum-journal.org/papers/q-2020-04-20-256/)
- Bartschi & Eidenbenz (2019). "Deterministic Preparation of Dicke States." [arXiv:1904.07358](https://arxiv.org/abs/1904.07358)
- Hadfield et al. (2019). "From the Quantum Approximate Optimization Algorithm to a Quantum Alternating Operator Ansatz." [Algorithms 12(2)](https://www.mdpi.com/1999-4893/12/2/34)
