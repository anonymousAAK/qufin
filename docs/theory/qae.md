# Quantum Amplitude Estimation

## The Problem

Given a unitary $\mathcal{A}$ that prepares:

$$\mathcal{A}|0\rangle = \sqrt{a}\,|\Psi_1\rangle|1\rangle + \sqrt{1-a}\,|\Psi_0\rangle|0\rangle$$

estimate $a = \sin^2(\theta)$ where $\theta \in [0, \pi/2]$.

**Application to option pricing**: Encode the option payoff into the amplitude $a$, then $\text{Price} = a \times \text{rescale\_factor}$.

## The Grover Operator

QAE uses the Grover operator:

$$\mathcal{Q} = -\mathcal{A} \, S_0 \, \mathcal{A}^\dagger \, S_\chi$$

where:
- $S_\chi$ marks "good" states (flips phase of $|1\rangle$ on objective qubits)
- $S_0$ is the zero-state reflection
- The **minus sign** is critical (Brassard's convention): without it, eigenvalues are $e^{\pm 4i\theta}$ instead of $e^{\pm 2i\theta}$

!!! warning "Global Phase Matters"
    The Grover operator **must** include the global phase of $-1$. This is a common bug in implementations. qufin applies `qc.global_phase += pi` to ensure correctness.

## QAE Variants

### Canonical QAE (QPE-based)

Uses Quantum Phase Estimation on $\mathcal{Q}$ to extract $\theta$:

1. Prepare $\mathcal{A}|0\rangle$ in the target register
2. Apply QPE with $n_{eval}$ qubits on $\mathcal{Q}$
3. Measure and read phase $\tilde{\theta} = y / 2^{n_{eval}}$
4. Estimate $a = \sin^2(\pi \tilde{\theta})$

**Accuracy**: $O(2^{-n_{eval}})$

**Circuit depth**: $O(2^{n_{eval}})$ (requires controlled-$\mathcal{Q}^{2^k}$)

### Iterative QAE (IQAE)

[Grinko et al., 2021] — No QPE overhead, uses adaptive schedule:

1. Start with $\theta \in [0, \pi/2]$
2. At round $i$: choose power $k_i$, run circuit with $\mathcal{Q}^{k_i}$
3. Measure probability $p_i$ that objective qubit is $|1\rangle$
4. **Key step**: $p_i = \sin^2((2k_i + 1)\theta)$ has $O(k_i)$ solutions. Enumerate ALL candidate $\theta$ intervals from both branches and intersect with current interval.
5. Repeat until interval width $< \epsilon$

**Accuracy**: $\epsilon$ (user-specified)

**Circuit depth**: Adaptive, typically $O(1/\epsilon)$

!!! note "Multi-Branch Resolution"
    The equation $\sin^2((2k+1)\theta) = p$ has multiple solutions. For each $k$, there are two branches:

    - Branch A: $\theta = (\arcsin(\sqrt{p}) + j\pi) / (2k+1)$
    - Branch B: $\theta = (\pi - \arcsin(\sqrt{p}) + j\pi) / (2k+1)$

    for $j = 0, 1, \ldots, 2k$. All must be checked.

### Maximum Likelihood AE (MLAE)

Run circuits at multiple depths $k_1, k_2, \ldots$ and fit $\theta$ via MLE:

$$\hat{\theta} = \arg\max_\theta \prod_i \text{Binomial}(h_i; N_i, \sin^2((2k_i+1)\theta))$$

### Faster QAE (FQAE)

Grover-search-based acceleration of QAE with provable $O(1/\epsilon)$ queries.

## Comparison

| Variant | Extra Qubits | Max Depth | Queries to $\epsilon$ | NISQ Friendly |
|---------|-------------|-----------|----------------------|---------------|
| Canonical | $n_{eval}$ | $2^{n_{eval}}$ | $O(1/\epsilon)$ | No |
| IQAE | 0 | Adaptive | $O(1/\epsilon)$ | Yes |
| MLAE | 0 | Fixed | $O(1/\epsilon)$ | Yes |
| FQAE | 0 | Adaptive | $O(1/\epsilon)$ | Moderate |

## Finance Applications

| Application | $\mathcal{A}$ encodes | Amplitude $a$ represents |
|-------------|----------------------|--------------------------|
| European option | GBM terminal price | Discounted expected payoff |
| Asian option | Path-average price | Average payoff |
| Barrier option | Path with barrier check | Conditional payoff |
| VaR/CVaR | Loss distribution | Tail probability |
| Credit risk | Default correlation | Portfolio loss |

## References

- Brassard et al. (2002). "Quantum Amplitude Amplification and Estimation." [AMS Contemporary Mathematics 305](https://arxiv.org/abs/quant-ph/0005055)
- Grinko et al. (2021). "Iterative Quantum Amplitude Estimation." [npj Quantum Information 7, 52](https://arxiv.org/abs/1912.05559)
- Suzuki et al. (2020). "Amplitude estimation without phase estimation." [Quantum Information Processing 19](https://arxiv.org/abs/1904.10246)
- Stamatopoulos et al. (2020). "Option Pricing using Quantum Computers." [Quantum 4, 291](https://arxiv.org/abs/1905.02666)
- Chakrabarti et al. (2021). "A Threshold for Quantum Advantage in Derivative Pricing." [Quantum 5, 463](https://arxiv.org/abs/2012.03819)
