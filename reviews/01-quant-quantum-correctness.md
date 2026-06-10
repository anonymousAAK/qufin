# qufin — Quant/Quantum Correctness Review

**Reviewer:** principal quant + quantum-algorithms (skeptical correctness audit)
**Branch:** `claude/determined-fermi-y8m78`  · commit `a5b185c`
**Date:** 2026-06-04
**Environment:** Python 3.11; numpy 2.4.6, scipy 1.17.1, pandas 3.0.3, qiskit 2.4.1, qiskit-aer 0.17, scikit-learn 1.9 (all newer than pyproject minimums). fastapi/torch/pennylane/cirq/braket/dwave not installed.
**Scope:** numerical/financial/quantum math in `options/**`, `risk/**`, `portfolio/classical/**`, `portfolio/{qubo,optimizers,encodings,mixers}`, `backends/{error_mitigation,noise_models}`.

---

## Severity counts

| Severity | Count |
|----------|-------|
| Critical | 3 |
| High     | 3 |
| Medium   | 4 |
| Low      | 3 |
| **Total**| **13** |

## Verdict

**The "mathematically correct" / "production-grade" claims do NOT hold for the quantum option-pricing and quantum-risk paths.** The *classical* core is genuinely strong — Black-Scholes (price/Greeks/parity/σ→0 limit), CRR binomial→BS convergence, Monte Carlo, parametric/historical VaR & ES, Cornish-Fisher, Vasicek/Gaussian-copula credit, mean-variance, Black-Litterman, HRP, GARCH, and von-Neumann entropy all reproduce textbook/closed-form values to within expected tolerances. But the headline *quantum* deliverables are broken by a **systematic qubit-endianness bug in every payoff/threshold comparator oracle** (price encoded ~2.5× too high in European QAE; 95% quantum VaR returns 2.59 vs true 1.645) and by **state-preparation circuits built with the non-invertible `Initialize` instruction**, which makes the Grover operator — and therefore *all* QAE estimators (canonical, IQAE, MLAE) — crash on European/multi-asset/MRQAE pricing under qiskit 2.x. PEC error-mitigation is a non-functional placeholder mislabeled as Temme et al. (2017), and "quantum HHL" is a classical `np.linalg.solve` in disguise. A library marketed as "trillion-dollar enterprise grade" with `Development Status :: 5 - Production/Stable` cannot ship these as working quantum algorithms.

---

## Critical findings

### C1. Quantum VaR returns badly wrong numbers (endianness bug in comparator oracle)
**Severity: Critical**
**File:** `src/qufin/risk/quantum_var.py:104-119` (`_build_tail_probability_problem`), same pattern at `:163-192` (`_build_conditional_value_problem`).

The comparator marks a state by applying X-gates conditioned on `bits = format(i, f"0{n_q}b")` (MSB-first) with `qc.x(b_idx)` indexing qubit `b_idx` left-to-right. But the distribution is loaded with `StatePreparation(amplitudes)`, which places value-index `i` at the standard qiskit basis state where **qubit 0 is the LSB**. The comparator therefore conditions on the **bit-reversed** value index, so it marks the wrong states.

**Evidence (statevector of the oracle, N(0,1), n_qubits=4, threshold x=1.0):**
```
Intended marked value-indices (value>1): [11, 12, 13, 14, 15]  sum p = 0.1130
Actually marked by circuit (anc=1):       {3, 7, 11, 13, 15}    sum p = 0.2642   <-- bit-reversals of intended
```
12=1100→0011=3, 14=1110→0111=7, etc. End-to-end:
```
x=1.0: EXACT circuit amplitude = 0.2642   TRUE discrete P(X>1)=0.1130   IQAE=0.2639
x=2.0: EXACT circuit amplitude = 0.2184   TRUE discrete P(X>2)=0.0214   IQAE=0.2183
quantum_var(95%) VaR = 2.5898   (true normal 95% VaR ≈ 1.645)   ES = 2.6984
```
The IQAE estimator is faithful (0.2642 vs 0.2639) — the **oracle** is wrong. (For n_qubits ≤ 2 it accidentally passes because bit-reversal is a no-op on the symmetric tail; this is why unit tests miss it.)

**Impact:** A user computing 95% VaR on a standard normal loss gets **2.59 instead of 1.645 — a 57% overstatement**, and the error is silent (looks plausible). Both VaR and ES are affected. Quantum VaR is unusable for n_qubits ≥ 3.

**Fix:** Reverse the bit string before applying the X-mask, i.e. iterate `for b_idx, b in enumerate(reversed(bits))`, or condition the multi-controlled gate on qubit `n_q-1-b_idx`. Use Qiskit's `IntegerComparator` (qiskit-algorithms) instead of a hand-rolled per-state comparator. Add a statevector regression test asserting `P(anc=1) == sum(p_i for value_i > threshold)` for n_qubits ∈ {3,4,5}.

---

### C2. European QAE pricing is doubly broken: (a) `Initialize` is non-invertible → Grover operator crashes; (b) same endianness bug → ~2.5× price error
**Severity: Critical**
**Files:** `src/qufin/options/amplitude_estimation/european_qae.py:94` (`qc.initialize(...)`), comparator at `:115-142`. Also affects `multi_asset_qae.py:283`, `mrqae.py:404`, and `distributions.py:208` (`build_loading_circuit`).

**(a) Crash.** `build_european_estimation_problem` builds `A` with `qc.initialize(amplitudes, ...)`. The Grover operator (`estimation_problem.py:91`) calls `self.state_preparation.inverse()`. Under qiskit 2.x, `Initialize` is non-unitary (it contains a reset) and raises:
```
qiskit.circuit.exceptions.CircuitError: 'Initialize is not unitary thus can not be inverted.
If you want an invertible state preparation, use StatePreparation instead.'
```
So canonical QAE, IQAE, and MLAE all crash the moment they try `build_grover_operator()` on a European/multi-asset/MRQAE problem. (The fix pattern already exists in the codebase: `risk/quantum_var.py:_decomposed_state_prep` uses `StatePreparation` + transpile.)

**(b) Wrong amplitude (independent of the crash).** The payoff comparator uses the same MSB-first `format(i,...)` + `qc.x(b_idx)` mask as C1, while the distribution is loaded LSB-first → bit-reversed marking. Statevector of `A` for S=K=100, σ=0.2, r=0.05, T=1, n_qubits=3:
```
European QAE circuit amplitude (statevector) = 0.31549
CORRECT amplitude  (Σ p_i·payoff_i/maxpayoff) = 0.12659
=> circuit price = 26.34   correct grid price = 10.57   (BS exact = 10.45)
=> overpriced by factor 2.49×
```

**Impact:** The flagship "quantum option pricing" either throws on any real backend, or (after a naive Initialize→StatePreparation swap) returns ~26 for a call worth ~10.45.

**Fix:** (1) Replace `qc.initialize` with `StatePreparation` everywhere a Grover operator/inverse is needed (european_qae, multi_asset_qae, mrqae, distributions.build_loading_circuit; also `options/asian.py:151`, `options/barrier.py:189`). (2) Fix the comparator bit order as in C1. Add an end-to-end test: `canonical/IQAE/MLAE on european problem → price within tol of BS`.

---

### C3. PEC ("Probabilistic Error Cancellation") is a non-functional placeholder mislabeled as Temme et al. (2017)
**Severity: Critical** (correctness of a claimed mitigation method; produces a high-variance estimator with no error-cancellation property)
**File:** `src/qufin/backends/error_mitigation.py:593-692` (`pec_mitigate`), supporting `characterize_noise_channel:403-489`, `quasi_probability_decomposition:492-545`.

`pec_mitigate` does not implement PEC. For every sample it runs the **same** noisy `meas_circ` (line 679) — it never inserts the inverse-channel basis operations that PEC is built on. The "quasi-probability sampling" reduces to drawing `n_flips ~ Binomial(n_gates, ε/(1+2ε))` and multiplying the observable by `sign·γ` with `γ=(1+2ε)^{n_gates}`. Because the underlying samples are identical in distribution, `E[sign·γ·val] = E[sign]·γ·E[val]`, and `E[sign]=Σ_{odd k} P(k)·(−1)` does **not** equal `1/γ`, so the estimator is **biased and does not cancel noise**; it only inflates variance by γ. The genuinely-computed `characterize_noise_channel` PTM and `quasi_probability_decomposition` coefficients are never consumed by `pec_mitigate`.

**Impact:** Any user invoking PEC for "unbiased expectation values" gets a noisy, biased number with sampling overhead but no mitigation. The docstring explicitly claims "unbiased expectation value ... Temme et al. (2017)".

**Fix:** Either implement real PEC (sample gate replacements from the quasi-probability decomposition computed from the characterized PTMs, accumulate `sign·γ·observable`), or downgrade the docstring/claims to "experimental stub" and remove it from the public mitigation API. At minimum, stop advertising it as Temme-et-al PEC.

---

## High findings

### H1. QAOA/VQE report bit-reversed portfolio weights (qiskit endianness vs `QUBO.evaluate`/`decode_weights`)
**Severity: High**
**Files:** `src/qufin/portfolio/qubo.py:220-249` (`evaluate`, `decode_weights`); consumed by `optimizers/qaoa.py:158-162` and `optimizers/vqe.py:157-161` via `final_result.most_frequent`.

`QUBO.evaluate`/`decode_weights` parse a bitstring left-to-right: `bitstring[0] → asset 0`. But the QAOA/VQE circuits drive asset *i* on qubit *i*, and qiskit returns counts MSB-first, so **qubit 0 (asset 0) is the rightmost character**. Hence `most_frequent='001'` (qubit 0 = 1) is decoded as "asset 2 selected".

**Evidence:**
```
x-gate on qubit0 → qiskit counts {'001': 100}    (qubit0 is rightmost)
QUBO.evaluate('100') = -0.49   decode('100') = [1,0,0]   (treats leftmost as asset0)
QUBO.evaluate('001') =  0.01   decode('001') = [0,0,1]
mu=[1,0,0]: a circuit that correctly selects asset0 (qubit0=1, bits='001')
            → decode_weights('001') = [0,0,1]  → reports weight on asset 2
```
Because the Markowitz QUBO matrix is symmetric and the H-init/X-mixer are permutation-symmetric, the *objective value* returned is still a valid QUBO objective (for the reversed labeling), which masks the bug — but the returned `weights` map to the **wrong assets** whenever the problem is asymmetric across assets (distinct `mu`, asymmetric `cov`, sectors, turnover with `previous_weights`).

**Impact:** A production user reads `result.weights` and allocates capital to the **reversed** asset set. `exhaustive_solve` is internally consistent (it builds bitstrings with the same convention it evaluates), so brute-force is correct and will silently *disagree on the selected assets* with QAOA/VQE even when objective values look comparable.

**Fix:** Standardize on one convention. Simplest: in `evaluate`/`decode_weights`/`feasibility_check`, reverse the incoming bitstring (`bitstring[::-1]`) so position↔qubit↔asset are aligned with qiskit, OR reverse `most_frequent` before passing to `evaluate`/`decode_weights` in the optimizers. Add a test with asymmetric `mu` asserting QAOA-selected assets == exhaustive-selected assets.

### H2. `most_frequent` ≠ lowest-energy: QAOA/VQE can return a suboptimal sampled bitstring
**Severity: High**
**Files:** `optimizers/qaoa.py:158`, `optimizers/vqe.py:157`.

The reported solution is `final_result.most_frequent` — the modal bitstring — not the minimum-energy bitstring among the samples. For shallow `p`/imperfect optimization the modal sample is frequently *not* the best one observed.

**Evidence (n=4 cardinality-2, Aer, p=2, COBYLA):**
```
Exhaustive best: '0110' obj=-1.10152   (top-4: 0110 -1.1015, 0011 -1.0922, 1010 -1.0315, 0101 -0.8886)
QAOA returns:    '1010' obj=-1.0315    (3rd best)
```
Even when a better feasible bitstring is sampled, it is discarded.

**Impact:** Returned portfolio objective is worse than achievable from the very same shot record; "best_objective" is a misnomer.

**Fix:** Select `argmin_x evaluate(x)` over all sampled bitstrings (optionally restricted to feasible ones), not the mode. Keep `most_frequent` only as a tiebreaker/diagnostic.

### H3. "Quantum HHL" is a classical exact solve; result is independent of clock qubits/shots/seed
**Severity: High** (unjustified quantum claim; `method="quantum_hhl"`)
**File:** `src/qufin/risk/quantum_linear_systems.py:490-499` (`hhl_solve`), surfaced by `solve_linear_system(... method="quantum")` at `:546+`.

The circuit is built only "for resource accounting" (`_circuit = build_hhl_circuit(...)`) and then **discarded**; the solution is `np.linalg.solve(H, b_padded)` (or `lstsq`). The returned `LinearSystemResult.method` is `"quantum_hhl"` and the config exposes `n_clock_qubits`, `shots`, `seed` that have **zero effect** on the output. No phase estimation, no backend execution, no amplitude readout.

**Impact:** Users believe they are running HHL and comparing quantum vs classical; they are comparing classical-vs-classical. Any "quantum advantage"/"condition-number sensitivity of HHL" narrative built on this is unfounded.

**Fix:** Either implement HHL (QPE + controlled rotation + uncomputation + amplitude readout on the backend) and validate against the classical solve, or rename to `classical_reference` / clearly document that "quantum" mode returns the exact classical solution as a stand-in. Do not label the output `quantum_hhl`.

---

## Medium findings

### M1. Binary-encoding QUBO: `decode_weights` bit significance is reversed vs `build_matrix`
**Severity: Medium**
**File:** `src/qufin/portfolio/qubo.py:177-216` (`_build_binary`, `_wf`) vs `:238-249` (`decode_weights`).

`_build_binary` assigns the qubit at flat index `i*B+b` the weight `_wf(b)=2^b/max_level`, i.e. the **first** qubit in an asset's block (b=0) is the LSB. But `decode_weights` does `level = int(bitstring[i*B:(i+1)*B], 2)`, treating the **first** char as the MSB (2^(B-1)). So the objective `x@Q@x` was minimized under one bit↔weight map while the reported weights use the reverse map.

**Evidence:** `decode_weights('0100', B=2)` returns asset0 level = `int('01',2)=1 → 1/3`, but in `_build_binary` the set qubit (idx 1, b=1) carries weight `2/3`. Mismatch.

**Impact:** With `encoding="binary"`, the decoded portfolio weights do not correspond to the integer levels the QUBO actually optimized; reported weights are wrong (and `evaluate` is computed on a separate left-to-right convention, compounding with H1).

**Fix:** Make a single helper that maps qubit index → (asset, bit-significance) and use it in both `_build_binary` and `decode_weights`. Add a round-trip test: for random level vectors, encode→evaluate and decode agree.

### M2. Canonical QAE confidence interval clamps θ_high to π/2, mishandling a > 0.5
**Severity: Medium**
**File:** `src/qufin/options/amplitude_estimation/canonical.py:159-166`.

The point estimate `a = sin²(πy/M)` is fine for any `a` (verified: a_true=0.7→0.69, 0.9→0.90). But the CI uses `theta_high = min(np.pi/2, theta + delta_theta)` and `theta_low = max(0, ...)`, then `a_low = sin²(theta_low)`, `a_high = sin²(theta_high)`. For `θ` near π/2 (a near 1) the mapping `sin²` is non-monotone across π/2, and clamping at π/2 makes `a_high = 1` regardless of precision; for `θ` slightly above π/2 (i.e. `y > M/2`) the reported `a_low/a_high` ordering and width become meaningless. The docstring's stated assumption "a ≤ 0.5" is neither enforced nor used.

**Impact:** Misleading/incorrect confidence intervals for amplitudes > ~0.5; silent for the point estimate.

**Fix:** Map both `y±1` candidates through `sin²(π·/M)` and take `min/max` of the resulting amplitudes as the CI; do not clamp θ at π/2. Or fold `y → min(y, M−y)` consistently before forming the interval.

### M3. IQAE `_find_next_k` is an ad-hoc heuristic, not the Grinko et al. schedule; tolerance not guaranteed within `max_iterations`
**Severity: Medium**
**File:** `src/qufin/options/amplitude_estimation/iqae.py:185-203`, loop `:220-276`.

`_find_next_k` picks `k = pi/(4·Δθ) − 0.5` with `min(k_new,1000)` and a `k_last+1` floor, and the multi-branch resolver `_theta_intervals_from_measurement` keeps the **largest-overlap** branch. This differs from the published IQAE (which enforces that `[θ_low,θ_high]` stays inside a single half-period of `sin²((2k+1)θ)` before refining). The CI is honest (coverage tested below), but convergence to `epsilon_target` is not guaranteed inside `max_iterations`.

**Evidence:** Coverage and ε are respected when iterations suffice —
```
a_true=0.2, eps=0.005, 20 seeds: CI coverage 20/20, CI-halfwidth ≤ eps 20/20, max|err| 0.0006
```
but with too few iterations the estimate can sit well outside `epsilon_target` while the CI remains wide-but-honest —
```
a_true=0.2, eps=0.01, max_iter=20: estimate 0.176 (|err| 0.0236 ≈ 2.4× eps)
```
So results are *not wrong* (CI is valid), but the headline `epsilon_target` is not a guarantee.

**Impact:** Users may trust `estimate` to `epsilon_target` accuracy when only the (possibly wide) CI is trustworthy; oracle-call budgets differ from the paper's `O((1/ε)·log(1/α))`.

**Fix:** Implement the Grinko et al. `find_next_K` (largest `K` keeping the interval in one half-period) and return a `converged: bool`. Surface a warning when `max_iterations` is hit before `epsilon_target`.

### M4. `_richardson_coefficients` extrapolates with no noise model / no fit diagnostics (ZNE robustness)
**Severity: Medium**
**File:** `src/qufin/backends/error_mitigation.py:121-175`.

The Richardson weights themselves are mathematically correct (verified: for scale factors [1,3,5] weights [1.875,−1.25,0.375] sum to 1.0; exact on degree-≤2 polynomials in the noise scale). The concern is methodological: `zne_extrapolate` fits an exact interpolating polynomial of degree `len(scale_factors)−1` through *noisy* observable estimates with **no regression/averaging and no residual/uncertainty reporting**. Exact Richardson on noisy points amplifies shot noise badly (large alternating coefficients), and there is no guard for the common case where the observable is non-polynomial in noise (exponential decay), where Richardson is biased.

**Impact:** Mitigated value can be noisier than the unmitigated `scale=1` value, with no warning. Acceptable as one option, but not "production-grade" as the only extrapolator.

**Fix:** Offer linear/exponential least-squares extrapolation with covariance-weighted fits, return fit residual and an error bar, and document that Richardson assumes a low-degree polynomial noise model.

---

## Low findings

### L1. `cvar_from_samples` docstring contradicts implementation
**Severity: Low**
**File:** `src/qufin/risk/cvar.py:170-191`.

Docstring says "CVaR_alpha = E[X | X ≥ VaR_alpha] (for losses)" but the code takes the `ceil(alpha·n)` **lowest** sorted costs (`sorted_costs[:k]`) — correct for the *minimization/optimization* CVaR used by QAOA/VQE, but the prose describes the opposite tail. By contrast `portfolio_cvar` (`:194-219`) correctly takes the largest losses. Confusing for a risk API.

**Impact:** Documentation/expectation mismatch; a user calling `cvar_from_samples` on a loss array expecting tail-loss ES gets the best-case mean.

**Fix:** Clarify that this is the optimization-CVaR (best α-fraction for minimization); cross-reference `portfolio_cvar` for risk-management ES.

### L2. Black-Litterman returns parameter-uncertainty covariance, not posterior return covariance
**Severity: Low**
**File:** `src/qufin/portfolio/classical/black_litterman.py:84,91`.

`posterior_cov` is `(τΣ)^{-1} + PᵀΩ^{-1}P)^{-1}` (the posterior covariance *of the mean estimate*, often called `M`). Many BL implementations report the covariance of *returns* used for downstream optimization as `Σ + M`. Returning `M` alone (which is ~`τΣ`-scale, i.e. much smaller than `Σ`) will mislead anyone feeding `posterior_cov` into mean-variance.

**Impact:** Downstream optimizers fed `posterior_cov` would drastically understate risk. The `posterior_mu` is correct and standard.

**Fix:** Either return `Σ + M` as `posterior_cov`, or rename the field `mean_estimate_cov` and document that `posterior_cov` is not the return covariance.

### L3. Misc small issues
**Severity: Low**
- `risk/classical_var.py:107` parametric ES uses `norm.pdf(z)/alpha` with `z = ppf(1-alpha)` — verified correct (matches closed form to 4 dp), but the inline comment `ES = mu + sigma*phi(z)/(1-alpha)` is wrong (it's `/alpha`, not `/(1-alpha)`). Comment-only defect.
- `options/distributions.py:96` log-normal grid is uniform in **price** space over `[e^{μ−3σ√T}, e^{μ+3σ√T}]`; combined with only `2^n` points this truncates/biases the lognormal tail and contributes (with the σ-scaling) to QAE grid-discretization error. Documented limitation, but worth a `n_sigma`/grid-quality note since payoff expectations are tail-sensitive.
- `backends/mock.py:19-35` `MockBackend` returns fixed counts independent of the circuit; QAOA/VQE "optimization" with `MockBackend` does nothing (every objective evaluation is identical). Fine for plumbing tests, but any benchmark using MockBackend is meaningless — ensure docs/benchmarks use Aer.

---

## What was verified correct (no defects found)

| Area | Check | Result |
|------|-------|--------|
| `options/classical/black_scholes.py` | S=K=100,σ=.2,r=.05,T=1 | call **10.45058** (exp 10.4506), put **5.57353** (exp 5.5735) |
| | put-call parity C−P = S−Ke^{−rT} | **4.877058 == 4.877058** (exact) |
| | Greek signs/ranges | Δcall 0.637∈(0,1), Δput −0.363, Γ 0.0188>0, vega 37.5>0, θcall −6.41<0, ρcall 53.2>0 ✓ |
| | σ→0 ITM call (S=110) | 14.8771 == discounted intrinsic 14.8771 ✓ |
| `options/classical/binomial.py` | CRR→BS | n=2000 price 10.44958, err **1.0e-3** ✓; Δ/Γ converge to BS |
| `options/classical/monte_carlo.py` | seeded MC → BS | price 10.412 ± 0.033, BS 10.451 (within ~1.2 SE) ✓ |
| `options/heston.py` | xi→0, v0=θ=0.04 → BS(σ=0.2) | strong 10.462±0.023, weak 10.463±0.023, BS 10.451 ✓ |
| `risk/classical_var.py` | parametric VaR/ES vs closed form | VaR 0.03193 vs 0.03190; ES 0.04030 vs 0.04025 ✓ |
| | historical vs parametric; CVaR≥VaR | consistent; ES ≥ VaR in loss terms ✓ |
| `risk/cornish_fisher.py` | normal sample → z_cf≈z; left-skew → CF VaR > Gaussian | CF 6.70 vs Gauss 5.30 on skew −1.42 ✓ |
| `risk/credit/gaussian_copula.py` | Vasicek vs MC (500 obligors, pd .02, ρ .2) | Vasicek VaR 0.1286 vs MC 0.132; ES≥VaR ✓ |
| `portfolio/classical/mean_variance.py` | 2-asset min-var vs closed form | [0.7119,0.2881] == closed form ✓; max-Sharpe sums to 1 |
| `portfolio/classical/black_litterman.py` | no-views → posterior==prior | ✓ (π=[0.066,0.099]) |
| `portfolio/classical/hrp.py` | weights ≥0, sum 1, sensible ordering | ✓ |
| `risk/garch.py` | fit + 5-step forecast under arch/numpy-2 | converges, AIC/forecast produced ✓ |
| `risk/quantum_entropy.py` | von-Neumann via normalized eigvals, log base e, eff. rank | formulas correct on inspection |
| `backends/error_mitigation.py` | Richardson coeffs sum=1; exact on deg-≤2 | ✓ (ZNE *coefficients* correct; see M4 for methodology) |
| QAE estimators on a clean Ry oracle | canonical/IQAE/MLAE accuracy | a_true 0.05→0.9 all within ~0.01–0.02; IQAE 20/20 CI coverage |

Note: the QAE *estimators themselves* are sound; the failures (C1, C2) are entirely in the **oracle construction** (comparator endianness) and **state preparation** (`Initialize` vs `StatePreparation`).

---

## Reproduction commands (abridged)

```bash
# C2(a) Initialize crash
python -c "from qufin.options.amplitude_estimation.european_qae import *; from qufin.backends.qiskit_backend import QiskitAerBackend; \
from qufin.options.amplitude_estimation.canonical import *; \
p,r=build_european_estimation_problem(EuropeanQAESpec(n_qubits=3)); \
CanonicalAmplitudeEstimation(p,CanonicalQAEConfig(),QiskitAerBackend()).estimate()"
# -> CircuitError: 'Initialize is not unitary thus can not be inverted.'

# C1 / C2(b) endianness: statevector of oracle marks bit-reversed indices (see evidence blocks above)
# C3 PEC: inspect pec_mitigate — meas_circ is rerun unchanged every sample (line 679)
# H1 endianness: x-gate on qubit0 -> qiskit '001'; QUBO.evaluate/decode read '001' as asset2
# H3 HHL: hhl_solve returns np.linalg.solve(H,b); independent of n_clock_qubits/shots/seed
```
