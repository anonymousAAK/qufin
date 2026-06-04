# qufin Documentation, Claims & DX Review

**Reviewer:** Technical Documentation Lead / DX Reviewer  
**Date:** 2026-06-04  
**Branch:** claude/determined-fermi-y8m78  
**Environment:** Python 3.11, numpy 2.4.6, qiskit 2.4.1 (qufin installed editable with [dev])  

---

## Severity-Count Summary

| Severity | Count |
|:---------|------:|
| Critical | 4 |
| High | 9 |
| Medium | 7 |
| Low | 4 |
| **Total** | **24** |

---

## README Quickstart: Pass / Fail Verdict

| Example | Status | First Error |
|:--------|:-------|:------------|
| Black-Scholes / IQAE (Option Pricing) | **FAIL** | `ImportError: cannot import name 'bs_price'` |
| QAOA Portfolio | **FAIL** | `ImportError: cannot import name 'make_benchmark'` + `ImportError: cannot import name 'build_qubo'` + `TypeError: QAOAOptimizer does not exist` |
| More Examples: GBM / Heston | **FAIL** | `ImportError: cannot import name 'HestonParams'` |
| More Examples: BacktestEngine | **FAIL** | Constructor signature wrong (`rebalance_freq`/`window` don't exist); `compute_metrics` not exported |
| More Examples: auto_select_backend | **PASS** | Runs correctly (circuit parameter required) |
| Benchmarks section: BenchmarkRunner | **FAIL** | `ImportError: cannot import name 'make_benchmark'`; wrong runner interface |

**Summary: 1 of 6 README Quickstart examples runs without error. The 5 failures are not edge cases — they are the headline examples on the front page.**

---

## Advertised vs. Actual Headline Numbers

| Claim | Source | Verified Value | Accurate? |
|:------|:-------|:--------------|:----------|
| 159 modules | README badge, JOSS paper | 159 `.py` files (includes 20 `__init__.py`); 139 non-init modules | Technically true (counts `__init__.py`), misleading |
| 14 subpackages | README badge, JOSS paper | 14 top-level dirs under `src/qufin/`; 19 Python packages total (including sub-subpackages) | True if "top-level only" |
| 2,499 tests | README badge, JOSS paper | `pytest --collect-only` reports **2,507** collected | Wrong (off by 8) |
| 11 backends | README badge, JOSS paper | 11 concrete `Backend` subclasses in source | Correct |
| 5 error mitigation strategies | README badge | Capabilities section lists 8 distinct strategies (ZNE, TREX, readout calibration, PEC, CDR, DD, M3, noise-aware optimizer); JOSS paper table says "5 + DD + M3" | Badge is wrong/inconsistent |
| 4 QAE variants | README badge, JOSS paper | 6 core AE algorithm modules (canonical, IQAE, MLAE, FQAE, MRQAE, QMC); 12 AE-related files total | Badge undercounts; README Capabilities section lists 9 |
| 91% coverage | README badge | `fail_under = 90` in pyproject.toml; coverage file present; no fresh run performed | Badge value unverified from fresh run in this env |
| Development Status :: 5 - Production/Stable | pyproject.toml classifier | Installed version is `0.1.dev1+gb5ef2090f.d20260509`; no git tags; CHANGELOG claims v1.1.0 | **Critical mismatch**: dev pre-release classified as Production/Stable |
| Version 1.1.0 | CHANGELOG, CITATION.cff | `qufin.__version__ == '0.1.dev1+gb5ef2090f.d20260509'`; no v1.1.0 git tag | **Critical**: CHANGELOG and CITATION.cff claim released v1.1.0 that has no tag |

---

## Verdict on "Honest" Positioning

qufin prominently markets itself as providing "honest benchmarks" and "production-grade engineering." This review found that all four README Quickstart examples (except `auto_select_backend`) fail on import with `ImportError` or `TypeError` before a single line of algorithmic code executes. The functions `bs_price`, `european_qae_price`, `make_benchmark`, `build_qubo`, `QAOAOptimizer`, `HestonParams`, and `compute_metrics` are all advertised in the README but do not exist under those names in the source. Six of ten tutorial notebooks contain broken imports. The JOSS paper — the peer-reviewed artifact — also contains two broken code blocks. The advertised test count, QAE variant count, and error mitigation count are all wrong. The version metadata presents a `0.1.dev` pre-release as `Development Status :: 5 - Production/Stable`. The project's "honest" positioning is directly undermined by its own front-page examples not running.

---

## Finding Details

### CRITICAL

---

#### C-01: README Quickstart Example 1 — `bs_price` and `european_qae_price` do not exist

**Claim (README.md:97–114):**
```python
from qufin.options.classical.black_scholes import bs_price
from qufin.options.amplitude_estimation.european_qae import european_qae_price
```

**Verification:** `python /tmp/test_quickstart1.py` → `ImportError: cannot import name 'bs_price'`  
**True value:** The module exports `call_price`, `put_price`, `price_and_greeks`, etc. `bs_price` is a method on the `EuropeanOption` class in `options/european.py`. `european_qae_price` is not exported from any module; the correct API is `build_european_estimation_problem(EuropeanQAESpec(...))` + `IterativeAmplitudeEstimation(...).estimate()`.  
**Fix:** Replace the README example with the real API. The docs `quickstart.md` already uses the correct OOP form (`opt.bs_price()`).

---

#### C-02: README Quickstart Example 2 — `make_benchmark`, `build_qubo`, `QAOAOptimizer` do not exist

**Claim (README.md:119–134):**
```python
from qufin.benchmarks.problems import make_benchmark
from qufin.portfolio.qubo import build_qubo
from qufin.portfolio.optimizers.qaoa import QAOAOptimizer
problem = make_benchmark(15)
Q = build_qubo(problem.mu, problem.sigma, risk_aversion=0.5, k=5)
optimizer = QAOAOptimizer(backend=QiskitAerBackend(shots=4096), p=2, mixer="xy_ring")
result = optimizer.solve(Q, n_assets=15, k=5)
result.selected / result.objective
```

**Verification:** `python /tmp/test_quickstart2.py` → `ImportError: cannot import name 'make_benchmark'`  
**True value:**
- `make_benchmark` → does not exist; use `portfolio_small()`, `portfolio_medium()`, `portfolio_large()`
- `build_qubo` → does not exist; the correct class is `PortfolioQUBO(mu=..., cov=..., gamma=..., cardinality=...)` followed by `.build_matrix()`
- `QAOAOptimizer` → does not exist; correct class is `QAOAPortfolio(qubo, config, backend)`
- `QiskitAerBackend(shots=4096)` → `TypeError: __init__() got an unexpected keyword argument 'shots'`; constructor takes `method` and `seed`; `shots` is passed to `.run()`
- `result.selected` / `result.objective` → `QAOAResult` has `best_bitstring` and `best_objective`

**Fix:** Rewrite example using real API. This is the flagship portfolio example.

---

#### C-03: `Development Status :: 5 - Production/Stable` with `0.1.dev` version and no git tags

**Claim (pyproject.toml:24):** `"Development Status :: 5 - Production/Stable"`  
**Claim (CHANGELOG.md:10):** `## [1.1.0] - 2026-05-22`  
**Claim (CITATION.cff:5):** `version: "1.1.0"`  
**Verification:** `python -c "import qufin; print(qufin.__version__)"` → `0.1.dev1+gb5ef2090f.d20260509`; `git tag` → no output (no tags)  
**True value:** The package has never been tagged. hatch-vcs falls back to `0.1.dev`. CHANGELOG and CITATION claim v1.1.0 was released 2026-05-22 but no corresponding git tag exists.  
**Fix:** Create the v1.1.0 git tag OR change classifier to `4 - Beta` and version to `1.1.0` in a `pyproject.toml` static field. Do not claim Production/Stable for untagged dev builds.

---

#### C-04: JOSS Paper Code Examples Are Broken

**Claim (papers/joss/paper.md, Block 0):**
```python
ideal = QiskitAerBackend(shots=4096)
noisy = NoisyAerBackend(profile=NoiseProfile.EAGLE_R3, shots=4096)
```
**Claim (papers/joss/paper.md, Block 1):**
```python
spec = EuropeanQAESpec(s=100, k=105, ..., option_type="call")
```

**Verification:**
- `QiskitAerBackend(shots=4096)` → `TypeError: __init__() got an unexpected keyword argument 'shots'`
- `NoiseProfile.EAGLE_R3` → `AttributeError`: `NoiseProfile` is a `@dataclass`, not an Enum; the constant is the module-level `IBM_EAGLE_R3`
- `EuropeanQAESpec(s=100, ..., option_type="call")` → `TypeError: __init__() got an unexpected keyword argument 's'`; correct fields are `s0` and `is_call=True`  
**Fix:** The JOSS paper — the artifact for peer review — contains broken code in two of two code blocks. Fix before submission.

---

### HIGH

---

#### H-01: README "More Examples" — Three of Four Code Blocks Broken

**Claim (README.md:143–172):**
1. `HestonParams` — `ImportError: cannot import name 'HestonParams' from qufin.data.synthetic`; `heston_paths` takes individual keyword args, not a params dataclass.
2. `BacktestEngine(rebalance_freq="monthly", window=252)` — constructor has no `rebalance_freq` or `window` parameter; real signature is `BacktestEngine(returns, dates=None, train_window=252, test_window=21, ...)`.
3. `engine.run(prices_df, strategy_fn)` — takes `returns` (not `prices_df`) and does not accept a prices DataFrame; real signature is `run(strategy, strategy_name="unnamed")`.
4. `compute_metrics(portfolio_values)` — `ImportError: cannot import name 'compute_metrics' from qufin.backtesting.metrics`; real function is `performance_summary(returns, ...)`.

**Severity:** High — these block every "More Examples" user immediately.

---

#### H-02: Benchmarks Section — `make_benchmark` and Wrong `BenchmarkRunner` Interface

**Claim (README.md:262–270):**
```python
runner.run(make_benchmark(15), algorithms=["qaoa", "vqe", "mean_variance"])
runner.summary(results)
```
**True value:** `make_benchmark` does not exist. `BenchmarkRunner.run` does not exist; the real methods are `run_problem(problem)` and `run_all(problems)`. There is no `summary` method; results are `list[BenchmarkRow]`. Solvers must be registered manually via `runner.register(SolverEntry(...))`.

---

#### H-03: Test Count Badge Is Wrong

**Claim (README.md:23):** `[![Tests](https://img.shields.io/badge/tests-2499%20passing-brightgreen)]`  
**Verification:** `pytest --collect-only 2>&1 | grep collected` → `2507 tests collected in 1.43s`  
**True value:** 2,507. The badge also says "passing" — no test run was performed here to confirm pass rate.  
**Fix:** Update badge to 2507.

---

#### H-04: Error Mitigation Count Inconsistency

**Claim (README.md:28):** `5 error mitigation strategies`  
**Claim (JOSS paper table):** `5 + DD + M3`  
**Claim (README Capabilities):** Lists 8 items: Readout calibration, TREX, ZNE, Dynamical Decoupling, PEC, CDR, M3, Noise-aware variational  
**True value:** There are 7–8 distinct mitigation mechanisms implemented (ZNE, TREX, readout calibration, PEC, CDR, DD with 3 sequences, M3, noise-aware optimizer). The badge number "5" is inconsistent with the capabilities section and the JOSS table.

---

#### H-05: QAE Variant Count Undercounts Implementations

**Claim (README.md:28, JOSS paper):** `4 QAE variants`  
**True value:** Six distinct QAE algorithm modules exist: `canonical.py`, `iqae.py`, `mlae.py`, `fqae.py`, `mrqae.py`, `qmc.py`. The README Capabilities section itself lists nine (`Canonical QAE, IQAE, MLAE, FQAE, Path-Dependent QAE, American QAE, QMC (Montanaro), QSP, Asian QAE`). The badge contradicts the capabilities section and undercounts reality.

---

#### H-06: Tutorial Notebooks — 9 Broken Imports in 8 Notebooks

Static analysis of all tutorial notebooks revealed the following broken imports (verified by execution):

| Notebook | Cell | Import | Error |
|:---------|:-----|:-------|:------|
| `05_bs_to_qae.ipynb` | cell3 | `from qufin.options.classical.black_scholes import bs_price, bs_greeks` | `ImportError: bs_price` |
| `06_amplitude_estimation.ipynb` | cell3 | `from qufin.options.classical.black_scholes import bs_price` | `ImportError: bs_price` |
| `07_risk_var_cvar.ipynb` | cell15 | `from qufin.risk.stress import stress_test` | `ImportError: stress_test` (real fn: `stress_test_suite`) |
| `08_noise_mitigation.ipynb` | cell16 | `from qufin.backends.dynamical_decoupling import apply_dd` | `ImportError: apply_dd` (real fn: `insert_dd_sequences`) |
| `01_classical_portfolio.ipynb` | cell13 | `from qufin.portfolio.classical.hrp import hrp_optimize` | `ImportError: hrp_optimize` (real fn: `hrp`) |
| `09_real_hardware.ipynb` | cell11 | `from qufin.backends.transpiler import finance_transpile` | `ImportError: finance_transpile` (real class: `FinanceTranspiler`) |
| `10_quantum_advantage.ipynb` | cell9 | `from qufin.options.classical.black_scholes import bs_price` | `ImportError: bs_price` |
| `10_quantum_advantage.ipynb` | cell13 | `from qufin.benchmarks.problems import make_benchmark` | `ImportError: make_benchmark` |
| `04_vqe_portfolio.ipynb` | cell11 | `from qufin.portfolio.optimizers.warm_start import warm_start_portfolio` | `ImportError: warm_start_portfolio` (real fn: `warm_start_qaoa`) |

These are static checks; notebooks are not executed in CI (no `nbmake` or similar).

---

#### H-07: "30-50% CNOT Reduction" Claim Is Unsubstantiated Marketing

**Claim (README.md:275):** `Transpiler: QUBO-aware ZZ optimization, 30-50% CNOT reduction`  
**Claim (transpiler.py:370):** docstring repeats "Targets 30-50% CNOT reduction for portfolio QAOA circuits"  
**Verification:** `benchmarks/` directory has no results demonstrating this range. `results.json` contains only `{"command": "benchmark", "data": {..., "status": "completed"}}` with no circuit-level metrics. No paper or benchmark table documents this figure.  
**Fix:** Either add a benchmarking script that measures and documents actual CNOT reduction on the standard problem sizes, or soften the claim to "targets significant CNOT reduction via Qiskit optimization_level=3".

---

#### H-08: `subpackages` Count Ambiguous

**Claim (README.md:28):** `14 subpackages`  
**True value:** 14 top-level sub-directories under `src/qufin/`. However there are 19 Python packages total (including `options/classical`, `options/amplitude_estimation`, `portfolio/classical`, `portfolio/optimizers`, `risk/credit`). The README architecture tree explicitly shows sub-subpackages yet the badge counts only top-level. Not wrong, but misleading for readers comparing package depth.

---

#### H-09: Author/Maintainer Inconsistency Between CITATION.cff and pyproject.toml

**Claim (pyproject.toml:13–14):**
```
authors = [{ name = "Adarsh Keshri" }]
maintainers = [{ name = "anonymousAAK" }]
```
**Claim (CITATION.cff:10–14):**
```
authors:
  - given-names: Adarsh
    family-names: Keshri
```
**Claim (README.md BibTeX block:339):**
```bibtex
author  = {Adarsh Keshri},
```
CITATION.cff and README BibTeX use "Adarsh Keshri". pyproject.toml splits into `author = Adarsh Keshri` vs `maintainer = anonymousAAK` (a GitHub handle). This creates conflicting attribution metadata for packaging and citation. For enterprise procurement (particularly legal/IP review), author identity must be unambiguous.

---

### MEDIUM

---

#### M-01: `QiskitAerBackend(shots=4096)` Pattern Appears Throughout Docs

The README Quickstart (line 107), the JOSS paper (Block 0), and the docs user guide `backends.md` all use `QiskitAerBackend(shots=4096)` which raises `TypeError`. The correct pattern is `QiskitAerBackend(seed=42)` and passing `shots` to `.run(circuit, shots=4096)`. Pervasive across multiple documents.

---

#### M-02: "Module Count" Counts `__init__.py` Files

**Claim (README.md:28, JOSS paper):** `159 modules`  
**True value:** 159 total `.py` files = 139 non-`__init__` modules + 20 `__init__.py` files. By conventional Python usage "module" means a `.py` file, so 159 is not wrong, but presenting `__init__.py` packaging shims as "modules" inflates the metric. A reader expecting 159 useful API surfaces will find 139.

---

#### M-03: IBM 156-Qubit Claim is Backend-Specific, Not General

**Claim (README.md:247):** `IBMRuntimeBackend — IBM QPU (156 qubits)`  
**True value:** IBM's Heron r2 systems (e.g., `ibm_kingston`, `ibm_marrakesh`) have 156 qubits as of 2025–2026. However, the default backend name in `ibm_runtime.py` is `"ibm_brisbane"` (Eagle r3, 127 qubits). The table description is accurate for Heron r2 but not the default; a user who reads the table, instantiates `IBMRuntimeBackend()` with defaults, and counts qubits will get 127, not 156.

---

#### M-04: Inconsistent Error Mitigation Hierarchy in README vs JOSS Paper

README Capabilities groups DD under "Level 2" alongside ZNE and counts M3 separately; JOSS paper table says "5 + DD + M3" suggesting DD and M3 are not in the "5". The badge says simply "5". A reader cannot determine which 5 are the 5. The Capabilities text already provides the most accurate picture; the badge should be updated or the table footnoted.

---

#### M-05: SECURITY.md Version Table Claims "0.1.x (latest)" but CHANGELOG Says v1.1.0

**Claim (SECURITY.md):**
```
| Version | Supported |
| 0.1.x (latest) | Yes |
```
**Claim (CHANGELOG.md):** latest release is `[1.1.0]`  
These are contradictory. Either the security policy is stale or the CHANGELOG release claims are premature (since no v1.1.0 git tag exists).

---

#### M-06: `QAOAConfig` and `QAOAResult` Are Undocumented Public Classes

`QAOAConfig` and `QAOAResult` in `portfolio/optimizers/qaoa.py` have no docstrings, despite being the primary interface for the QAOA solver. The `QiskitAerBackend.run()` and `statevector()` methods are also undocumented. These are among the most-used public APIs.

---

#### M-07: mkdocs Builds Without Errors But Material for MkDocs Warns of Breaking Changes

`mkdocs build --strict` exits 0 (no strict errors), but prints a prominent warning that MkDocs 2.0 will break all plugins, all theme overrides, and currently has no migration path and is unlicensed for production. This is a dependency risk for the documentation pipeline.

---

### LOW

---

#### L-01: `IBM_HERON_R2` vs `IBMHeron_R2` vs `NoiseProfile.EAGLE_R3` — Naming Inconsistency

Module-level constants are `IBM_EAGLE_R3`, `IBM_HERON_R2` (uppercase snake). The JOSS paper references `NoiseProfile.EAGLE_R3` (attribute access on class). The user guide `backends.md` uses `IBM_HERON_R2`. These are inconsistent across documents and the JOSS paper usage is broken.

---

#### L-02: `QAOAResult.best_bitstring` vs README `result.selected`

README example (line 131) prints `result.selected` and `result.objective`. Actual `QAOAResult` fields are `best_bitstring` and `best_objective`. This is the QAOA example result interface, not just an internal name.

---

#### L-03: `HestonParams` Mentioned in README "More Examples" Does Not Exist

The README example implies a `HestonParams` dataclass exists for wrapping Heston model parameters. In reality, `heston_paths()` takes individual kwargs (`v0`, `kappa`, `theta`, `xi`, `rho`). The docs `quickstart.md` does not make this mistake.

---

#### L-04: CONTRIBUTING.md References `pre-commit install` But No `.pre-commit-config.yaml` Was Checked

`CONTRIBUTING.md` instructs `pre-commit install`. This is a standard workflow, but worth confirming the hooks file exists. (Not a blocking issue but should be validated in CI.)

---

## Supporting Data

### Module Counts (actual, by `find`)

```
Total .py files in src/qufin/:  159
  - __init__.py files:           20
  - non-init modules:           139
Top-level sub-dirs (subpkgs):   14
Total Python packages:          19 (including sub-subpackages)
```

### Test Count

```
pytest --collect-only: 2507 tests collected
README badge: 2499
Delta: +8
```

### Backend Count

Concrete `Backend` subclasses: `MockBackend`, `QiskitAerBackend`, `NoisyAerBackend`, `IBMRuntimeBackend`, `PennyLaneBackend`, `CirqBackend`, `BraketBackend`, `CudaQBackend`, `DWaveBackend`, `IonQBackend`, `QuantinuumBackend` = **11**. Claim of 11 is correct.

### QAE Variant Count

Core AE algorithm files: `canonical.py`, `iqae.py`, `mlae.py`, `fqae.py`, `mrqae.py`, `qmc.py` = **6**  
Option-specific wrappers: `european_qae.py`, `asian_qae.py`, `path_dependent_qae.py`, `american_qae.py`, `multi_asset_qae.py`, `qsp_pricing.py` = **6**  
Badge says 4; README capabilities section lists 9; actual core AE algorithms = 6.

### Docstring Coverage

```
Public functions/classes:  1156
Documented:               1055
Docstring coverage:        91.3%
```

Notable undocumented public APIs: `QAOAConfig`, `QAOAResult`, `QiskitAerBackend.run`, `QiskitAerBackend.statevector`, `QiskitAerBackend.backend_id`.

### Version Consistency

| Source | Version |
|:-------|:--------|
| `qufin.__version__` | `0.1.dev1+gb5ef2090f.d20260509` |
| `importlib.metadata` installed | `0.1.dev31+ga5b185c2f` |
| CHANGELOG latest entry | `1.1.0 - 2026-05-22` |
| CITATION.cff | `1.1.0` |
| git tags | *(none)* |
| pyproject.toml classifier | `Development Status :: 5 - Production/Stable` |

### mkdocs Build

`mkdocs build --strict` exits 0. No missing nav pages, no broken internal links detected. All 31 expected `.md` files are present. Material for MkDocs prints a deprecation warning about MkDocs 2.0 but does not fail the build.
