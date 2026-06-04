# Review 03 — Testing, CI/CD & Reproducibility (`qufin`)

**Reviewer role:** Principal test/release engineer (skeptical audit)
**Repo:** `/home/user/qufin` · branch `claude/determined-fermi-y8m78` · HEAD `a5b185c`
**Date:** 2026-06-04
**Environment:** Python 3.11, pytest 9.0.3, coverage 7.14.1, hypothesis 6.155, numpy 2.4.6, qiskit 2.4.1. Installed editable `[dev]`. `fastapi/torch/pennylane/cirq/braket` NOT installed.

---

## Severity summary

| Severity | Count | Findings |
|---|---|---|
| **CRITICAL** | 2 | C1 (headline "2499 tests" is a fabricated static badge), C2 (advertised regression suite is empty — 0 tests) |
| **HIGH** | 5 | H1 (coverage omit inflates 91%→~79%), H2 (optional code never run in CI — only mocked), H3 (252 mock-only tests for omitted modules pad the count), H4 (loose/OR-of-weak quantum correctness tolerances), H5 (CI matrix omits 3.13/3.14 that pyproject claims to support) |
| **MEDIUM** | 7 | M1 (no test-count gate), M2 (GitHub Actions unpinned + PyPI OIDC), M3 (lint/typecheck minimal coverage), M4 (288 weak-only-assertion tests), M5 (codecov `fail_ci_if_error:false`), M6 (hypothesis no deadline/derandomize, DB not CI-cached), M7 (`-m "not slow"` means CI runs fewer tests than headline) |
| **LOW** | 5 | L1 (6 zero-assertion smoke tests), L2 (`.hypothesis/` gitignored yet committed), L3 (CLI MC has no seed flag), L4 (stress suite mislabeled — it's correctness/convergence, not load), L5 (mean 1.55 asserts/test) |

**Net:** the test *suite is real and largely passes*, RNG/determinism in source is genuinely good, but the **two headline marketing numbers are both dishonest** and CI never exercises ~13% of the codebase it ships.

---

## HONESTY VERDICT on the headline stats

### "2499 tests passing" — **DISHONEST (fabricated number).**

- It is a **hardcoded static badge**, not a generated figure:
  `README.md:22` → `![Tests](https://img.shields.io/badge/tests-2499%20passing-brightgreen)`; repeated `README.md:316` and `CHANGELOG.md:12,45`.
- Actual collection: **2507 tests** (`pytest --collect-only` → `2507 tests collected`). Confirmed by summing `--co -q` per-file counts (= 2507).
- In **this `[dev]` environment**, 43 skip (39 in `test_api_server.py` via `pytest.importorskip("fastapi")` at `tests/unit/test_api_server.py:40`; 4 hardware tests via `skipif` in `tests/integration/test_hardware_smoke.py:18`). So a full local run would report **≈2464 passed**, not 2499.
- In **CI** the marker filter `-m "not hardware and not slow"` (`ci.yml:20`) deselects 47 slow tests → **2460 collected** (`2460/2507 tests collected (47 deselected)`); CI installs only `[dev]` (no fastapi) so ~39 more skip → **≈2417 pass in CI**.
- **2499 matches none of {2507 collected, 2464 local-pass, 2460 CI-collected, 2417 CI-pass}.** It is a number nobody can reproduce. There is no CI assertion that pins it (M1), so it will silently rot.

### "91% coverage" — **MISLEADING (inflated ~12 points by the omit list).**

- `[tool.coverage.run].omit` (`pyproject.toml:152-166`) excludes **13 real modules** = **2148 statements = 13.0% of the package** (16,482 total statements; computed via `coverage.parser.PythonParser`):

  | module | stmts | module | stmts |
  |---|--:|---|--:|
  | api/server.py | 365 | backends/cirq_backend.py | 208 |
  | data/bloomberg.py | 358 | backends/braket_backend.py | 179 |
  | backends/dwave_backend.py | 403 | backends/pennylane_backend.py | 158 |
  | data/refinitiv.py | 116 | backends/cudaq_backend.py | 100 |
  | viz/dashboard.py | 98 | backends/ibm_runtime.py | 54 |
  | backends/ionq_backend.py | 51 | backends/quantinuum_backend.py | 47 |
  | _version.py | 11 | **TOTAL** | **2148** |

- These modules are **never run in CI** (their optional deps aren't installed — H2) and their only tests **mock the dependency** (H3), so under coverage they would measure ≈0%. Reconstructing:
  - headline denominator = 16482 − 2148 = **14334**; 91% → ≈13044 covered statements.
  - true coverage over the *whole* package = 13044 / 16482 ≈ **79.1%** (≈80% even if you generously assume 10% of omitted lines are hit).
- So the honest, all-module figure is **~79%, not 91%** — and the gate `fail_under=90` (`pyproject.toml:169`, also `--cov-fail-under=90` in `ci.yml:20`) is measured against the *padded* denominator, so it does not actually guarantee 90% of shipped code.

**Bottom line:** the suite exists and passes (no collection errors, representative slices green), determinism in source is well done, but **both advertised numbers are untrustworthy**: the test count is invented, and the coverage figure is propped up by omitting an eighth of the code that CI never executes.

---

## One-paragraph verdict

This is a competently-built test suite wearing dishonest marketing. The mechanics are mostly sound — 2507 real tests collect with zero errors, source-level RNG discipline is genuinely good (every stochastic algorithm defaults `seed=42` via `np.random.default_rng`, zero global `np.random.*` mutation), the `MockBackend` is seeded, and the "stress" suite contains legitimate convergence/parity checks. But the two numbers the owner advertises are both false: **"2499 tests" is a hand-typed badge that matches no actual run** (real: 2507 collected / ~2464 local-pass / ~2417 CI-pass), and **"91% coverage" is inflated to a true ~79%** by an `omit` list that hides 2,148 statements (13% of the code) — code that CI never even imports because its optional deps are uninstalled and its only tests mock the dependency. Layer on a CI matrix that skips the 3.13/3.14 it claims to support, an *empty* `tests/regression/` despite an advertised "paper-reproduction" marker, unpinned Actions feeding an OIDC PyPI publish, and ~290 tests whose strongest assertion is `> 0` / `is not None`, and the verdict is clear: **not trustworthy as-is, and nowhere near "trillion-dollar enterprise grade."** Fixable, but the headline claims must be corrected or the gaps closed before either number is repeated.

---

## CRITICAL findings

### C1 — "2499 tests passing" is a fabricated static badge
- **Where:** `README.md:22`, `README.md:316`, `CHANGELOG.md:12`, `CHANGELOG.md:45`
- **Evidence:**
  ```
  $ grep -nE "2499" README.md
  22:[![Tests](https://img.shields.io/badge/tests-2499%20passing-brightgreen)]()
  316:pytest                             # Full suite (2499 tests)
  $ pytest --collect-only | tail -1
  2507 tests collected in 1.48s
  $ pytest --collect-only -m "not slow and not hardware" | tail -1
  2460/2507 tests collected (47 deselected)
  ```
  No CI step asserts any test count.
- **Impact:** A core advertised metric is unverifiable and wrong by ~8–90 depending on how you count. Erodes trust in every other claim. For a "production/stable" lib this is a credibility failure.
- **Fix:** Delete the hardcoded badge or generate it from `pytest --co -q | tail -1` in CI. State the real number ("2507 collected; ~2417 run in CI"). Add a CI assertion (see M1).

### C2 — Advertised regression suite is empty (0 tests)
- **Where:** `tests/regression/` contains only an empty `__init__.py`. The `regression` marker is declared (`pyproject.toml:146`, "paper-reproduction regression tests") and the JOSS-style claims tout reproduced paper numbers.
- **Evidence:**
  ```
  $ wc -l tests/regression/*.py
  0 tests/regression/__init__.py
  $ grep -rn "mark.regression" tests --include="*.py" | wc -l
  0
  ```
- **Impact:** The library claims paper-reproduction regression coverage that **does not exist**. No test pins any published benchmark/paper value; nothing guards against numerical drift in the flagship algorithms.
- **Fix:** Either implement real `@pytest.mark.regression` tests that assert against published reference values with explicit tolerances, or remove the marker and stop advertising regression/paper-reproduction.

---

## HIGH findings

### H1 — `omit` list inflates coverage 91% → true ~79%
- **Where:** `pyproject.toml:152-166` (omit) + `:169` (`fail_under=90`) + `ci.yml:20` (`--cov-fail-under=90`).
- **Evidence:** 2148 / 16482 statements (13.0%) omitted (table above). With the demo run:
  ```
  $ coverage run -m pytest tests/unit/test_backends.py -q && coverage report --include="*/backends/*"
  ... omitted pennylane/cirq/braket/... backends DO NOT APPEAR in the report at all
  TOTAL 1358 1261 7%
  ```
  (The omitted backends are suppressed even when explicitly `--include`d.) Reconstruction → ~79.1% all-module.
- **Impact:** The gate `fail_under=90` does not protect 90% of shipped code; the badge overstates by ~12 pts.
- **Fix:** Stop omitting code you ship. Either (a) install the optional extras in a CI leg and measure them, or (b) report two numbers honestly ("91% on core; X% on optional integrations"), or (c) move truly-unmeasurable thin adapters behind `# pragma: no cover` per-line rather than wholesale module omits, and lower the headline to the real all-module figure.

### H2 — Optional code (api/ml/all backends) is NEVER executed in CI — only mocked
- **Where:** `ci.yml:18` installs only `pip install -e ".[dev]"`; `[api]`, `[ml]`, `[all]`, etc. are never installed in any CI job.
- **Evidence:** All omitted modules import via try/except guards but their functional bodies require absent deps:
  ```
  $ python -c "import qufin.backends.pennylane_backend"   # OK (guarded)
  $ python -c "import qufin.api.server"                   # OK (fastapi guarded)
  ```
  No CI leg installs fastapi/torch/pennylane/cirq/braket → the real `pennylane.device(...)`, `fastapi` routes, `torch` discriminator path (`src/qufin/ml/quantum_gan_finance.py:34`) are never run.
- **Impact:** ~13% of the codebase (every cloud backend, the REST API, the dashboard, the torch HQGAN path) ships with **zero real execution in CI**. A breaking change to any of these passes CI green.
- **Fix:** Add a CI matrix leg `pip install -e ".[all,api,ml]"` (or per-extra legs) that runs the corresponding tests *without* mocks, and includes those modules in coverage.

### H3 — 252 mock-only tests for omitted modules pad the count and mask zero real coverage
- **Where:** `tests/unit/test_pennylane_backend.py` (23), `test_cirq_backend.py` (23), `test_braket_backend.py` (30), `test_cudaq_backend.py` (18), `test_dwave_backend.py` (36), `test_ionq_backend.py` (15), `test_quantinuum_backend.py` (17), `test_bloomberg.py` (36), `test_refinitiv.py` (29), `test_dashboard.py` (25). **Total 252 tests.**
- **Evidence:** `tests/unit/test_pennylane_backend.py:1-3` docstring: *"All tests mock PennyLane so they pass WITHOUT pennylane installed."* They inject a fake module into `sys.modules` and assert against the mock:
  ```
  $ pytest tests/unit/test_pennylane_backend.py tests/unit/test_bloomberg.py tests/unit/test_dashboard.py
  84 passed
  $ grep -cE "patch\(|MagicMock|Mock\(" tests/unit/test_cirq_backend.py
  37
  ```
- **Impact:** Double whammy — these 252 tests inflate the "2499" count, yet the modules they "test" are *also* omitted from coverage, so the heavy mocking is never penalized. They verify that the adapter calls a mock the way the test set up the mock — near-tautological for catching real integration bugs.
- **Fix:** Run these against the real libraries in an optional-extras CI leg (H2) and include the modules in coverage; downgrade the mock-only variants to clearly-labeled "import contract" smoke tests.

### H4 — Quantum "correctness" tests use OR-of-weak / oversized tolerances
- **Where:** `tests/unit/test_vqe.py:145-149`, `:292`, `:349`.
- **Evidence:**
  ```python
  exact = exhaustive_solve(small_qubo)               # test_vqe.py:137
  # "Generous bound: VQE objective within factor of 3 or within 1.0"
  assert (result.best_objective <= exact.best_objective * 3.0
          or result.best_objective <= exact.best_objective + 1.0)   # :146-149
  ```
  Warm-start "non-degradation": `assert result_warm.best_objective <= result_cold.best_objective + 1.0` (`:292,:349`).
- **Impact:** The docstring claims results are "verifiable against exhaustive solver," but a `*3.0` factor **OR** a `+1.0` absolute slack passes for essentially any non-pathological output (objectives here are O(0.01–0.1)). These tests would not catch a regression that materially degrades VQE/warm-start quality. Same shallowness recurs in core risk tests, e.g. `tests/unit/test_classical_var.py:79` asserts only `result.var > 0` (no numerical reference).
- **Fix:** Replace OR-bounds with a single justified tolerance tied to shot noise (e.g. assert VQE within `exact * (1 + k/sqrt(shots))`), and add value-pinned assertions (e.g. monte_carlo_var vs. closed-form parametric VaR within rtol).

### H5 — CI matrix omits Python 3.13 / 3.14 that pyproject advertises as supported
- **Where:** `ci.yml:10` → `python: ["3.10", "3.11", "3.12"]`; `pyproject.toml:30-31` classifiers claim `Python :: 3.13` and `:: 3.14`.
- **Evidence:**
  ```
  $ grep -n "3.1" pyproject.toml | grep classifiers -A6   # lists 3.13, 3.14
  $ grep python: .github/workflows/ci.yml                  # 3.10, 3.11, 3.12 only
  ```
- **Impact:** Advertised platform support is untested. 3.14 especially (free-threading, numpy/qiskit ABI churn) is a real risk for a numeric lib.
- **Fix:** Either add 3.13/3.14 to the matrix or remove the classifiers. Don't claim support you don't test.

---

## MEDIUM findings

### M1 — No CI gate verifying the test count
- **Where:** `ci.yml` (absent). **Impact:** the "2499" claim (C1) can never be caught drifting; tests can silently vanish (e.g. a module-level import error skipping a whole file) without failing CI as long as the rest pass. **Fix:** add a step asserting `pytest --co -q | tail -1` ≥ expected, or fail if any file errors at collection.

### M2 — GitHub Actions unpinned; PyPI publish via OIDC with floating action
- **Where:** `ci.yml:13,14,21,89,90,94` (`@v4`, `@v5`, `codecov-action@v4`, `pypa/gh-action-pypi-publish@release/v1`); release job has `permissions: id-token: write` (`ci.yml:85-87`).
- **Evidence:** every `uses:` is a moving tag; none pinned to a commit SHA.
- **Impact:** Supply-chain exposure — a compromised/retagged action runs in a job that mints an OIDC token to publish to PyPI. This directly contradicts "enterprise grade."
- **Fix:** Pin all actions to full commit SHAs (`uses: pypa/gh-action-pypi-publish@<sha>`), enable Dependabot for the pins (already configured for `github-actions`, `dependabot.yml:31`).

### M3 — Lint and typecheck cover only a sliver
- **Where:** `ci.yml:33` lint runs on 3.12 only; `ci.yml:47-52` mypy checks only `backtesting/`, 3 risk files, `portfolio/classical/`. `pyproject.toml:128-138` strict typing applies to the same narrow set; the rest has `disallow_untyped_defs = false`.
- **Impact:** ~90% of source is effectively untype-checked; ruff results may differ across the 3.10–3.14 the lib claims. **Fix:** run mypy on `src/qufin` (allow gradual `ignore_errors` per-module if needed) and lint on the full matrix or at least the min+max version.

### M4 — 288 tests assert only weak/shallow properties
- **Where:** AST scan across `tests/**`. Worst files: `test_coverage_boost.py` (27), `test_auto_select.py` (15), `test_streaming.py` (12), `test_dashboard.py` (11), `test_widgets.py` (10).
- **Evidence:**
  ```
  Tests whose ONLY checks are weak (is-not-None / isinstance / bare truthiness / len>=0 / True),
  no raises/approx/np.testing: 288   (of 2540 test funcs)
  ```
  `tests/unit/test_coverage_boost.py:1-7` is explicitly a coverage-padding file ("Additional unit tests to boost coverage on under-tested modules"; inline comments like "7 uncovered lines", "15 uncovered – requires yfinance, mock it"). 121 tests, many shallow.
- **Impact:** ~11% of tests would pass against substantially broken implementations; coverage % is propped up by line-touching tests with no behavioral assertion.
- **Fix:** Strengthen assertions (compare to references/invariants), and stop writing tests whose stated purpose is "boost coverage."

### M5 — Coverage upload failures silently ignored
- **Where:** `ci.yml:25` `fail_ci_if_error: false`, and upload runs only on `ubuntu-latest && python==3.12` (`ci.yml:22`). **Impact:** broken coverage reporting never fails CI; the badge can go stale undetected. **Fix:** set `fail_ci_if_error: true` (the gate is already `--cov-fail-under`, so token issues should be visible).

### M6 — Hypothesis: no deadline override / not derandomized / DB not CI-cached
- **Where:** `tests/property/test_bs_properties.py`, `tests/property/test_portfolio_properties.py`; no `register_profile`/`load_profile` anywhere; `tests/conftest.py` sets no hypothesis profile.
- **Evidence:** `@settings(max_examples=…)` only; no `deadline=None`, no `derandomize=True`, no `@seed`. `.hypothesis/` is gitignored (`.gitignore`), so CI starts with an empty example DB each run.
- **Impact:** Default 200 ms deadline can flake the optimization/portfolio property tests on slow CI runners; without derandomize + persisted DB, a failure found on one run won't reproduce on the next (CI re-rolls examples), making flakes non-actionable. **Fix:** add a CI hypothesis profile with `deadline=None, derandomize=True` (or cache `.hypothesis/` in CI).

### M7 — CI runs fewer tests than the headline (47 slow excluded)
- **Where:** `ci.yml:20` `-m "not hardware and not slow"`; 47 slow-marked tests (`pytest --co -m slow` → `47/2507`). **Impact:** the most expensive convergence/integration checks (e.g. 7 in the stress suite, 7 in `test_vqe_aer.py`) never run in CI, so the green check is weaker than implied. **Fix:** run slow tests on a scheduled (nightly) workflow or on `main`.

---

## LOW findings

- **L1 — 6 zero-assertion smoke tests** (`tests/unit/test_explainability.py:271,278,283`; `test_api_server.py:574`; `test_plugins.py:173`; `test_bloomberg.py:167`). Bodies just call a function "# no crash". Acceptable as smoke tests but assert no correctness. Add at least a return-type/None check.
- **L2 — `.hypothesis/` is in `.gitignore` yet `.hypothesis/constants/*` are tracked** (`git ls-files .hypothesis` returns many files). Inconsistent; remove from the index.
- **L3 — CLI Monte-Carlo has no `--seed` flag** (`grep seed src/qufin/cli.py` → none), so CLI MC pricing is not reproducible from the command line even though the library functions are seeded.
- **L4 — "Stress" suite is mislabeled.** `tests/stress/test_stress_suite.py` (88 tests) is actually correctness/convergence (put-call parity, finite-diff Greeks, MC→BS convergence) — good content, wrong name; no load/concurrency/large-scale stress is performed.
- **L5 — Mean 1.55 asserts/test, median 1** across 2540 functions — low assertion density for a numeric library; many tests check a single property.

---

## What is genuinely good (no inflation here)

- **Source RNG discipline is excellent.** Every stochastic algorithm defaults `seed=42` and uses `np.random.default_rng(seed)`: QAOA (`src/qufin/portfolio/optimizers/qaoa.py:36,130`), Monte Carlo (`src/qufin/options/classical/monte_carlo.py:34,65,111,157`), qGAN (`src/qufin/ml/qgan.py:38,70`), `monte_carlo_var` (`src/qufin/risk/classical_var.py:125,146`). **Zero** bare global `np.random.*` mutation in `src/` (grep returned 0). `MockBackend` is seeded (`src/qufin/backends/mock.py:19-22`); `conftest.py` exposes a seeded `rng` fixture.
- **An explicit determinism test exists** (`tests/unit/test_classical_var.py:83 test_deterministic` asserts identical output for equal seeds).
- **No live-network dependency:** yfinance/FRED/crypto are mocked (`tests/unit/test_crypto.py:31`, `tests/unit/test_coverage_boost.py:1225,1260`); no test hits the network.
- **No collection errors**; representative slices pass (black_scholes+VaR+property+integration = 57 passed / 4 hardware-skipped).
- **Stress/convergence suite** has well-chosen invariants with seeded, sensibly-toleranced checks (e.g. `test_european_mc_converges_to_bs`: `european_mc(..., n_paths=500_000, seed=42)` asserted `abs(mc-bs) < 0.15`).
- **Release job depends on `[test, lint, typecheck]`** (`ci.yml:80`) and uses OIDC trusted publishing (good — modulo the unpinned action, M2).

---

## Priority fix list (highest leverage first)

1. **Correct or auto-generate the two headline numbers** (C1, H1) — stop advertising 2499/91%; publish the real ~2507-collected and ~79% all-module figures, or close the gaps.
2. **Add an optional-extras CI leg** (`[all,api,ml]`) that runs the backend/api/ml tests *without mocks* and measures them in coverage (H2, H3) — this single change also de-inflates coverage honestly.
3. **Implement the regression suite or drop the claim** (C2).
4. **Pin all GitHub Actions to SHAs** (M2) given the OIDC PyPI publish.
5. **Tighten the loose quantum/risk tolerances** (H4) and add value-pinned assertions to the ~290 weak/empty tests (M4, L1).
6. **Align the test matrix with advertised Python support** (H5) and add a test-count gate (M1).
