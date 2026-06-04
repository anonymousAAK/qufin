# qufin — Architecture / API / Typing Review

**Reviewer scope:** package & API design, backend abstraction, typing & mypy coverage, error handling, config singleton, plugin system, CLI, result-object consistency, import hygiene.
**Branch:** `claude/determined-fermi-y8m78` · **Size:** 159 modules, 47,893 LOC · **Env:** Py 3.11, numpy 2.4.6, pandas 3.0.3, qiskit 2.4.1, mypy 2.1.0, ruff 0.15 (fastapi/torch/pennylane/cirq/braket NOT installed).

---

## Severity counts

| Severity | Count |
|----------|-------|
| Critical | 4 |
| High     | 7 |
| Medium   | 7 |
| Low      | 4 |
| **Total**| **22** |

## Type-checking coverage (quantified)

| Metric | Value |
|--------|-------|
| Total source files | 159 |
| Total source LOC | 47,893 |
| Files type-checked **in CI** | 14 |
| LOC type-checked **in CI** | 2,232 |
| **% of LOC type-checked in CI** | **4.7 %** |
| **% of files type-checked in CI** | **8.8 %** |
| mypy errors when run on the **whole** package (with the project's own lenient config) | **86 errors across 30 files** |
| mypy errors caught by CI | **0** (the 14 CI files are clean; every one of the 86 errors is in an unchecked file) |
| `py.typed` shipped to downstream consumers? | **Yes** (`src/qufin/py.typed`) |

## Verdict

`qufin` presents an enterprise veneer — `Development Status :: Production/Stable`, a backend ABC, a plugin system, a settings layer, a REST/CLI surface, 90% coverage gate — but the load-bearing guarantees are hollow. Only **4.7 % of the code is type-checked in CI**, yet a `py.typed` marker tells every downstream consumer the package is fully typed; running mypy on the rest surfaces **86 errors**, several of which are **guaranteed runtime crashes** in the API layer (`PortfolioQUBO.solve_qaoa` does not exist, `quantum_var(confidence=…)` is the wrong signature, the Celery backtest task calls `engine.run()` with no `strategy`) — and those exact files are *also* excluded from test coverage, so nothing catches them. The backend "abstraction" is leaky (`run`/`statevector` typed as `Any`, and annealing backends satisfy the ABC only by raising `NotImplementedError`, an LSP violation). There is no exception hierarchy (101 bare `ValueError`s, one lone `PluginError`), the settings "singleton getter" mutates global state as a side effect of being *read* with overrides, the plugin validator contradicts its own contract, and `import qufin` eagerly drags in scipy+sklearn+pandas (~1.4 s) even to import a one-line mock. This is not yet trillion-dollar-enterprise grade; the gap is structural, not cosmetic.

---

## Critical

### C1. `py.typed` shipped, but 95 % of the package is unchecked and contains 86 type errors
**Severity:** Critical · **File:** `pyproject.toml:120-138`, `.github/workflows/ci.yml:38-52`, `src/qufin/py.typed`

**Evidence:**
```
$ # CI typecheck job runs mypy only on backtesting/ + 3 risk files + portfolio/classical/
$ python  # measured coverage
Total src .py files: 159, total LOC: 47893
CI-typechecked files: 14, LOC: 2232
Percent of LOC type-checked in CI: 4.7%
$ mypy src/qufin/        # whole package, project's own config
Found 86 errors in 30 files (checked 159 source files)
$ ls -la src/qufin/py.typed     # present -> advertises "fully typed" to consumers
-rw-r--r-- 1 root root 0 ... src/qufin/py.typed
```
Error categories (whole package):
```
20 [assignment]  15 [no-any-return]  12 [arg-type]  10 [attr-defined]
 7 [union-attr]   7 [index]           3 [no-untyped-def] 3 [dict-item]
 2 [var-annotated] 2 [misc]  2 [call-arg]  1 [return-value]  1 [abstract]
```
**Impact:** Shipping `py.typed` is a public promise that downstream `mypy` runs against qufin's *own* annotations are sound. They are not — consumers inherit 86 errors and wrong signatures. The `[[tool.mypy.overrides]]` block sets `disallow_untyped_defs=true` for a handful of modules and CI checks an even smaller subset, so 95 % of the surface (all quantum algorithms, all backends, the API, ML) has zero enforced typing. SemVer "stability" is unmeasured for 95 % of the code.

**Recommendation:** Either (a) run `mypy src/qufin/` over the entire package in CI (start with the current lenient config, ratchet `disallow_untyped_defs` on per module, drive errors to 0), or (b) **remove `py.typed`** until the whole tree type-checks. Do not ship a typed marker over an untyped, error-laden codebase. Treat C2 as the proof that this is not academic.

### C2. Multiple guaranteed runtime crashes in the API layer (uncaught because it is neither typed-checked nor covered)
**Severity:** Critical · **File:** `src/qufin/api/server.py:382`, `src/qufin/api/server.py:568`, `src/qufin/api/jobs.py:274`

**Evidence:**
```
$ mypy src/qufin/
src/qufin/api/server.py:382: error: "PortfolioQUBO" has no attribute "solve_qaoa"  [attr-defined]
src/qufin/api/server.py:568: error: Unexpected keyword argument "confidence" for "quantum_var"  [call-arg]
src/qufin/api/jobs.py:274:  error: Missing positional argument "strategy" in call to "run" of "BacktestEngine"  [call-arg]
$ python -c "from qufin.portfolio.qubo import PortfolioQUBO; print(hasattr(PortfolioQUBO,'solve_qaoa'))"
False
$ python -c "from qufin.risk.quantum_var import quantum_var; import inspect; print(inspect.signature(quantum_var))"
(loss_distribution: 'DistributionSpec', backend: 'Backend', config: 'QuantumVaRConfig | None' = None) -> 'QuantumVaRResult'
$ python -c "from qufin.backtesting.engine import BacktestEngine; import inspect; print(inspect.signature(BacktestEngine.run))"
(self, strategy: 'StrategyFn', strategy_name: 'str' = 'unnamed') -> 'BacktestResult'
```
**Impact:** The REST optimize endpoint calls `qubo.solve_qaoa(...)` — that method does not exist (`AttributeError` 500). The REST risk endpoint calls `quantum_var(confidence=...)` — wrong kwarg (`TypeError` 500). The Celery `run_backtest` task calls `engine.run()` with no `strategy` (`TypeError`). These are not edge cases; they are the primary code paths of the public API. They are invisible today only because `api/server.py` is in `[tool.coverage] omit` (`pyproject.toml:164`), `api/jobs.py` requires celery (not installed → never imported in tests), and neither is in the CI mypy set.

**Recommendation:** Fix the three call sites, add API integration tests with fastapi/celery in the test extra, remove `api/server.py` from coverage `omit`, and add `qufin.api.*` to the CI mypy targets. This is the single most important item for "serious enterprise adoption."

### C3. `get_settings(**overrides)` silently mutates the global singleton — shared-state corruption, not thread-safe
**Severity:** Critical · **File:** `src/qufin/utils/settings.py:35-40`

**Evidence:**
```python
def get_settings(**overrides: Any) -> Settings:
    global _settings
    if _settings is None or overrides:
        _settings = Settings(**overrides)   # ANY call with overrides REPLACES the global
    return _settings
```
```
$ python  # reproduce the side effect
a = get_settings()                 # default_shots = 1024
b = get_settings(default_shots=99) # override -> rebinds global _settings
c = get_settings()                 # NO override...
c.default_shots == 99              # ...but returns the mutated singleton (not 1024)
```
**Impact:** A "getter" with the contract of returning configuration in fact performs a global write whenever any caller passes an override. One component calling `get_settings(seed=…)` permanently changes `default_backend`, `seed`, `cache_dir`, etc. for *every other component* and every subsequent no-arg call. There is no lock (`PLW0603` is globally ignored to allow this), so concurrent access (the very API/Celery workers in C2) races on `_settings`. It also makes tests order-dependent and non-isolated. The override path also *drops* all previously-set fields not in `overrides` (re-constructs from scratch).

**Recommendation:** Split the API: `get_settings()` returns the read-only global (constructed once, lazily, under a `threading.Lock`); provide an explicit `configure(**overrides)` that returns a *new* `Settings` (a `model_copy(update=…)`), and an explicit `reset_settings()` for tests. Never mutate global state from a read accessor. Consider context-local settings (`contextvars`) for request-scoped overrides in the API.

### C4. `default_backend` default value does not match any registered backend name
**Severity:** Critical · **File:** `src/qufin/utils/settings.py:20` vs `src/qufin/backends/auto_select.py:174-227, 325`

**Evidence:**
```
$ grep default_backend src/qufin/utils/settings.py
default_backend: str = Field(default="qiskit-aer", ...)     # HYPHEN
$ grep -n qiskit_aer src/qufin/backends/auto_select.py
available["qiskit_aer"] = True                              # UNDERSCORE everywhere
for name in ("qiskit_aer", "mock"): ...
if name == "qiskit_aer": ...
```
**Impact:** The configured default backend id `"qiskit-aer"` is never a key in `get_available_backends()` (`"qiskit_aer"`) nor accepted by `auto_select_backend(preference=...)` nor `BackendRegistry.get`. Any code that wires `settings.default_backend` into backend resolution gets a silent miss / `KeyError`. Pure naming-convention drift between the config layer and the backend layer.

**Recommendation:** Pick one canonical spelling (recommend `"qiskit_aer"` to match the registry and the package module names) and add a unit test asserting `settings.default_backend in get_available_backends()`. Add a `Literal[...]`/validator on the field so an invalid id fails fast.

---

## High

### H1. Backend abstraction is leaky — circuit typed as `Any`; algorithms cannot be backend-agnostic
**Severity:** High · **File:** `src/qufin/backends/base.py:51-56`

**Evidence:**
```python
@abstractmethod
def run(self, circuit: Any, shots: int = 1024) -> CircuitResult: ...
@abstractmethod
def statevector(self, circuit: Any) -> NDArray[np.complex128]: ...
```
`analyze_circuit`/`auto_select_backend` also take `circuit: Any` and reach into qiskit-specific attributes (`circuit.data`, `circuit.find_bit`, `instruction.operation.name`) — `auto_select.py:67-87`.
**Impact:** The "backend-agnostic" claim (docstring `base.py:41`) is false: the circuit type is `Any`, and the only structural analysis assumes a `qiskit.QuantumCircuit`. There is no abstract `Circuit` type, so the type system gives downstream zero help and a Cirq/Braket circuit passed to `analyze_circuit` silently yields an empty analysis (everything falls through `hasattr`). The abstraction leaks qiskit through an `Any` hole.

**Recommendation:** Introduce a `Circuit` Protocol (or a bound `TypeVar`) capturing the minimal surface the framework needs, and parametrize `Backend`/`analyze_circuit` on it. At minimum, type `circuit` as `qiskit.QuantumCircuit` honestly if that is the real contract, and document that non-qiskit backends transpile from it.

### H2. LSP violation: annealing/sampler backends conform to the ABC only by raising `NotImplementedError`
**Severity:** High · **File:** `src/qufin/backends/dwave_backend.py:249-260` (and ionq/quantinuum follow the same pattern)

**Evidence:**
```python
class DWaveBackend(Backend):
    def run(self, circuit, shots=1024) -> CircuitResult:
        raise NotImplementedError("DWaveBackend is an annealing backend; use solve_qubo()...")
    def statevector(self, circuit) -> NDArray[np.complex128]:
        raise NotImplementedError("...statevector is not supported.")
$ python  # all 11 backends "conform"
QiskitAerBackend  subclass=True  unimplemented_abstract=[]
DWaveBackend      subclass=True  unimplemented_abstract=[]   # but run() always raises
... (all 11 report unimplemented_abstract=[])
```
**Impact:** `Backend` declares `run`/`statevector` as the universal contract, but a `DWaveBackend` (a `Backend` per the type system) crashes on both. Any generic code — including `auto_select_backend`, which *returns* a `Backend` — that does `backend.run(circuit)` will explode at runtime for annealers. The ABC conflates two unrelated execution models (gate-circuit vs. QUBO-sampling) under one interface.

**Recommendation:** Split the hierarchy: a `GateBackend(Backend)` with `run`/`statevector`, and a separate `SamplerBackend`/`AnnealerBackend` with `solve_qubo`. `auto_select_backend` should be typed to return the model the caller asked for. Do not satisfy an ABC with stubs that raise.

### H3. No exception hierarchy — 101 `ValueError` / 27 `RuntimeError`, only one custom exception in the whole library
**Severity:** High · **File:** package-wide; `src/qufin/plugins.py:480` is the *only* custom exception

**Evidence:**
```
$ grep -roE "raise [A-Za-z_]+" src/qufin | sort | uniq -c | sort -rn
101 raise ValueError   28 raise ImportError   27 raise RuntimeError
  7 raise PluginError    6 raise NotImplementedError   5 raise HTTPException
  3 raise KeyError       2 raise ConnectionError       1 raise TypeError
$ grep -rn "class .*Error" src/qufin --include=*.py | grep -iv test
src/qufin/plugins.py:480:class PluginError(Exception)      # the ONLY one
$ find src/qufin -name exceptions.py -o -name errors.py    # none
```
**Impact:** Enterprise consumers cannot write `except qufin.QufinError` to distinguish *library-raised* conditions from genuine bugs (`KeyError`/`ValueError` from numpy, dict access, etc.). User-input validation errors (bad `alpha`, non-PSD covariance) are indistinguishable from internal invariants. `PluginError` doesn't even subclass a shared base. This blocks robust error handling, retries, and observability in production.

**Recommendation:** Add `qufin/exceptions.py` with a root `QufinError(Exception)` and a small tree (`ConfigurationError`, `ValidationError`, `BackendError`, `BackendUnavailableError`, `PluginError`, `ConvergenceError`). Make user-input failures subclass `ValidationError` and external/dependency failures subclass a `BackendError`/`DataSourceError`. Re-parent `PluginError`. Export from top-level.

### H4. `import qufin` eagerly loads the entire scientific stack (~1.4 s); importing a one-line mock pays the full cost
**Severity:** High · **File:** `src/qufin/__init__.py:10-22` (+ every subpackage `__init__`)

**Evidence:**
```
$ python -c "import time,sys; t=time.perf_counter(); import qufin; print((time.perf_counter()-t)*1000)"
1422.2  # ms
$ python -c "import qufin, sys; print('scipy',  'scipy'  in sys.modules)"   # True
                                  # pandas True, sklearn True, scipy True
$ python -c "import qufin.backends.mock, sys; print('qufin.risk' in sys.modules, 'sklearn' in sys.modules)"
True True       # importing the trivial mock ran qufin/__init__ -> all 11 subpkgs -> scipy+sklearn
$ python -X importtime ... | sort  # top self-time culprits
51133us scipy.stats._stats_py   49918us scipy.stats._continuous_distns
37039us pydantic_settings.sources.providers.azure   32961us scipy.ndimage
22472us pyarrow.compute   ...
```
(matplotlib and qiskit are *not* eager — verified `False` — so the lazy patterns work for those; the problem is the core stack.)
**Impact:** `qufin/__init__.py` imports all 11 subpackages, and each subpackage `__init__` eagerly imports its leaf modules (`risk/__init__.py` = 11 imports, `ml/__init__.py` = 8), which transitively pull scipy.stats/ndimage/special/optimize, sklearn, pandas, pyarrow. A CLI `--version`, a Lambda cold start, or a unit test that only needs `MockBackend` pays 1.4 s and ~hundreds of MB RSS. `pydantic_settings` even imports its Azure provider at import time. This is a real DX/serverless tax and amplifies cold-start cost for the very API in C2.

**Recommendation:** Make the top-level package lazy via PEP 562 `__getattr__` (import subpackages on first attribute access). Within subpackage `__init__`s, prefer lazy `__getattr__` over eager leaf imports. Keep heavy deps (scipy.stats, sklearn) behind function-local imports in the modules that need them. Target sub-200 ms `import qufin`.

### H5. Plugin backend validation contradicts its documented contract (duck-types instead of checking the ABC)
**Severity:** High · **File:** `src/qufin/plugins.py:169-183` vs docstring `plugins.py:14`

**Evidence:**
```python
# docstring: "The backend class must inherit from qufin.backends.base.Backend."
def _validate_backend(cls: Any, name: str) -> None:
    required_attrs = ("backend_id", "run")     # only 2 attrs, by hasattr
    for attr in required_attrs:
        if not hasattr(cls, attr):
            raise PluginError(...)
```
**Impact:** (1) The validator never checks `issubclass(cls, Backend)`, so the documented inheritance requirement is unenforced — any object with `backend_id`/`run` attributes passes, including a non-`Backend` that lacks `statevector`/`is_simulator`, breaking the abstraction the registry relies on. (2) `statevector` is not in `required_attrs`, so a "validated" plugin can be missing a core ABC method. (3) `ep.load()` (`plugins.py:161`) executes arbitrary third-party code with no signature/version guard beyond the two-attr check. (4) `register_strategy` (`plugins.py:193`) does zero validation of the function signature.

**Recommendation:** In `_validate_backend`, assert `inspect.isclass(cls) and issubclass(cls, Backend)` (the entry-point contract) and instantiate-or-`abstractmethods` check; raise `PluginError` otherwise. Validate strategy callables against an expected `Protocol`/signature. Document the trust model for `ep.load()`.

### H6. Inconsistent public-export discipline — public factory functions omitted from `__all__`; `from … import *` will not export them
**Severity:** High · **File:** `src/qufin/backends/__init__.py:15-24` (and `data`, `options`)

**Evidence:**
```
$ python  # public, documented factories NOT listed in __all__
backends  extra_public_nonmodule = ['get_braket_backend','get_cirq_backend','get_cudaq_backend',
                                    'get_dwave_backend','get_ibm_backend','get_ionq_backend', ...]
data      extra_public_nonmodule = ['get_bloomberg_source','get_fred_provider','get_refinitiv_source']
options   extra_public_nonmodule = ['get_amplitude_estimation','get_distribution_loaders']
```
`backends.__all__` lists 8 names but defines 9 `get_*_backend` factories that are the documented entry points for optional backends — none are in `__all__`.
**Impact:** `__all__` is the SemVer-relevant public contract. Names absent from it are (a) excluded by `from qufin.backends import *`, (b) often skipped by API-doc generators and stub checks, and (c) ambiguously "public-ish," so it is unclear whether they're covered by stability guarantees. This is exactly the surface enterprise users pin against.

**Recommendation:** Decide the contract: if `get_*_backend`/`get_*_source` are public (they are documented and the only way to reach optional backends), add them to `__all__`; if private, prefix with `_`. Add a CI test that asserts `__all__` ⊇ all non-underscore module-level callables (or an explicit allow-list).

### H7. Broken public export — `qufin.api.__all__` lists `JobQueue`, which the module does not provide
**Severity:** High · **File:** `src/qufin/api/__init__.py:13-18`

**Evidence:**
```
$ python -c "import qufin.api as a; print(a.__all__, hasattr(a,'JobQueue'))"
['JobQueue', 'JobStatus', 'create_app'] False
$ python -c "from qufin.api import JobQueue"
ImportError: cannot import name 'JobQueue' from 'qufin.api'
$ python -c "from qufin.api import *"      # would raise AttributeError on 'JobQueue'
```
The module imports only `JobStatus` and exposes `JobQueue` exclusively via the `get_job_queue()` factory; `JobQueue` itself is never bound at package level.
**Impact:** A name in `__all__` that does not resolve means `from qufin.api import JobQueue` and `from qufin.api import *` both raise `ImportError`/`AttributeError`. Documented, advertised public symbol is simply broken. (No subpackage other than `api` has this defect — verified across all 13.)

**Recommendation:** Either add `from qufin.api.jobs import JobQueue` to `api/__init__.py` (note: that import requires celery, so guard it or keep the factory and **remove `"JobQueue"` from `__all__`**). Add a CI test: for every subpackage, `assert all(hasattr(m, n) for n in m.__all__)`.

---

## Medium

### M1. CLI subcommands are stubs — they never invoke any qufin algorithm
**Severity:** Medium · **File:** `src/qufin/cli.py:136-286`

**Evidence:** `handle_optimize`/`handle_price`/`handle_risk`/`handle_benchmark` build a dict of strings (`"status": "completed"`, a `"description"` f-string) and return it. No pricing, no optimization, no VaR is ever computed. E.g. `handle_price` returns `{"solver":"analytical","description":"Black-Scholes european pricing: S=100, K=105"}` — no price. The `try/except Exception` blocks wrap pure dict construction that cannot raise.
**Impact:** The packaged `qufin` console script (`[project.scripts] qufin = "qufin.cli:main"`) is a façade for a Production/Stable library. `qufin price ...` produces a description, not a number. Users will reasonably believe the CLI computes results. Misleading for adoption and for the "production-grade" claim.

**Recommendation:** Either wire the handlers to the real algorithm entry points (and add tests asserting numeric output), or clearly label the CLI as a demo/skeleton in `--help` and docs until implemented. Remove dead `try/except` around non-failing code.

### M2. `auto_select` / availability probing knows only 6 of 11 backends — D-Wave, IonQ, Quantinuum, IBM-runtime unreachable via selection
**Severity:** Medium · **File:** `src/qufin/backends/auto_select.py:174-227, 307-356`

**Evidence:** `get_available_backends()` probes `mock, qiskit_aer, cudaq, pennylane, cirq, braket` only. `_try_create_backend` has branches for the same 6 (plus mock). But `backends/__init__.py` defines factories for `ibm, pennylane, cirq, braket, cudaq, dwave, ionq, quantinuum` (8) and `BackendRegistry` is generic. So `dwave/ionq/quantinuum/ibm_runtime` are never auto-selectable and never reported by the availability map.
**Impact:** The registry/selection layer and the factory layer disagree on the backend set. `auto_select_backend(preference="dwave")` silently falls through to qiskit/mock. The advertised 11-backend matrix is half-wired into the discovery path.

**Recommendation:** Drive availability and `_try_create_backend` from a single registry/table (name → import path → probe), so adding a backend updates discovery, selection, and factories at once. Add a test asserting the factory set == the probe set.

### M3. Broad `except Exception` swallowing (53 sites, 5 silently `pass`), including in backend selection
**Severity:** Medium · **File:** `src/qufin/backends/auto_select.py:353-354`, `dwave_backend.py:543-548`, +51 more

**Evidence:**
```
$ grep -rn "except Exception" src/qufin --include=*.py | wc -l   -> 53
$ grep -rn -A1 "except Exception" src/qufin | grep -c "pass"      -> 5
# auto_select._try_create_backend:
    except Exception:
        return None        # any error (OOM, bad credentials, bug) -> "backend unavailable"
```
**Impact:** `_try_create_backend` converts *every* failure (a bug in the backend, a credentials error, an OOM) into "not available," so users get a silent fallback to `mock`/`qiskit_aer` and never learn their requested backend is misconfigured. The 5 silent `pass` sites discard diagnostics entirely. This is the opposite of enterprise observability.

**Recommendation:** Catch `ImportError` narrowly for availability; let real errors propagate or wrap them in `BackendError` (see H3) and `logger.warning` with the original exception. Never blanket-swallow to `None`/`pass`.

### M4. 367 `Any` annotations and 50 `# type: ignore` — typing is largely escape-hatched even where present
**Severity:** Medium · **File:** package-wide (`backends/base.py`, `auto_select.py`, `plugins.py`, `dwave_backend.py`, …)

**Evidence:**
```
$ grep -rn ": Any\b|-> Any\b|\[Any\]" src/qufin --include=*.py | wc -l    -> 367
$ grep -rn "type: ignore" src/qufin --include=*.py | wc -l                 -> 50
$ mypy src/qufin/ | grep -c no-any-return                                  -> 15  (warn_return_any)
```
**Impact:** Even in the modules that *are* annotated, `Any` is pervasive (circuit objects, samplers, plugin classes, kwargs), so type checking provides little real safety. 15 `no-any-return` show `Any` leaking through ostensibly-typed function boundaries despite `warn_return_any=true`. Combined with C1, the practical type guarantee is near zero.

**Recommendation:** Replace `Any` on public boundaries with Protocols/precise types (circuit, sampler, backend class). Track an `Any`-budget and ratchet it down. Audit each `# type: ignore` for a specific error code rather than blanket ignores.

### M5. `global` mutable registries with no thread-safety (`plugins`, `BackendRegistry`)
**Severity:** Medium · **File:** `src/qufin/plugins.py:190, 365`, `src/qufin/backends/auto_select.py:113-166`

**Evidence:** `_strategy_registry`, `_data_source_registry` are module-level mutable dicts mutated by `register_*`/`clear_*`; `BackendRegistry` caches `_instances` in a plain dict. No locks anywhere.
**Impact:** In the multi-worker API/Celery deployment (C2), concurrent `register_strategy`/`get` calls race; `BackendRegistry.get` caches one shared mutable backend instance across threads. Combined with C3 (settings) this is a systemic shared-mutable-state problem for the concurrency model the project ships an API for.

**Recommendation:** Guard registries with a lock or document single-threaded-init-only semantics; make `BackendRegistry` instance caching opt-in / thread-safe. Provide per-request isolation in the API.

### M6. 79 bespoke `*Result` dataclasses with no common base or serialization contract; result-object style split (dataclass vs pydantic)
**Severity:** Medium · **File:** package-wide (`backends/base.py`, `dwave_backend.py`, `risk/*`, …); pydantic confined to `api/server.py`

**Evidence:**
```
$ grep -roE "class [A-Za-z0-9_]*Result" src/qufin | wc -l   -> 79 result classes
$ grep -rln "@dataclass" src/qufin | wc -l                  -> 113 files
$ grep -rln "BaseModel)" src/qufin                          -> api/server.py only (13 models)
```
**Impact:** 79 result types are all hand-rolled dataclasses with no shared base, no uniform `to_dict()`/serialization, no versioning. Some have `to_dict`, some don't (the API at `jobs.py:275` defensively checks `hasattr(result,"to_dict")`). Pydantic is used only at the HTTP boundary, so every result must be re-serialized ad hoc. There is no consistent contract for persisting/transmitting results — a problem for an enterprise audit/replay story.

**Recommendation:** Define a `QufinResult` base (or a serialization protocol with `to_dict`/`from_dict`/`schema_version`) that all result dataclasses implement; consider a single `asdict`-based mixin. Standardize on dataclass for compute results + pydantic for I/O, and bridge them in one place.

### M7. `from __future__ import annotations` leaks `annotations` into the public namespace of the top package
**Severity:** Medium · **File:** `src/qufin/__init__.py:3`

**Evidence:**
```
$ python -c "import qufin; print([n for n in dir(qufin) if not n.startswith('_') and n not in qufin.__all__])"
['annotations', 'utils']
```
**Impact:** `qufin.annotations` resolves to the `__future__` feature object and `qufin.utils` is importable but undocumented/unexported. Minor, but it pollutes the public surface and tab-completion, and `utils` being reachable-but-not-in-`__all__` repeats the H6 ambiguity at the top level (is `qufin.utils.get_settings` public?).

**Recommendation:** These are cosmetic given `__all__` gates `import *`, but consider `del`-ing leaked names or relying on a lint test that the public surface == `__all__`. Decide whether `qufin.utils`/`get_settings` is public and, if so, export it deliberately.

---

## Low

### L1. `Bitstring`/`Shots` `NewType`s defined in `_typing.py` are essentially unused; type aliases are thin
**Severity:** Low · **File:** `src/qufin/_typing.py:13-14`
**Evidence:** `Bitstring = NewType("Bitstring", str)`, `Shots = NewType("Shots", int)` are declared but the codebase uses raw `str`/`int` (`shots: int` in `Backend.run`, `base.py:51`). **Impact:** The typing module advertises domain types that aren't threaded through the API, so they provide no safety. **Recommendation:** Either adopt them consistently (e.g. `shots: Shots`) or drop them.

### L2. `_typing.py` is private (`_`-prefixed) but holds the canonical domain types
**Severity:** Low · **File:** `src/qufin/_typing.py`
**Evidence:** Module is `_typing` (private), yet `AssetReturns`/`CovMatrix`/`Weights` are the natural public vocabulary downstream code would annotate against. **Impact:** Consumers can't import shared aliases without reaching into a private module. **Recommendation:** Expose a public `qufin.typing` (or re-export the aliases from the top package) if they're meant to be used by consumers.

### L3. `get_settings`/`__init__` do not expose `cli`, `plugins`, `api`, `utils` at top level — discoverability gap
**Severity:** Low · **File:** `src/qufin/__init__.py:10-22`
**Evidence:** Top-level `__all__` exports the 11 science subpackages but not `api`, `cli`, `plugins`, `utils`, `backends`-factories. **Impact:** Public-but-peripheral surfaces (plugin registration, settings) require deep imports; uneven with how `backends`/`risk` are surfaced. **Recommendation:** Decide and document which of these are public; export deliberately or mark internal.

### L4. `Settings` lacks validation on `log_level` / `default_shots` upper bound and `cache_dir` writability
**Severity:** Low · **File:** `src/qufin/utils/settings.py:20-28`
**Evidence:** `log_level: str` accepts any string (no `Literal`/validator); `default_shots` has `ge=1` but no upper bound; `cache_dir` is not checked for writability. **Impact:** Misconfiguration (`QUFIN_LOG_LEVEL=verbose`) surfaces late. **Recommendation:** Use `Literal["DEBUG","INFO","WARNING","ERROR","CRITICAL"]` for `log_level` and validate `default_backend` (see C4).

---

## Appendix — commands used

```
python -c "import time,sys; t=time.perf_counter(); import qufin; ..."        # import cost + eager deps
python -X importtime -c "import qufin" 2>log; sort ...                         # slowest imports
python  # measured CI typecheck LOC/file coverage (glob backtesting + risk + portfolio.classical)
mypy src/qufin/                                                                # 86 errors / 30 files
mypy src/qufin/options/amplitude_estimation/  src/qufin/portfolio/optimizers/qaoa.py  \
     src/qufin/risk/quantum_var.py  src/qufin/api/  src/qufin/ml/                 # per-area runs
grep -roE "raise [A-Za-z_]+" src/qufin | sort | uniq -c                        # exception inventory
grep -rn "except Exception" src/qufin --include=*.py | wc -l                   # 53 broad catches
ruff check src/qufin/ ; ruff check src/qufin/ --select B006,RUF012             # lint clean; B006 clean; RUF012=4 (constants)
python  # __all__ correctness per subpackage; api.JobQueue broken; backend ABC conformance (11/11)
python  # get_settings(**overrides) global-mutation reproduction
```
