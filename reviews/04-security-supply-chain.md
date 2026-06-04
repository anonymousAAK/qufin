# qufin — Security & Supply-Chain Review

**Scope:** secrets/credentials, REST API + Celery/Redis, dangerous primitives (repo-wide), Docker, Kubernetes/Helm, CI/CD, dependency supply chain.
**Branch:** `claude/determined-fermi-y8m78`  •  **Commit:** `a5b185c`  •  **Date:** 2026-06-04
**Tooling:** `bandit 1.9.4`, `pip-audit` (installed in env), manual code review of every in-scope file.

---

## Severity counts

| Severity | Count |
|---|---|
| Critical | 2 |
| High | 5 |
| Medium | 6 |
| Low | 5 |
| **Total** | **18** |

## Tooling headline

- **bandit `-ll` (Medium/High):** *No issues.* Full scan (all severities) over 37,045 LOC: 0 High, 0 Medium, 24 Low (23 `B101 assert_used`, several `B110 try/except/pass`, `B404/B603/B607` for a hardcoded `git` subprocess in `benchmarks/manifest.py`, and one `B105` false positive at `data/streaming.py:331` — an empty-string `"secret": ""` literal, not a credential). **None are real vulnerabilities.** bandit's clean bill is genuine for the Python source: there is **no** `eval`, `exec`, `pickle.load(s)`, `marshal`, `yaml.load`, `os.system`, `shell=True`, `__import__`, `joblib.load`, `torch.load`, `np.load(allow_pickle=True)`, `tempfile.mktemp`, or `requests(verify=False)` anywhere in `src/` (verified by grep). The single network egress call (`data/crypto.py`) has a timeout and no key in the URL.
- **pip-audit:** **23 known vulnerabilities across 7 packages** — but these are all in the **ambient environment** (`cryptography 41.0.7`, `idna 3.11`, `pip 24.0`, `pyjwt 2.7.0`, `setuptools 68.1.2`, `urllib3 2.6.3`, `wheel 0.42.0`), **not** in qufin's declared dependency tree. None of these packages are direct qufin deps. The real supply-chain risk is **structural** (no lockfile, no hashes, unbounded `>=` floors), not these specific CVEs — see SC-1.

## Verdict on production readiness

**Not production-ready as shipped.** The library code itself is unusually clean — strong input validation via pydantic on every REST endpoint, JSON-only Celery serialization (no insecure deserialization), and zero dangerous primitives. **However, the deployment posture is the problem.** The shipped container and Helm chart launch the API with **authentication disabled by default** (`create_app()` with `api_keys=None`), behind an **unauthenticated Redis broker** that doubles as a remote task-injection vector, with **no Kubernetes `securityContext`** (containers can run privileged/writable-root), **API tokens placed in a ConfigMap instead of a Secret**, and a supply chain with **no lockfile/hashes** and **un-pinned base images and GitHub Actions**. For a "trillion-dollar enterprise grade" target, the application logic is a B+ but the operational/supply-chain layer is failing. The two Critical items (auth-off-by-default + unauthenticated broker) are directly exploitable for unauthorized compute/data access and likely RCE on workers.

---

# Findings

## CRITICAL

### C-1 — REST API ships with authentication disabled by default
**Severity: Critical**
**Files:** `src/qufin/api/server.py:628-696`, `Dockerfile:57`, `k8s/templates/deployment.yaml:23-34`

**Evidence.** `create_app()` accepts `api_keys: list[str] | None = None`; the auth dependency short-circuits when keys are unset:
```python
# server.py:684-696
async def verify_api_key(request, api_key=Security(api_key_header)) -> str | None:
    if _api_keys is None:
        return None            # <-- auth fully bypassed
    if api_key is None or api_key not in _api_keys:
        raise HTTPException(401, ...)
```
The shipped container invokes the factory with **no arguments**, so `api_keys` is `None`:
```dockerfile
# Dockerfile:57
CMD ["gunicorn", "qufin.api.server:create_app()", ...]
```
The Helm deployment exposes `/health`, `/v1/optimize`, `/v1/price`, `/v1/risk`, `/v1/jobs*` with the same factory and **no way to pass keys** (no `API_KEYS` env wiring in `configmap.yaml` or `server.py`). The module-level fallback `app = create_app()` (server.py:868) is also key-less.

**Attack scenario.** Anyone who can reach the Service/Ingress can submit unlimited optimization/pricing/risk jobs and enumerate/read every other tenant's job results via `GET /v1/jobs` (which lists *all* jobs with their result payloads, server.py:841-861) and `GET /v1/jobs/{job_id}`. No tenant isolation exists. In a multi-tenant or internet-exposed deployment this is unauthorized data access + free compute (DoS amplification via the QAE/QAOA paths).

**Remediation.**
- Make authentication fail-closed: refuse to start (or require an explicit `QUFIN_ALLOW_NO_AUTH=1`) when no keys are configured in a non-debug environment.
- Wire `api_keys` from a secret-backed env var (e.g. read `QUFIN_API_KEYS` inside `create_app`), surface it in the Helm chart as a `Secret`, and pass it in the gunicorn `CMD` via a factory wrapper.
- Compare keys with `hmac.compare_digest` (current `in _api_keys` set-membership is not constant-time).
- Scope `GET /v1/jobs` to the authenticated principal; never return all jobs globally.

---

### C-2 — Unauthenticated Redis broker = remote Celery task injection / worker RCE
**Severity: Critical**
**Files:** `docker-compose.yml:43-54`, `k8s/templates/service.yaml:21-75`, `k8s/templates/configmap.yaml:13-21`, `src/qufin/api/jobs.py:162-219`

**Evidence.** Redis runs with **no password** and is the Celery broker + result backend. All broker URLs are credential-free:
```yaml
# configmap.yaml:14-16 / docker-compose.yml:33-35
REDIS_URL:            "redis://{{ .Release.Name }}-redis:6379/0"
CELERY_BROKER_URL:    "redis://{{ .Release.Name }}-redis:6379/1"
CELERY_RESULT_BACKEND:"redis://{{ .Release.Name }}-redis:6379/1"
```
docker-compose additionally **publishes 6379 to the host** (`ports: ["6379:6379"]`). The k8s Redis (`service.yaml:44-46`) is `redis:7-alpine` with **no `--requirepass`, no securityContext, no NetworkPolicy** — reachable by any pod in the cluster.

**Attack scenario.** An attacker with network reach to Redis (any co-located pod, or any host on the docker-compose network) can `LPUSH` crafted messages onto the Celery queue (`qufin.interactive`/`qufin.batch`). Celery workers blindly execute queued task bodies. Two compounding factors:
1. **Worker compromise / RCE.** Celery's broker is a trust boundary; an attacker who controls broker messages controls task routing and arguments. Even with JSON content (see mitigations below), known Celery message-forgery / argument-abuse techniques against an open broker can drive workers into unintended code paths, and the result backend can be poisoned. With a writable root filesystem (see H-3) the blast radius is full worker takeover.
2. **Data exfiltration / poisoning.** Results in db `/1` are readable/writable by the attacker — they can read computed risk/pricing outputs and forge results returned to API clients.

**Partial mitigation already present (good):** `create_celery_app` pins `task_serializer="json"`, `accept_content=["json"]`, `result_serializer="json"` (jobs.py:202-206) — this blocks classic *pickle* deserialization RCE on the payload. The cache layer is also JSON-only (`cache.py` uses `json.dumps/loads`, never pickle). This is why this is "task injection + likely worker RCE" rather than "guaranteed one-shot pickle RCE."

**Remediation.**
- Require Redis AUTH: set `--requirepass`, put the password in a `Secret`, and embed credentials in the broker URL (`redis://:<pw>@host:6379/1`). Prefer TLS (`rediss://`).
- Remove the host port mapping in docker-compose (`6379:6379`) — workers reach Redis over the internal network.
- Add a `NetworkPolicy` so only the API and worker pods may talk to Redis on 6379.
- Keep `accept_content=["json"]` (never re-enable pickle). Consider Celery message signing.

---

## HIGH

### H-1 — API tokens/keys stored in a ConfigMap, not a Secret
**Severity: High**
**Files:** `k8s/templates/configmap.yaml:22-27`, `k8s/values.yaml:54-61`

**Evidence.** `QUFIN_FRED_API_KEY` and `IBM_QUANTUM_TOKEN` are injected through a **ConfigMap**:
```yaml
# configmap.yaml:22-27
{{- if .Values.config.fredApiKey }}
QUFIN_FRED_API_KEY: {{ .Values.config.fredApiKey | quote }}
{{- end }}
{{- if .Values.config.ibmQuantumToken }}
IBM_QUANTUM_TOKEN: {{ .Values.config.ibmQuantumToken | quote }}
{{- end }}
```
`values.yaml:59-61` even comments "Set via secrets in production" — but the template hard-codes them into a ConfigMap regardless of how they are supplied.

**Impact.** ConfigMaps are **not encrypted at rest** by default in etcd and are exposed to any principal with `get/list configmap` RBAC (a much broader set than `get secret`), to `kubectl describe`, and to GitOps/CI diffs. An IBM Quantum token grants paid quantum-hardware compute under the owner's account; a leaked FRED key is lower impact but still a credential.

**Remediation.** Move both into a `kind: Secret` (or external secret operator / Vault), mount via `secretKeyRef` in `envFrom`/`env`, enable etcd encryption-at-rest, and never render token values into ConfigMap templates.

---

### H-2 — Kubernetes workloads have no `securityContext` (root, writable FS, full caps)
**Severity: High**
**Files:** `k8s/templates/deployment.yaml`, `k8s/templates/worker.yaml`, `k8s/templates/service.yaml:44-55` (Redis)

**Evidence.** None of the three workloads (api, worker, redis) set a pod or container `securityContext`. There is no `runAsNonRoot`, `runAsUser`, `readOnlyRootFilesystem`, `allowPrivilegeEscalation: false`, `seccompProfile`, or `capabilities: drop: [ALL]`. The Redis pod (`service.yaml:44-55`) also runs as the image default (root in `redis:7-alpine` before it drops to the redis user — but unconstrained at the k8s level).

**Impact.** Although the qufin Dockerfile sets `USER qufin`, the cluster does not *enforce* non-root, so a different image tag, an overridden command, or a future Dockerfile regression silently runs as root. With a writable root FS, an attacker who lands code execution on a worker (see C-2) can persist, modify binaries, and escalate. This fails CIS Kubernetes Benchmark and most enterprise admission policies (PSA `restricted`).

**Remediation.** Add to every pod/container:
```yaml
securityContext:
  runAsNonRoot: true
  runAsUser: 1000
  readOnlyRootFilesystem: true
  allowPrivilegeEscalation: false
  capabilities: { drop: ["ALL"] }
  seccompProfile: { type: RuntimeDefault }
```
Provide an `emptyDir` for any writable paths. Enforce via Pod Security Admission (`restricted`) on the namespace.

---

### H-3 — Container & deployment images are not pinned by digest
**Severity: High**
**Files:** `Dockerfile:10,25`, `docker-compose.yml:44`, `k8s/values.yaml:5-8`, `k8s/templates/service.yaml:46`

**Evidence.**
- `Dockerfile:10/25` — `FROM python:3.12-slim` (mutable tag, no `@sha256:`).
- `docker-compose.yml:44` / `service.yaml:46` — `redis:7-alpine` (mutable tag).
- `values.yaml:6-8` — `tag: "1.1.0"`, `pullPolicy: IfNotPresent` (a re-pushed `1.1.0` won't be re-pulled; node caches drift).

**Impact.** Mutable base tags break reproducibility and open a supply-chain substitution window: a compromised or silently-updated `python:3.12-slim`/`redis:7-alpine` is pulled into builds and clusters with no detection. `IfNotPresent` on a mutable app tag yields inconsistent versions across nodes.

**Remediation.** Pin all images by digest (`python:3.12-slim@sha256:...`, `redis:7-alpine@sha256:...`, `qufin@sha256:...`). Use immutable release tags + `pullPolicy: IfNotPresent`, or digests with `Always`. Add image scanning (Trivy/Grype) to CI and sign images (cosign).

---

### H-4 — GitHub Actions pinned to floating tags; CI workflow lacks least-privilege `permissions`
**Severity: High**
**Files:** `.github/workflows/ci.yml:1-94`, `.github/workflows/docs.yml`

**Evidence.**
- Third-party/first-party actions are pinned to **mutable major tags**, not commit SHAs: `actions/checkout@v4`, `actions/setup-python@v5`, `codecov/codecov-action@v4`, `pypa/gh-action-pypi-publish@release/v1`, plus the docs workflow's `upload-pages-artifact@v3` / `deploy-pages@v4`.
- `ci.yml` has **no top-level `permissions:` block** → the workflow inherits the repository/org default `GITHUB_TOKEN` scope (often read-write). Only the `release` job sets explicit `permissions`.

**Impact.** A floating action tag can be force-moved to malicious code (cf. the `tj-actions/changed-files` 2025 compromise) and would execute in CI with whatever `GITHUB_TOKEN` scope is inherited — potentially writing to the repo or releases. `gh-action-pypi-publish@release/v1` is especially sensitive: it runs in the trusted-publishing (`id-token: write`) `release` job.

**Mitigations present (good):** the `release` job is correctly gated on `needs: [test, lint, typecheck]` and triggers only on `refs/tags/v*` with a scoped `permissions: {id-token: write, contents: read}`; there is **no `pull_request_target`** anywhere (no untrusted-PR code execution with secrets).

**Remediation.** Pin every action to a full commit SHA (Dependabot's `github-actions` ecosystem, already enabled, will bump them). Add `permissions: { contents: read }` at the top of `ci.yml` and elevate per-job only where needed. Note `release` only `needs` test/lint/typecheck — **the `security` job (bandit/pip-audit) is not a release gate**, and it runs `pip-audit` with `continue-on-error: true`, so a vulnerable dependency cannot block a publish (see SC-2).

---

### H-5 — Celery `backtest` task instantiates `BacktestEngine(**params)` with unvalidated, attacker-influenceable kwargs
**Severity: High**
**Files:** `src/qufin/api/jobs.py:267-279`, `src/qufin/backtesting/engine.py`

**Evidence.** Three of the four registered tasks re-validate input through pydantic before use (`OptimizeRequest(**params)`, `PriceRequest(**params)`, `RiskRequest(**params)` — jobs.py:231/245/259), which is good. The fourth does **not**:
```python
# jobs.py:271-275
from qufin.backtesting.engine import BacktestEngine
engine = BacktestEngine(**params)   # no schema validation
result = engine.run()
```
Combined with the unauthenticated broker (C-2), `params` is attacker-controllable. `BacktestEngine` takes a `strategy_fn: Callable` and arbitrary config; arbitrary keyword expansion into a constructor is a kwargs-injection sink.

**Impact.** At minimum, attacker-controlled `params` can trigger unexpected constructor behavior, resource exhaustion (huge windows/iterations), or exceptions used for error-based info leak. The JSON-only broker prevents passing a live callable, which caps this below full RCE — but it is a clear validation gap that diverges from the other three handlers and should not exist on a task fed by an untrusted queue.

**Remediation.** Define a `BacktestRequest` pydantic model and validate `params` before constructing the engine; whitelist `strategy` by name (never accept arbitrary callables/dotted paths from the queue); bound numeric parameters. Fix C-2 in tandem.

---

## MEDIUM

### M-1 — Error/exception text returned to clients and stored in job records
**Severity: Medium**
**Files:** `src/qufin/api/server.py:804-807,828-832`, `src/qufin/api/jobs.py:395-396,445,478,493-495`

**Evidence.** `str(exc)` from internal failures is propagated into client-visible fields and job metadata: `meta.error = str(exc)` (jobs.py:396, 445, 493), `raise RuntimeError(f"Job {job_id} failed: {exc}")` (jobs.py:495), and `JobStatusResponse.error` is returned verbatim to API callers (server.py:808-814). 404 messages echo the user-supplied `job_id` (`f"Job {job_id} not found"`).

**Impact.** Raw exception strings can leak dependency versions, file paths, broker URLs (which embed Redis host/db), and stack-trace-adjacent internals — reconnaissance that aids further attacks. Lower severity because messages are `str(exc)`, not full tracebacks, but they are uncontrolled.

**Remediation.** Return a generic client message + correlation id; log the detailed exception server-side only. Never surface broker/backend strings to clients.

### M-2 — In-memory rate limiter and job store are ineffective in the shipped multi-worker topology
**Severity: Medium**
**Files:** `src/qufin/api/server.py:244-276,284-335`, `Dockerfile:60`

**Evidence.** `RateLimiter` and `InMemoryJobStore` keep state in process-local dicts. The container runs **gunicorn with `--workers 4`** (Dockerfile:60) and the Helm chart runs `replicaCount: 2`. Rate-limit state is keyed by `request.client.host` (server.py:702) and never shared across workers/replicas; the job store is likewise per-process.

**Impact.** Effective rate limit is multiplied by `workers × replicas` (here up to 8×), undermining the 60/min default and weakening DoS protection. Async jobs created on one worker are invisible to `GET /v1/jobs/{id}` served by another (correctness + a security-relevant inconsistency for any future authz tied to job ownership). Behind a proxy, `request.client.host` is the proxy IP unless `X-Forwarded-For` is trusted — so all clients may share one bucket.

**Remediation.** Back rate limiting and the job store with Redis (already a dependency) so state is shared; derive client identity from a trusted `X-Forwarded-For`/auth principal, not the socket peer.

### M-3 — No NetworkPolicy; Redis and API reachable cluster-wide
**Severity: Medium**
**Files:** `k8s/templates/` (absence)

**Evidence.** No `NetworkPolicy` resource exists. The Redis Service (`service.yaml:57-74`) is a cluster-wide `ClusterIP`; any pod in the cluster can connect to 6379 (which, per C-2, has no auth).

**Impact.** Flat east-west network: a compromise of any unrelated workload in the cluster yields direct access to the unauthenticated broker/cache. Defense-in-depth gap.

**Remediation.** Add default-deny ingress/egress `NetworkPolicy` and explicitly allow only api→redis and worker→redis on 6379; restrict API ingress to the ingress controller.

### M-4 — No security/transport headers or HTTPS enforcement; no request size limits
**Severity: Medium**
**Files:** `src/qufin/api/server.py:672-678` (no `add_middleware`)

**Evidence.** The app adds no middleware: no HSTS, `X-Content-Type-Options`, `X-Frame-Options`, or CSP; no enforced TLS; no maximum request-body size. `docs_url="/docs"` and `openapi_url` are unauthenticated (server.py:676-677), exposing the full API schema. (Note: absence of CORS middleware is *not* a vuln — FastAPI adds no permissive `Access-Control-Allow-Origin`, so browsers block cross-origin by default. There is no `allow_origins=["*"]` anywhere.)

**Impact.** Missing hardening headers; unbounded request bodies enable memory-pressure DoS (e.g. a massive `portfolio_weights` dict — only `tickers` has `min_length`, none have max bounds). Public OpenAPI eases enumeration on an already-unauthenticated API.

**Remediation.** Add a security-headers middleware, enforce TLS at the ingress with HSTS, set a body-size limit (proxy or middleware), cap collection sizes in the pydantic models (`max_length` on `tickers`, key/size caps on `portfolio_weights`), and gate `/docs`+`/openapi.json` behind auth in production.

### M-5 — `SECURITY.md` makes claims the code/CI don't fully honor
**Severity: Medium**
**Files:** `SECURITY.md`, `pyproject.toml`, `.github/workflows/ci.yml`

**Evidence.** SECURITY.md asserts: *"Supported version 0.1.x"* (the repo is v1.1.0 — `Chart.yaml`/`values.yaml`), *"Input validation at API boundaries"* (true for 3 endpoints, **not** the Celery backtest task — H-5), and *"Pinned dependency ranges — lower bounds enforced, upper bounds only where necessary"* (most deps have **no upper bound** — SC-1). The "no dangerous builtins / no secrets" claims *are* accurate.

**Impact.** A security policy that overstates guarantees creates false assurance for downstream consumers and auditors and understates patch scope (wrong supported-version line).

**Remediation.** Correct the supported-version table to 1.1.x, qualify the input-validation claim, and align the dependency-pinning statement with reality (or fix SC-1 and keep the claim).

### M-6 — CLI writes a temp parquet with `delete=False` and never cleans it up
**Severity: Medium**
**Files:** `src/qufin/cli.py:122-128`

**Evidence.**
```python
with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as fd:
    tmp_path = fd.name
df.to_parquet(tmp_path, index=False)
return tmp_path
```
This is the *safe* API (no `mktemp` race), but `delete=False` with no cleanup leaves result data in a world-readable temp dir indefinitely.

**Impact.** Low-grade information disclosure of computed financial outputs on shared/multi-user hosts; unbounded temp-dir growth. (Listed Medium-low because exploitation requires local access.)

**Remediation.** Default to writing under a user-owned cache dir with `0600` perms, or document/clean the temp file; create it with `mode=0o600` and an explicit lifecycle.

---

## LOW

### L-1 — 23 `B101 assert_used` in library code
**Files:** e.g. `risk/quantum_linear_systems.py:348`, `portfolio/optimizers/quantum_ipm.py:301`, `ml/*`, `options/implied_vol_surface.py:512,634`.
`assert` is stripped under `python -O`; these guard internal invariants (power-of-2 dims, fit-before-predict). Not security-critical, but invariant checks should not rely on `assert`. **Fix:** raise `ValueError`/`RuntimeError` for any check that must hold at runtime.

### L-2 — `try/except/pass` swallows errors silently (`B110`)
**Files:** `api/jobs.py:448`, `backends/braket_backend.py:429`, `backends/cirq_backend.py:624`, `benchmarks/*`.
Silent failure can mask broker/state errors (e.g. jobs.py:448 swallows all Celery status-sync exceptions). **Fix:** log at debug/warning with `exc_info`.

### L-3 — `git` invoked via `subprocess` with partial path (`B404/B603/B607`)
**File:** `benchmarks/manifest.py:45-51`. Hardcoded `["git", ...]` arg list, **no `shell=True`**, output only used for a benchmark manifest. False-positive-grade; harmless. **Fix (optional):** use an absolute git path or `shutil.which`.

### L-4 — Dependabot covers pip + actions but not Docker base images
**File:** `.github/dependabot.yml`. No `package-ecosystem: docker` entry, so `python:3.12-slim`/`redis:7-alpine` never get automated bump PRs. **Fix:** add a `docker` ecosystem watcher for `/` (Dockerfile) once images are digest-pinned (H-3).

### L-5 — `data/__init__.py` exposes provider factories with empty-string default keys
**File:** `src/qufin/data/__init__.py:16-19` (`get_refinitiv_source(app_key="")`) and `streaming.StreamConfig(api_key="")`. Empty-string defaults are not secrets, but they let a caller silently create an unauthenticated/misconfigured client that fails opaquely at the vendor. **Fix:** require the key explicitly or raise a clear error when empty.

---

## Positives (verified, worth preserving)

- **No insecure deserialization.** Celery is JSON-only (`accept_content=["json"]`, jobs.py:204); the result cache is JSON-only (`cache.py`). No `pickle`/`marshal`/`yaml.load` anywhere in `src/`.
- **No dangerous primitives.** Repo-wide grep found no `eval`/`exec`/`os.system`/`shell=True`/`__import__`/`torch.load`/`np.load(allow_pickle)`/`verify=False`/`mktemp`.
- **No hardcoded or committed secrets.** No `.env`/`.pem`/`.key` tracked; no high-entropy literals; `settings.py` uses `pydantic-settings` env vars; FRED/Eikon/IBM credentials flow through their SDKs and are **never logged or put in URLs/query strings** (Bloomberg logs only `host:port`; the lone HTTP egress in `crypto.py` carries no key).
- **Strong API input validation.** Every REST endpoint uses pydantic models with bounded `Field` constraints; enums constrain method selection.
- **SQL-safe cache.** `SQLiteCacheBackend` uses parameterized queries throughout (`cache.py`); cache keys are SHA-256.
- **Dockerfile fundamentals.** Multi-stage, non-root `USER qufin`, apt cache cleaned, `HEALTHCHECK` present.
- **CI/CD trust boundaries.** No `pull_request_target`; release job gated on tests and scoped to tags with trusted publishing.

---

## Prioritized remediation order

1. **C-1** — fail-closed auth + wire `api_keys` from a Secret (and scope `GET /v1/jobs`).
2. **C-2 / H-1** — Redis AUTH+TLS in a Secret; remove host port; move tokens to Secrets.
3. **H-2 / M-3** — add `securityContext` (restricted PSA) + default-deny NetworkPolicy.
4. **H-3 / L-4** — digest-pin all images; add Docker Dependabot + image scanning.
5. **H-4** — SHA-pin Actions; add top-level least-privilege `permissions`; make `pip-audit` a hard gate.
6. **H-5 / M-1 / M-2 / M-4** — validate the backtest task; sanitize error output; shared Redis-backed rate-limit/job store; security headers + body-size/collection caps.
7. **SC-1 / M-5** — add a hashed lockfile and constrained upper bounds; correct SECURITY.md.

---

# Supply-chain detail

### SC-1 — No lockfile, no hashes, unbounded dependency floors
**Severity: High (structural)** • **File:** `pyproject.toml:36-78`
Most runtime deps declare only a `>=` floor with no upper bound (`scipy>=1.11`, `pandas>=2.1`, `pyarrow>=15.0`, `cvxpy>=1.4`, `scikit-learn>=1.4`, `statsmodels>=0.14`, `arch>=6.3`, `yfinance>=0.2.40`, `fredapi>=0.5`, `networkx>=3.2`, `matplotlib>=3.8`, plus all optional extras). Only `numpy>=1.26,<3.0` and `qiskit>=1.0,<3.0` are capped — and `qiskit>=1.0,<3.0` deliberately **spans a major version** (1.x→2.x have breaking API changes). There is **no lockfile** (`requirements*.lock`, `uv.lock`, `poetry.lock`) and **no hash pinning** anywhere, including the Docker build (`pip install . && pip install uvicorn[standard] celery[redis] redis gunicorn` with no constraints/hashes). **Impact:** builds are non-reproducible; a newly-published malicious or breaking transitive release is pulled silently with no integrity check (no defense against a compromised PyPI artifact). For an enterprise target this is the single biggest supply-chain weakness. **Fix:** generate a hashed lock (`uv lock`/`pip-compile --generate-hashes`), install with `--require-hashes` in Docker, add tested upper bounds, and split the qiskit 1.x/2.x support into separate, tested constraints.

### SC-2 — `pip-audit` is non-blocking and not a release gate
**Severity: Medium** • **File:** `.github/workflows/ci.yml:64-65,78-80`
`pip-audit` runs with `continue-on-error: true`, and the `release` job's `needs:` excludes the `security` job entirely. A dependency CVE therefore cannot fail CI or block a PyPI publish. **Fix:** remove `continue-on-error` (or fail on High/Critical), add `security` to the release `needs:`, and run `pip-audit` against the **locked** set from SC-1.

### SC-3 — pip-audit environment findings (informational)
The 23 CVEs pip-audit reported (`cryptography 41.0.7`, `idna 3.11`, `pip 24.0`, `pyjwt 2.7.0`, `setuptools 68.1.2`, `urllib3 2.6.3`, `wheel 0.42.0`) are in the **review machine's ambient site-packages**, not qufin's dependency closure — none are listed in `pyproject.toml`. They are not qufin vulnerabilities, but they illustrate why a hashed lock + CI audit against the real tree (SC-1/SC-2) matters: without one, whatever the build host happens to resolve is what ships.
