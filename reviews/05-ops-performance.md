# qufin Production-Readiness Review — Ops & Performance
**Branch:** claude/determined-fermi-y8m78  
**Date:** 2026-06-04  
**Reviewer:** Principal SRE / Platform Engineering  
**Scope:** REST API, Job Queue, Caching, Observability, Config, Resilience, Concurrency, Performance, Deployment

---

## Severity Summary

| Severity | Count |
|---|---|
| Critical | 7 |
| High | 9 |
| Medium | 8 |
| Low | 5 |
| **Total** | **29** |

---

## Verdict

The "Enterprise" tier claims (REST API, Celery+Redis job queue, caching, Docker, K8s) are, at best, structural scaffolding — not a working system. The async job pipeline is architecturally broken: the API server and the Celery subsystem are two disconnected islands, and submitted async jobs will hang in `PENDING` forever. The Celery worker command references a non-existent module-level `app`, so the worker cannot even start. Market data computations in the synchronous API path use **hardcoded synthetic random numbers** instead of real prices, exposing a critical data-integrity hole disguised behind a silent try/except fallback pattern. There is zero observability (no structured logging, no metrics, no tracing anywhere in the API). The rate limiter and job store are in-memory singletons that vanish on restart and diverge across the 4 Gunicorn workers. API keys are not managed through the settings system and are injected as plaintext values in a Kubernetes `ConfigMap` alongside quantum-hardware tokens. IBM hardware jobs block the event loop indefinitely. Quantum algorithms expose 2^n exponential loops without any asset-count guardrail. This codebase is not operable at scale today; it requires substantial re-engineering before any enterprise deployment claim is credible.

---

## Critical Findings

### C-1: Async Jobs Are Never Executed — Permanent `PENDING` State
**Severity:** Critical  
**File:** `src/qufin/api/server.py:741–786`, `src/qufin/api/jobs.py`

**Evidence:**  
The `async_mode=True` path in all three endpoints (`/v1/optimize`, `/v1/price`, `/v1/risk`) calls `_job_store.create(...)` and immediately returns a `JobResponse(status=PENDING)`. `_job_store` is an `InMemoryJobStore` — a plain `dict`. Nothing dispatches work to a Celery queue. The `JobQueue` class in `jobs.py` is never imported or used anywhere in `server.py`. The `/v1/jobs/{job_id}` poll endpoint reads from the same `InMemoryJobStore` and will return `PENDING` indefinitely.

```python
# server.py:742-743 — jobs are accepted but never dispatched
job_id = _job_store.create("optimization", req.model_dump())
return JobResponse(job_id=job_id, status=JobStatus.PENDING)
```

**Impact:** Any client using `async_mode=True` receives a `job_id` that will never transition out of `PENDING`. This is the primary advertised mode for long-running quantum computations. It is completely non-functional.

**Recommendation:** Wire the endpoint handlers to `JobQueue.submit()` (from `jobs.py`), or remove the `async_mode` feature from the public API until it is implemented. Store Celery `AsyncResult` references in Redis, not in an in-memory dict.

---

### C-2: Celery Worker Cannot Start — No Module-Level App Object
**Severity:** Critical  
**File:** `docker-compose.yml:28`, `k8s/templates/worker.yaml:29`, `src/qufin/api/jobs.py`

**Evidence:**  
Both the compose file and Helm chart instruct Celery to autodiscover its app via:
```
celery -A qufin.api.jobs worker
```
Celery's `-A` flag requires a module-level `Celery` instance named `app` (or similar). Searching `jobs.py` shows no module-level `app` assignment — `create_celery_app()` is a factory function, and `celery_app` is a local variable inside it. Running this command will produce `AttributeError: module 'qufin.api.jobs' has no attribute 'app'` and the worker will refuse to start.

**Impact:** The entire Celery worker tier is dead on arrival in both compose and Kubernetes deployments.

**Recommendation:** Add `app = create_celery_app(broker_url=os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0"))` at module level in `jobs.py`, respecting environment variable configuration.

---

### C-3: Synthetic Data Silently Returned Instead of Real Market Data
**Severity:** Critical  
**File:** `src/qufin/api/server.py:349–353`, `src/qufin/api/server.py:521–523`

**Evidence:**  
`_run_optimize()` and `_run_risk()` both generate synthetic Gaussian returns using `np.random.default_rng(42)` with a **hardcoded seed**, and comment `# Synthetic returns (qufin.data.market not available)`. All exceptions from the real computation path are silently swallowed and fall back to equal-weight or random data. A client requesting QAOA optimization on real tickers receives weights computed from fake data with no indication in the response.

```python
# server.py:349-352 — always synthetic, seed fixed
rng = np.random.default_rng(42)
n_days = 252
returns = rng.normal(0.0005, 0.02, size=(n_days, len(req.tickers)))
```

**Impact:** The API produces numerically deterministic but **financially meaningless** results for every request. This is a data-integrity defect, not a performance issue. In a production trading environment, acting on these outputs could cause significant financial harm.

**Recommendation:** Integrate `YahooEquityProvider` (or another real data source) into the API handlers. If data fetching fails, return HTTP 503 or 422 with an explicit error — never silently substitute synthetic data.

---

### C-4: IBM Quantum `job.result()` Blocks the Async Event Loop
**Severity:** Critical  
**File:** `src/qufin/backends/ibm_runtime.py:93`

**Evidence:**  
```python
result = job.result()  # blocking — no timeout
```
`IBMRuntimeBackend.run()` is a synchronous blocking call. Real IBM hardware jobs can queue for minutes to hours. When called from a FastAPI `async def` handler (without `asyncio.to_thread()` or `run_in_executor()`), this blocks the entire uvicorn event loop, making the server unresponsive to all other requests for the job's duration.

**Impact:** A single QAE request using real IBM hardware renders the API completely unavailable until IBM returns. With `--workers 4` (Gunicorn), four concurrent such requests saturate all workers.

**Recommendation:** Wrap all quantum backend `.run()` calls in `asyncio.to_thread()` within FastAPI handlers, or enforce that hardware jobs always use `async_mode=True` and execute via Celery. Set a hard timeout on `job.result()`.

---

### C-5: All Exception Handlers Silently Swallow Errors and Return Wrong Results
**Severity:** Critical  
**File:** `src/qufin/api/server.py:366, 375, 390, 469, 486, 538, 555, 575`

**Evidence:**  
Every computation branch is wrapped in `except Exception: <fallback>` with no logging and no indication to the caller that the requested method failed. A request for `method=qaoa` that throws an `ImportError` or `MemoryError` returns an equal-weight portfolio with `method="qaoa"` in the response — indistinguishable from a successful QAOA run.

```python
except Exception:
    # Simple equal-weight fallback
    w = np.ones(n) / n
```

**Impact:** Operators have no visibility into computation failures. Clients cannot distinguish correct quantum results from silent fallbacks. Systematic failures (e.g., missing Qiskit install) are invisible.

**Recommendation:** At minimum, log the exception with stack trace. Consider returning HTTP 500 or a response field `fallback_reason` when the requested method fails. The current behavior violates the principle of least surprise.

---

### C-6: API Keys and Quantum Tokens Stored in Kubernetes ConfigMap (Plaintext)
**Severity:** Critical  
**File:** `k8s/templates/configmap.yaml:22–27`

**Evidence:**  
```yaml
{{- if .Values.config.fredApiKey }}
QUFIN_FRED_API_KEY: {{ .Values.config.fredApiKey | quote }}
{{- end }}
{{- if .Values.config.ibmQuantumToken }}
IBM_QUANTUM_TOKEN: {{ .Values.config.ibmQuantumToken | quote }}
{{- end }}
```
`ConfigMap` data is stored in plaintext in etcd. Any pod with `kubectl get configmap` RBAC access (including many service accounts) can read these values. There are no `kind: Secret` resources in the Helm chart.

**Impact:** Credential exposure to any cluster workload with basic read access. IBM Quantum tokens grant billable hardware access.

**Recommendation:** Move `fredApiKey` and `ibmQuantumToken` to `kind: Secret` with `valueFrom.secretKeyRef` in pod specs. Use an external secrets manager (Vault, AWS Secrets Manager, Sealed Secrets) for production. Do not store credentials in `values.yaml` defaults.

---

### C-7: Rate Limiter Is In-Memory, Not Thread-Safe, Not Process-Safe
**Severity:** Critical  
**File:** `src/qufin/api/server.py:255–276`

**Evidence:**  
`RateLimiter` uses a plain `defaultdict(list)` with no locking. Gunicorn runs `--workers 4` (Dockerfile line 61), meaning 4 separate processes each have their own `RateLimiter` instance. A client can make 4×60 = 240 requests per minute per worker process before any single instance rate-limits it. Within a single async worker, `check_rate_limit` mutates `_requests[key]` (read-then-write) without a lock — concurrent coroutines can race.

**Impact:** Rate limiting provides effectively zero protection in production multi-process deployment. A malicious or runaway client can exhaust server capacity.

**Recommendation:** Replace with Redis-backed rate limiting (e.g., `slowapi` with Redis backend, or a custom `INCR`/`EXPIRE` Redis counter). At minimum, add a `threading.Lock` around `is_allowed()`.

---

## High Findings

### H-1: Import Time 1.33 Seconds — Blocks K8s Readiness Probes
**Severity:** High  
**File:** `src/qufin/__init__.py`

**Evidence:** From `importtime` measurement: `import qufin` takes **1,330,323 µs (1.33 seconds)**. The package eagerly imports all submodules at the top level, including heavy quantum modules (`qufin.options.amplitude_estimation` = 14.1ms cumulative, `qufin.risk` = 132ms). The K8s readiness probe has `initialDelaySeconds: 5` — a slow machine or cold container cache could breach this.

**Impact:** Slow pod startup; increased time-to-serve on scale-out events; resource waste.

**Recommendation:** Use lazy imports for heavy optional submodules. The top-level `__init__.py` should not eagerly import `backends`, `risk`, `options`, and `viz` — use `__getattr__` lazy loading or remove the top-level star import.

---

### H-2: Celery Broker URL Hardcoded to `localhost` — Silently Falls Back
**Severity:** High  
**File:** `src/qufin/api/jobs.py:163, 308`

**Evidence:**  
```python
def create_celery_app(broker_url: str = "redis://localhost:6379/0", ...):
```
The `Settings` class in `settings.py` has no `CELERY_BROKER_URL` field. The docker-compose and K8s configs set `CELERY_BROKER_URL` as an environment variable, but nothing in `jobs.py` reads it — the default `localhost` is used unless the caller explicitly passes it. If `JobQueue` is instantiated without arguments (which it is in `JobQueue.__init__`), it will fail to connect to the service-mesh Redis host and silently disable Celery (`self._celery_app = None`), falling back to synchronous in-process execution.

**Impact:** Silent loss of the distributed job queue in any containerized environment.

**Recommendation:** Add `celery_broker_url` and `celery_result_backend` to `Settings` and read them via `QUFIN_CELERY_BROKER_URL`.

---

### H-3: In-Memory Job Store Lost on Restart / Not Shared Across Workers
**Severity:** High  
**File:** `src/qufin/api/server.py:284–335`, `src/qufin/api/server.py:669`

**Evidence:**  
`_job_store = InMemoryJobStore()` is a module-level dict. Each Gunicorn worker has its own copy. A job created by worker A is invisible to worker B. On pod restart, all jobs disappear. The `/v1/jobs` endpoint is documented as production job management but is a volatile, non-shared store.

**Impact:** Jobs lost on any restart or rolling deployment. Clients polling job status from a different worker than the one that accepted the job will receive 404.

**Recommendation:** Job state must be persisted in Redis or a database. This is one of the core requirements for a distributed job queue. The `JobQueue` class in `jobs.py` already models this correctly via Celery — the `InMemoryJobStore` must be replaced.

---

### H-4: No Readiness Endpoint — Liveness and Readiness Are Identical
**Severity:** High  
**File:** `src/qufin/api/server.py:711–722`, `k8s/templates/deployment.yaml:35–44`

**Evidence:**  
Both `livenessProbe` and `readinessProbe` use `GET /health`, which only checks that the Python process started and can import `qufin.__version__`. It does not check Redis connectivity, Celery worker availability, or any dependency health. If Redis goes down, the pod remains `Ready` and continues receiving traffic, returning errors.

**Impact:** Traffic is routed to a degraded pod. K8s cannot remove unhealthy pods from the service. The liveness/readiness distinction is meaningless.

**Recommendation:** Add a dedicated `/ready` endpoint that checks Redis ping and returns 503 if dependencies are unavailable. Keep `/health` as liveness-only.

---

### H-5: Celery Worker Has No Health/Liveness Probe
**Severity:** High  
**File:** `k8s/templates/worker.yaml`

**Evidence:**  
The worker `Deployment` has no `livenessProbe` or `readinessProbe`. Celery workers can silently deadlock (e.g., stuck waiting for IBM hardware), leak file descriptors, or enter a zombie state while appearing healthy to K8s.

**Impact:** A stuck worker consumes resources and holds task leases (`task_acks_late=True`) without making progress. K8s will never restart it.

**Recommendation:** Add a `livenessProbe` using `celery -A qufin.api.jobs inspect ping` or a custom `celery-healthcheck` binary. Set a `terminationGracePeriodSeconds` that allows in-flight tasks to complete.

---

### H-6: Exponential-Time Algorithms Reachable via API With No Asset-Count Guard
**Severity:** High  
**File:** `src/qufin/portfolio/optimizers/grover_search.py:84,102`, `src/qufin/portfolio/optimizers/robust.py:286`

**Evidence:**  
`_compute_all_energies()` iterates `range(2**n)` over all bitstrings, where `n` is the number of assets. The `RobustOptimizer` does the same. These are called through the `QAOA` optimization path accessible via `/v1/optimize`. There is no `max_length` on the `tickers` field in `OptimizeRequest` (only `min_length=2`). A 30-asset request enumerates 2^30 ≈ 1 billion combinations; a 40-asset request requires terabytes of memory.

**Impact:** A single API request with 30+ tickers OOM-kills the worker pod. With no authentication enabled by default, this is an unauthenticated DoS vector.

**Recommendation:** Add `max_length=25` (or similar) to `tickers` in `OptimizeRequest`. Add asset-count validation inside `_compute_all_energies()` that raises `ValueError` above a safe threshold (e.g., 20 assets).

---

### H-7: Zero Observability — No Structured Logging, No Metrics, No Tracing in API
**Severity:** High  
**File:** `src/qufin/api/server.py`, `src/qufin/api/jobs.py`, `src/qufin/utils/logging.py`

**Evidence:**  
- `server.py` has zero `import logging` or `logger.*` calls. No request logging, no error logging, no computation-time logging.
- `logging.py` uses plain text format (`%(asctime)s [%(name)s] %(levelname)s: %(message)s`) — not JSON structured logs.
- `grep prometheus_client src/` returns zero results.
- `grep opentelemetry src/` returns zero results.
- There is no `/metrics` endpoint.
- The `get_queue_stats()` method in `jobs.py` is never exposed via API.

**Impact:** An operator cannot determine: request latency percentiles, error rates, queue depth, active worker count, or the cause of a stuck job. Debugging a production incident is nearly impossible.

**Recommendation:** (1) Add structlog or `python-json-logger` for structured JSON logging in `logging.py`. (2) Add `prometheus-client` with a `/metrics` endpoint and counters/histograms for request latency, queue depth, and error rates. (3) Add OpenTelemetry instrumentation for distributed tracing of quantum job execution paths.

---

### H-8: Redis Used as Both Broker and Result Backend on the Same DB Index
**Severity:** High  
**File:** `docker-compose.yml:16–17`, `k8s/templates/configmap.yaml:16–17`

**Evidence:**  
```yaml
CELERY_BROKER_URL: "redis://redis:6379/1"
CELERY_RESULT_BACKEND: "redis://redis:6379/1"
```
Broker and result backend share DB index 1. While Celery technically supports this, mixing the queue data structures (lists) with result key-value data in the same DB makes `FLUSHDB` dangerous, complicates monitoring, and increases the blast radius of a `result_expires` misconfiguration that clears broker messages.

**Impact:** A result store flush clears pending task queue messages. Monitoring Redis keyspace stats is ambiguous.

**Recommendation:** Separate broker (DB 0 or 1) from result backend (DB 2). Consider using RabbitMQ for the broker, which provides proper AMQP dead-lettering absent here.

---

### H-9: No Dead-Letter Queue — Failed Tasks Vanish
**Severity:** High  
**File:** `src/qufin/api/jobs.py:202–219`

**Evidence:**  
Celery configuration in `create_celery_app()` has no `task_routes` for dead-lettering, no `task_reject_on_worker_lost`, no `acks_on_failure_or_timeout`. Tasks that fail after 1 retry simply raise the exception, which Celery marks `FAILURE` in the result backend. After `result_expires=3600`, the failure record is deleted. There is no alerting, no DLQ for inspection, and no requeue capability.

**Impact:** Failed quantum jobs disappear after 1 hour. No audit trail. No operator notification. Systematic failures (e.g., IBM quota exhausted) are silent.

**Recommendation:** Add a Celery failure callback (`task_failure_callback`) that publishes to a dead-letter queue or emits a structured log event. Integrate with alerting (PagerDuty, Slack) on repeated failures.

---

## Medium Findings

### M-1: `CachedTranspiler` Has a TOCTOU Race Condition
**Severity:** Medium  
**File:** `src/qufin/utils/circuit_cache.py:97–118`

**Evidence:**  
The `transpile()` method checks the cache under the lock, then releases the lock to call `qiskit.compiler.transpile()`, then re-acquires the lock to insert. Two concurrent threads with the same cache key will both miss, both transpile (wasted work), and the second writer silently overwrites the first. `self._stats.misses` is incremented twice for one logical miss.

**Impact:** Redundant transpilation overhead; inaccurate cache statistics. Not data-corrupting but defeats the purpose of the cache under concurrent load.

**Recommendation:** Use a per-key future/lock pattern (promise-based) or accept double-transpile as the cost of release-outside-lock design. Document the intentional race.

---

### M-2: Settings Singleton Is Not Safe for Multi-Worker Override
**Severity:** Medium  
**File:** `src/qufin/utils/settings.py:37–40`

**Evidence:**  
```python
global _settings
if _settings is None or overrides:
    _settings = Settings(**overrides)
```
The `global` mutation (`PLW0603`) is unguarded by a lock. Under Gunicorn with `--workers 4` (processes), each process gets its own copy — so this is process-safe but not meaningful for shared config. Under threads (e.g., `--worker-class gthread`), two threads calling `get_settings(overrides=...)` simultaneously could race. More critically, `overrides` being truthy causes a new `Settings` to be constructed on every call — potentially re-reading environment variables mid-request.

**Recommendation:** Remove the `or overrides` branch; use a separate `Settings(**overrides)` constructor for test isolation. Add a threading lock if threaded workers are ever used.

---

### M-3: PVC `accessMode: ReadWriteOnce` with `replicaCount: 2`
**Severity:** Medium  
**File:** `k8s/values.yaml:51`, `k8s/templates/deployment.yaml`

**Evidence:**  
The API deployment defaults to `replicaCount: 2` but the `PersistentVolumeClaim` uses `accessMode: ReadWriteOnce`. On most cloud storage backends (EBS, Azure Disk), `ReadWriteOnce` allows only one node to mount the volume at a time. When the second pod lands on a different node, it will be stuck in `ContainerCreating`.

**Impact:** Rolling deployments stall. The second replica may never become ready, defeating HA.

**Recommendation:** Use `ReadWriteMany` (NFS, EFS, Azure Files) or remove the shared PVC dependency. If each pod needs local scratch space, use `emptyDir` per pod.

---

### M-4: No Kubernetes `Secret` Resource — API Keys via `values.yaml` Defaults
**Severity:** Medium  
**File:** `k8s/values.yaml:63–64`

**Evidence:**  
```yaml
fredApiKey: ""
ibmQuantumToken: ""
```
Values are set via `--set config.fredApiKey=...` at `helm install`, which appears in `helm history` and `kubectl get events` in plaintext. No Kubernetes `Secret` kind is generated.

**Impact:** Credential leakage through Helm history, CI/CD logs, and cluster audit logs.

**Recommendation:** Create a `templates/secret.yaml` with `kind: Secret` and reference via `secretKeyRef`. Enforce sealed-secrets or external-secrets operator for CI/CD.

---

### M-5: RedisCacheBackend `invalidate_by_algorithm` Is Non-Atomic O(n) Scan
**Severity:** Medium  
**File:** `src/qufin/api/cache.py:370–380`

**Evidence:**  
```python
for rkey in self._client.scan_iter(match=pattern):
    raw = self._client.get(rkey)  # non-atomic read
    if ... data.get("algorithm") == algorithm:
        self._client.delete(rkey)  # separate round-trip
```
This is a GET-then-DELETE with no pipeline or transaction. On a large cache, `scan_iter` can take seconds and block the Redis server for cursory scans. Between the `GET` and `DELETE`, another process can update the key.

**Impact:** Slow cache invalidation under load; potential stale-read window. `scan_iter` without a `count` hint performs full keyspace scans.

**Recommendation:** Use Redis secondary indexes (a Set per algorithm holding member keys) to enable O(1) `SMEMBERS` + `DEL` via pipeline. Alternatively, use Redis key expiry instead of explicit invalidation.

---

### M-6: FRED `get_yield_curve()` Makes 7 Serial Blocking API Calls With No Timeout
**Severity:** Medium  
**File:** `src/qufin/data/macro.py:112–117`

**Evidence:**  
```python
for mat, sid in zip(maturities, series_ids, strict=False):
    data = self._fred.get_series(sid, ...)  # 7 sequential calls
```
The `fredapi` library uses `requests` under the hood. No `timeout` is set, and each call is made serially. FRED rate-limits to ~120 requests/minute per API key. No retry or backoff logic is present.

**Impact:** A single `get_yield_curve()` call can block for 7+ seconds on a slow network, or indefinitely on FRED downtime. Under concurrent load, multiple workers exhaust the FRED rate limit.

**Recommendation:** Parallelize with `asyncio.gather` or `ThreadPoolExecutor`. Add `timeout=10` to each call. Add exponential backoff with `tenacity` or `backoff` library. Cache results with a TTL matching FRED's update frequency (daily for most series).

---

### M-7: No Request Timeout Middleware — Slow Quantum Computations Tie Up Workers
**Severity:** Medium  
**File:** `src/qufin/api/server.py:672–863`

**Evidence:**  
FastAPI has no request timeout middleware configured. The synchronous `_run_optimize()`, `_run_price()`, and `_run_risk()` handlers are called directly from `async def` endpoint functions without `asyncio.to_thread()`. A QAOA simulation or Monte Carlo with `n_simulations=100_000` that takes 60+ seconds blocks the event loop. Gunicorn's `--timeout 300` is a process-level timeout (SIGKILL after 300s) — it does not provide per-request timeout or graceful response.

**Impact:** One slow request (real quantum computation) starves all other requests on that worker for its entire duration.

**Recommendation:** Wrap all `_run_*` calls in `await asyncio.to_thread(...)`. Add `starlette.middleware.timeout.TimeoutMiddleware` with a configurable deadline (e.g., 30s for sync, unlimited for async_mode).

---

### M-8: `list_jobs` Returns All Jobs With No Pagination
**Severity:** Medium  
**File:** `src/qufin/api/server.py:841–861`

**Evidence:**  
```python
@app.get("/v1/jobs", response_model=list[JobStatusResponse])
async def list_jobs(...):
    return [... for j in _job_store.list_jobs()]
```
`InMemoryJobStore.list_jobs()` returns every job ever created (no cleanup runs automatically). The endpoint serializes and returns all of them without limit or pagination. If results were ever persisted, this would be an unbounded response.

**Impact:** Memory exhaustion on response serialization; large payload sizes; accidental full data disclosure.

**Recommendation:** Add `limit` and `offset` (or cursor) query parameters. Call `cleanup_expired()` periodically (e.g., via a background task or scheduled Celery beat task).

---

## Low Findings

### L-1: Gunicorn Entrypoint Uses Invalid Factory Syntax
**Severity:** Low  
**File:** `Dockerfile:57`

**Evidence:**  
```dockerfile
CMD ["gunicorn", "qufin.api.server:create_app()", ...]
```
Gunicorn's factory pattern expects `module:callable` without parentheses: `qufin.api.server:create_app`. The `()` suffix is not valid Gunicorn factory syntax and will cause a startup error.

**Impact:** The Docker image fails to start with the production Gunicorn command. The module-level `app = create_app()` at the bottom of `server.py` means `qufin.api.server:app` would work, but provides no per-deployment configuration.

**Recommendation:** Change to `"qufin.api.server:create_app"` (factory pattern) or `"qufin.api.server:app"` (module-level instance). Verify with `docker run --rm qufin:latest` before releasing.

---

### L-2: `parallel_execute` Uses Threads for CPU-Bound Quantum Simulation
**Severity:** Low  
**File:** `src/qufin/utils/parallel.py:49–58`

**Evidence:**  
`ThreadPoolExecutor` is used for parallel circuit execution. Qiskit Aer simulator is CPU-bound NumPy/C++ code. Python's GIL prevents true parallelism for pure-Python CPU work; Aer releases the GIL for native code, so this may work for Aer but will not for pure-Python simulators. No process-based fallback.

**Impact:** Potential GIL contention reducing effective parallelism for mixed workloads.

**Recommendation:** For CPU-bound backends, use `ProcessPoolExecutor`. Note that Qiskit circuits are not trivially picklable — test before switching.

---

### L-3: Settings `log_level` Defaults to `WARNING` — Too Quiet for Production Debugging
**Severity:** Low  
**File:** `src/qufin/utils/settings.py:26`

**Evidence:**  
`log_level: str = Field(default="WARNING", ...)`. The Dockerfile sets `QUFIN_LOG_LEVEL=INFO`, but the settings default is `WARNING`. Any deployment that does not explicitly set the environment variable will suppress INFO and DEBUG logs, making debugging very difficult.

**Impact:** Reduced operator visibility in misconfigured deployments.

**Recommendation:** Set default to `INFO`. Reserve `WARNING` default for library usage, not server deployments.

---

### L-4: No `PodDisruptionBudget` — Rolling Deploys Can Take API Fully Offline
**Severity:** Low  
**File:** `k8s/templates/` (absent)

**Evidence:**  
Neither the API nor the worker deployment has a `PodDisruptionBudget`. With `replicaCount: 2` for the API, a cluster node drain or rolling upgrade can evict both pods simultaneously.

**Impact:** Zero-downtime deploys not guaranteed. Kubernetes node maintenance causes full API outage.

**Recommendation:** Add `kind: PodDisruptionBudget` with `minAvailable: 1` for both API and worker deployments.

---

### L-5: No `securityContext` on Any Pod — Containers May Run as Root
**Severity:** Low  
**File:** `k8s/templates/deployment.yaml`, `k8s/templates/worker.yaml`

**Evidence:**  
While the `Dockerfile` creates a `qufin` non-root user, no Kubernetes `securityContext` enforces this. The Helm chart does not set `runAsNonRoot: true`, `allowPrivilegeEscalation: false`, or `readOnlyRootFilesystem: true`.

**Impact:** A container escape or misconfigured image could run as root inside the cluster.

**Recommendation:** Add pod-level and container-level `securityContext` fields. Use `runAsNonRoot: true`, `runAsUser: 1000`, `allowPrivilegeEscalation: false`.

---

## Appendix: Evidence Matrix

| Area | Claim | Reality |
|---|---|---|
| Async Job Queue | Celery+Redis distributed jobs | Broken: jobs never dispatched; worker won't start |
| Market Data | Real price-based computation | All API computations use hardcoded synthetic random data |
| Observability | Production-grade API | Zero structured logging, zero metrics, zero tracing |
| Rate Limiting | Per-client protection | In-memory, not shared across 4 workers, no locking |
| Secrets Management | Enterprise security | API tokens in Kubernetes ConfigMap plaintext |
| Horizontal Scale | K8s HPA + multi-replica | PVC ReadWriteOnce blocks second replica on different node |
| Health Checks | Readiness/liveness | Both identical; no dependency health checked |
| Dead-letter | Fault-tolerant queue | No DLQ; failed jobs disappear after 1h |
