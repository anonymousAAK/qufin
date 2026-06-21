# --------------------------------------------------------------------------
# qufin multi-stage Dockerfile
# --------------------------------------------------------------------------
# Build:   docker build -t qufin:latest .
# GPU:     docker build --build-arg GPU=1 -t qufin:gpu .
# Run:     docker run -p 8000:8000 -e QUFIN_API_KEYS="key1,key2" qufin:latest
#
# REQUIRED runtime env:
#   QUFIN_API_KEYS   Comma-separated API keys. The container REFUSES to start
#                    with authentication disabled — set at least one key.
#                    (Set QUFIN_ALLOW_NO_AUTH=1 only for trusted, isolated dev.)
# Recommended runtime env:
#   REDIS_URL, CELERY_BROKER_URL, CELERY_RESULT_BACKEND (with credentials),
#   QUFIN_LOG_LEVEL, QUFIN_CACHE_BACKEND
# --------------------------------------------------------------------------

# ---- Stage 1: builder (install deps, build wheel) -----------------------
# Pinned by digest for reproducibility / supply-chain integrity.
# Tag at pin time: python:3.12-slim (resolved 2026-06-04).
FROM python:3.14-slim@sha256:c845af9399020c7e562969a13689e929074a10fd057acd1b1fad06a2fb068e97 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy the pinned, reproducible constraints first (cache-friendly).
COPY pyproject.toml README.md LICENSE constraints.txt ./
COPY src/ src/
# Install against the known-good pinned set so builds are reproducible.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-compile --prefix=/install -c constraints.txt . && \
    pip install --no-compile --prefix=/install \
        "uvicorn[standard]" "celery[redis]" redis gunicorn

# ---- Stage 2: slim runtime ----------------------------------------------
# Same digest pin as the builder stage.
FROM python:3.14-slim@sha256:c845af9399020c7e562969a13689e929074a10fd057acd1b1fad06a2fb068e97 AS runtime

ARG GPU=0

LABEL maintainer="Adarsh Keshri" \
      description="qufin: quantum algorithms for quant finance"

# Copy pre-built packages from builder
COPY --from=builder /install /usr/local

# GPU variant: install CUDA-Q on top (only when GPU=1)
RUN if [ "$GPU" = "1" ]; then \
        pip install --no-cache-dir cuda-quantum; \
    fi && \
    rm -rf /root/.cache/pip

# Non-root user (UID/GID 1000 to match the K8s securityContext)
RUN groupadd -r -g 1000 qufin && \
    useradd -r -u 1000 -g qufin -m qufin

WORKDIR /app
COPY --chown=qufin:qufin src/ src/
COPY --chown=qufin:qufin pyproject.toml README.md LICENSE ./

USER qufin

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    QUFIN_LOG_LEVEL=INFO \
    GUNICORN_WORKERS=4 \
    GUNICORN_TIMEOUT=300

EXPOSE 8000

# Liveness against the API health endpoint (served by qufin.api.server:/health).
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=4).status==200 else 1)" || exit 1

# Fail-closed auth wiring:
#   * QUFIN_API_KEYS must be set (comma-separated) OR QUFIN_ALLOW_NO_AUTH=1.
#   * create_app() reads QUFIN_API_KEYS from the environment and fails closed
#     when it is absent, so the container never launches key-less by default.
#     Keys are never passed on the command line (avoids leaking them via `ps`).
CMD ["sh", "-c", "\
if [ -z \"$QUFIN_API_KEYS\" ] && [ \"$QUFIN_ALLOW_NO_AUTH\" != \"1\" ]; then \
  echo 'FATAL: QUFIN_API_KEYS is not set. Refusing to start with auth disabled.' >&2; \
  echo '       Set QUFIN_API_KEYS=\"key1,key2\" (recommended) or QUFIN_ALLOW_NO_AUTH=1 for trusted/isolated dev only.' >&2; \
  exit 1; \
fi; \
exec gunicorn 'qufin.api.server:create_app()' \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --workers \"${GUNICORN_WORKERS:-4}\" \
  --timeout \"${GUNICORN_TIMEOUT:-300}\"\
"]
