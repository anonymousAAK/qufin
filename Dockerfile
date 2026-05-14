# --------------------------------------------------------------------------
# qufin multi-stage Dockerfile
# --------------------------------------------------------------------------
# Build:   docker build -t qufin:latest .
# GPU:     docker build --build-arg GPU=1 -t qufin:gpu .
# Run:     docker run -p 8000:8000 qufin:latest
# --------------------------------------------------------------------------

# ---- Stage 1: builder (install deps, build wheel) -----------------------
FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY pyproject.toml README.md LICENSE ./
COPY src/ src/
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-compile --prefix=/install . && \
    pip install --no-compile --prefix=/install uvicorn[standard] celery[redis] redis gunicorn

# ---- Stage 2: slim runtime ---------------------------------------------
FROM python:3.12-slim AS runtime

ARG GPU=0

LABEL maintainer="anonymousAAK" \
      description="qufin: quantum algorithms for quant finance"

# Copy pre-built packages from builder
COPY --from=builder /install /usr/local

# GPU variant: install CUDA-Q on top (only when GPU=1)
RUN if [ "$GPU" = "1" ]; then \
        pip install --no-cache-dir cuda-quantum; \
    fi

# Non-root user
RUN groupadd -r qufin && useradd -r -g qufin -m qufin
USER qufin

WORKDIR /app
COPY --chown=qufin:qufin src/ src/
COPY --chown=qufin:qufin pyproject.toml README.md LICENSE ./

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    QUFIN_LOG_LEVEL=INFO

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["gunicorn", "qufin.api.server:create_app()", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--workers", "4", \
     "--timeout", "300"]
