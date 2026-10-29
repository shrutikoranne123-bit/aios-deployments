# =============================================================================
# Retrieval API service image (lean, self-contained).
# =============================================================================
# Serves retrieval/retrieval_new.py (FastAPI app `app`) with uvicorn. Much
# lighter than the root ingestion image — no torch / docling / unstructured.
#
# The retrieval package is SELF-CONTAINED: everything it imports lives under
# retrieval/ (including vendored copies of the few shared modules in
# retrieval/_vendor/ and retrieval/sources/). It does NOT need the top-level
# `services/` package, so this image copies only the retrieval folder.
#
# Build from the REPOSITORY ROOT:
#   docker build -f retrieval/Dockerfile -t asma-retrieval:latest .
# (or from inside retrieval/ with:  docker build -t asma-retrieval:latest . )
#
# Run:
#   docker run --rm -p 8000:8000 --env-file retrieval/.env asma-retrieval:latest
# =============================================================================

FROM python:3.11-slim

WORKDIR /app

# libpq5 is the only system shared lib needed at runtime (psycopg2-binary ships
# its own, but keeping libpq5 is safe for the PGVector path). build-essential +
# libpq-dev are needed only to build any sdist wheels, then purged.
COPY retrieval/requirements.txt ./retrieval/requirements.txt
RUN set -eux; \
    BUILD_DEPS="build-essential libpq-dev"; \
    apt-get update; \
    apt-get install -y --no-install-recommends $BUILD_DEPS libpq5 ca-certificates; \
    pip install --no-cache-dir -r retrieval/requirements.txt; \
    # Strip caches / bytecode to keep the image small.
    find /usr/local/lib/python3.11 -type d -name "__pycache__" -prune -exec rm -rf {} + || true; \
    find /usr/local/lib/python3.11 -name "*.pyc" -delete || true; \
    apt-get purge -y --auto-remove $BUILD_DEPS; \
    rm -rf /var/lib/apt/lists/*

# Application code — the self-contained retrieval package only.
#   retrieval/            the API, router, tool store
#   retrieval/_vendor/    vendored copies of the shared models/services it needs
#   retrieval/sources/    vendored database/api/cloud_storage tool connectors
COPY retrieval/ ./retrieval/


# Runtime env.
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    # Persist saved tool-calling configs on a writable path. Mount a volume
    # here in production so saved tools survive restarts.
    TOOL_STORE_PATH=/app/data/tool_configs.json \
    # Default server port (overridable).
    PORT=8000

# Non-root user for safety; owns the writable data dir.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/data \
    && chown -R appuser:appuser /app/data
USER appuser

EXPOSE 8000

# Container healthcheck hits the liveness probe the app already exposes.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os,urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:'+os.getenv('PORT','8000')+'/retrieve/health', timeout=4).status==200 else 1)"

# `retrieval:app` resolves to retrieval/__init__.py -> retrieval_new.app.
# Use the shell form so $PORT expands. One worker by default; scale with
# replicas (k8s) rather than in-process workers so the tool_store file lock
# stays simple.
CMD ["sh", "-c", "uvicorn retrieval:app --host 0.0.0.0 --port ${PORT:-8000} --workers ${UVICORN_WORKERS:-1}"]
