# Retrieval API — Deployment Guide

This directory contains everything needed to deploy the **retrieval service**
(`retrieval_new.py` + `query_router.py` + `tool_store.py`) as a standalone,
long-running HTTP API — separate from the heavy ingestion pipeline.

## What gets deployed

A FastAPI app (`retrieval:app`) exposing:

| Endpoint | Purpose |
|---|---|
| `GET /retrieve/health` | Liveness/readiness probe |
| `POST /retrieve` | Retrieve chunks + generate an answer |
| `POST /retrieve/smart` | Auto-route a query to RAG / external tool, then answer |
| `POST /tools`, `GET /tools`, `GET/PUT/DELETE /tools/{id}` | Saved tool-calling configs (org-scoped) |
| `POST /retrieve/mongo-debug` | MongoDB index diagnostics |
| `GET /docs` | Swagger UI |

It supports all vector-store backends (Qdrant, Pinecone, Redis, PGVector,
MongoDB) and the tool-calling kinds `http`, `database`, `api`, `cloud_storage`.

## Files

| File | Role |
|---|---|
| `requirements.txt` | Lean, pinned runtime deps (no torch/docling) |
| `Dockerfile` | Slim image; runs uvicorn. **Build from repo root.** |
| `.dockerignore` | Keeps the image small and secret-free |
| `.env.example` | Documented env vars — copy to `.env` |
| `docker-compose.yml` | Local / single-host run |
| `k8s/` | Kubernetes manifests (secret, deployment, service, hpa, ingress) + `kustomization.yaml` |

## Important: build context is the repository root

The retrieval code imports the top-level `services/` package (for the
`database`/`api`/`cloud_storage` tool connectors). The image must include both
`retrieval/` and `services/`, so **always build from the repo root** with the
Dockerfile path flag:

```bash
docker build -f retrieval/Dockerfile -t asma-retrieval:latest .
```

## Option A — Docker Compose (local / single host)

```bash
cd retrieval
cp .env.example .env          # set AWS_REGION and Bedrock model ids
docker compose up --build
```

Then:
- API:     http://localhost:8000
- Swagger: http://localhost:8000/docs
- Health:  http://localhost:8000/retrieve/health

Saved tool configs persist in the `retrieval-tool-store` named volume.

## Option B — Plain Docker

```bash
# from repo root
docker build -f retrieval/Dockerfile -t asma-retrieval:latest .
docker run --rm -p 8000:8000 \
  --env-file retrieval/.env \
  -v asma-retrieval-data:/app/data \
  asma-retrieval:latest
```

## Option C — Kubernetes

1. Build and push the image (from the repo root):
   ```bash
   IMG=410133709370.dkr.ecr.ap-south-1.amazonaws.com/asmaaios-rag-retrieval:v1
   docker build -f retrieval/Dockerfile -t "$IMG" .
   docker push "$IMG"
   ```
2. Review the manifests under `k8s/` — set the image tag (`k8s/deployment.yaml`
   or the `images:` block in `k8s/kustomization.yaml`), namespace, and the
   `rag-retrieval-env` Secret values (`k8s/secret.yaml`, or create the Secret
   separately). Prefer binding an **IRSA role** to the pod ServiceAccount for
   Bedrock access instead of static AWS keys.
3. Apply the whole folder with kustomize:
   ```bash
   kubectl apply -k k8s
   ```
   Or apply individual files:
   ```bash
   kubectl apply -f k8s/secret.yaml \
     -f k8s/deployment.yaml -f k8s/service.yaml -f k8s/hpa.yaml
   ```
4. Verify:
   ```bash
   kubectl -n dev rollout status deploy/rag-retrieval
   kubectl -n dev port-forward svc/rag-retrieval 8000:80
   curl http://localhost:8000/retrieve/health
   ```

## Configuration (env vars)

See `.env.example` for the full list. Highlights:

- `AWS_REGION` + Bedrock model ids — required (embeddings + answer generation).
  Credentials come from the standard boto3 chain (IAM role preferred).
- `TOOL_STORE_PATH` — where saved `/tools` configs persist. Back it with a
  volume so saved tools survive restarts. **Credentials in this file are stored
  in plaintext** — secure it with filesystem permissions / a mounted secret.
- `PORT` (default 8000), `UVICORN_WORKERS` (default 1 — scale with replicas).

> **Vector-store credentials** (Qdrant/Pinecone/Redis/PG/Mongo) are sent
> per request in the `vector_store_details` body field, so they are **not**
> part of the deployment config.

## Optional tool-kind extras

The base image is lean. If you use these tool kinds, enable the matching extras
in `requirements.txt` and rebuild:

- `kind: "api"` tools → uncomment `transformers` (used by
  `services.tokenizer_service` during API-source fetches). Adds ~hundreds of MB.
- `kind: "cloud_storage"` tools → uncomment `google-cloud-storage` and/or
  `azure-storage-blob` (S3 already works via boto3).

## Scaling & concurrency notes

- Scale horizontally with `replicas` / the HPA, not with in-process
  `UVICORN_WORKERS`. The tool-store uses a cross-process file lock, which is
  simplest with one writer per pod.
- In Kubernetes the tool-store uses an ephemeral `emptyDir`, so saved `/tools`
  configs do **not** survive pod restarts. If you rely on saved tools, add a
  PVC (RWO for a single writer, or RWX like EFS if multiple replicas must share
  them) and point the `tool-store` volume in `k8s/deployment.yaml` at it.
