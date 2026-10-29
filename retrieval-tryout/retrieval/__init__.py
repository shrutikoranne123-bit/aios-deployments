"""Retrieval package: multi-backend RAG retrieval + query routing.

Re-exports the FastAPI app factory and module-level app so the service can be
launched as either:

    uvicorn retrieval:app
    uvicorn retrieval.retrieval_new:create_app --factory
"""

from .retrieval_new import app, create_app

__all__ = ["app", "create_app"]
