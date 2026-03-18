"""reviewmind/api/router.py — Main router connecting all endpoints."""

from fastapi import APIRouter

from reviewmind.api.endpoints.feedback import router as feedback_router
from reviewmind.api.endpoints.health import router as health_router
from reviewmind.api.endpoints.ingest import router as ingest_router
from reviewmind.api.endpoints.query import router as query_router
from reviewmind.api.endpoints.status import router as status_router

api_router = APIRouter()
api_router.include_router(health_router, tags=["health"])
api_router.include_router(query_router, tags=["query"])
api_router.include_router(ingest_router, tags=["ingest"])
api_router.include_router(status_router, tags=["status"])
api_router.include_router(feedback_router, tags=["feedback"])
