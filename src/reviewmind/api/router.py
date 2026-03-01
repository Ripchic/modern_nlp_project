"""reviewmind/api/router.py — Main router connecting all endpoints."""

from fastapi import APIRouter

from reviewmind.api.endpoints.health import router as health_router

api_router = APIRouter()
api_router.include_router(health_router, tags=["health"])
