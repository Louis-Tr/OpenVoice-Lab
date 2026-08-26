"""Liveness and future readiness HTTP support."""

from fastapi import APIRouter

from app.health.service import HealthService
from app.schemas.health import HealthResponse


def create_router(service: HealthService) -> APIRouter:
    """Bind the health contract to its application service."""
    router = APIRouter(tags=["health"])

    @router.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return service.status()

    return router
