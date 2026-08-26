"""Thin HTTP controller for model discovery."""

from fastapi import APIRouter

from app.models.registry import ModelRegistry
from app.schemas.model import ModelSummary


def create_router(registry: ModelRegistry) -> APIRouter:
    """Bind the model-listing contract to the registry service."""
    router = APIRouter(tags=["models"])

    @router.get("/models", response_model=list[ModelSummary])
    async def list_models() -> list[ModelSummary]:
        return registry.list_available()

    return router
