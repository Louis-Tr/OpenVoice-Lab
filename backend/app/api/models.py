"""HTTP contract for model discovery."""

from fastapi import APIRouter

from app.schemas.model import ModelSummary

router = APIRouter(tags=["models"])


@router.get("/models", response_model=list[ModelSummary])
async def list_models() -> list[ModelSummary]:
    """Return no models until a registry source is configured."""
    return []

