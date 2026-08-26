"""Liveness and future readiness HTTP support."""

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict[str, str]:
    """Report process liveness; dependency readiness checks will be added later."""
    return {"status": "ok"}

