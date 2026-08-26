"""Thin HTTP controller for synthesis requests."""

from fastapi import APIRouter, status

from app.schemas.synthesis import SynthesisRequest, SynthesisResult
from app.synthesis.service import SynthesisService


def create_router(service: SynthesisService) -> APIRouter:
    """Bind the HTTP contract to its application service."""
    router = APIRouter(tags=["synthesis"])

    @router.post(
        "/synthesis",
        response_model=SynthesisResult,
        status_code=status.HTTP_200_OK,
    )
    async def create_synthesis(request: SynthesisRequest) -> SynthesisResult:
        return await service.synthesize(request)

    return router
