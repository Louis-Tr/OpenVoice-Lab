"""Thin HTTP controller for synthesis requests."""

from fastapi import APIRouter, Request, status

from app.schemas.synthesis import SynthesisRequest, SynthesisResult
from app.synthesis.jobs import SynthesisJobService


def create_router(service: SynthesisJobService) -> APIRouter:
    """Bind the HTTP contract to its application service."""
    router = APIRouter(tags=["synthesis"])

    @router.post(
        "/synthesis",
        response_model=SynthesisResult,
        status_code=status.HTTP_200_OK,
    )
    async def create_synthesis(
        request: SynthesisRequest,
        http_request: Request,
    ) -> SynthesisResult:
        return await service.submit_and_wait(request, http_request.is_disconnected)

    return router
