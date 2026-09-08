"""Thin HTTP controller for synthesis requests."""

from typing import Annotated

from fastapi import APIRouter, Header, Request, Response, status

from app.schemas.synthesis import SynthesisRequest, SynthesisResult
from app.schemas.synthesis_job import SynthesisJob
from app.synthesis.job_store import TERMINAL_STATUSES
from app.synthesis.jobs import SynthesisJobService
from app.synthesis.service import SynthesisService


def create_router(service: SynthesisService, jobs: SynthesisJobService) -> APIRouter:
    """Bind the HTTP contract to its application service."""
    router = APIRouter(tags=["synthesis"])

    @router.post(
        "/synthesis",
        response_model=SynthesisResult,
        status_code=status.HTTP_200_OK,
    )
    async def create_synthesis(request: SynthesisRequest) -> SynthesisResult:
        return await service.synthesize(request)

    @router.post("/synthesis/jobs", response_model=SynthesisJob, status_code=202)
    async def create_synthesis_job(
        request: SynthesisRequest,
        http_request: Request,
        response: Response,
        idempotency_key: Annotated[
            str, Header(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._:-]+$")
        ],
    ) -> SynthesisJob:
        job = await jobs.start(request, idempotency_key)
        response.status_code = 200 if job.status in TERMINAL_STATUSES else 202
        response.headers["Location"] = str(
            http_request.url_for("get_synthesis_job", job_id=job.job_id)
        )
        response.headers["Cache-Control"] = "no-store"
        if job.status not in TERMINAL_STATUSES:
            response.headers["Retry-After"] = "2"
        return job

    @router.get("/synthesis/jobs/{job_id}", response_model=SynthesisJob)
    async def get_synthesis_job(job_id: str, response: Response) -> SynthesisJob:
        response.headers["Cache-Control"] = "no-store"
        job = jobs.get(job_id)
        if job.status not in TERMINAL_STATUSES:
            response.headers["Retry-After"] = "2"
        return job

    return router
