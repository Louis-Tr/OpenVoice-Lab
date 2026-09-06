"""Asynchronous synthesis job HTTP controller."""

from fastapi import APIRouter, Header, Request, status

from app.schemas.jobs import SynthesisJobStatus
from app.schemas.synthesis import SynthesisRequest
from app.synthesis.jobs import SynthesisJobService


def create_router(service: SynthesisJobService) -> APIRouter:
    router = APIRouter(prefix="/synthesis/jobs", tags=["synthesis jobs"])

    @router.post("", response_model=SynthesisJobStatus, status_code=status.HTTP_202_ACCEPTED)
    async def submit_job(
        request: SynthesisRequest,
        http_request: Request,
        idempotency_key: str | None = Header(default=None, max_length=128),
    ) -> SynthesisJobStatus:
        client = http_request.client.host if http_request.client is not None else "unknown"
        return await service.submit(
            request,
            client_scope=client,
            idempotency_key=idempotency_key,
        )

    @router.get("/{job_id}", response_model=SynthesisJobStatus)
    async def get_job(job_id: str) -> SynthesisJobStatus:
        return service.get(job_id)

    @router.post("/{job_id}/cancel", response_model=SynthesisJobStatus)
    async def cancel_job(job_id: str) -> SynthesisJobStatus:
        return await service.cancel(job_id)

    return router
