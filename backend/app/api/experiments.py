"""Thin HTTP controllers for Stage 11 evidence and live comparisons."""

import asyncio

from fastapi import APIRouter, Query, status
from fastapi.responses import StreamingResponse

from app.experiments.service import ExperimentService
from app.experiments.store import TERMINAL_STAGES
from app.schemas.experiment import (
    ExperimentComparisonJob,
    ExperimentComparisonRequest,
    ExperimentFixturePage,
    ExperimentModelSummary,
    ExperimentReport,
)


def create_router(service: ExperimentService) -> APIRouter:
    router = APIRouter(prefix="/experiments/stage11", tags=["stage11-experiment"])

    @router.get("/report", response_model=ExperimentReport)
    async def get_report() -> ExperimentReport:
        return service.report()

    @router.get("/fixtures", response_model=ExperimentFixturePage)
    async def get_fixtures(
        query: str | None = Query(default=None, max_length=100),
        term: str | None = Query(default=None, max_length=80),
        category: str | None = Query(default=None, max_length=80),
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=30, ge=1, le=100),
    ) -> ExperimentFixturePage:
        return service.fixtures(
            query=query, term=term, category=category, offset=offset, limit=limit
        )

    @router.get("/models", response_model=list[ExperimentModelSummary])
    async def get_models() -> list[ExperimentModelSummary]:
        return service.models()

    @router.post(
        "/comparisons",
        response_model=ExperimentComparisonJob,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def start_comparison(
        request: ExperimentComparisonRequest,
    ) -> ExperimentComparisonJob:
        return await service.start(request)

    @router.get("/comparisons/{job_id}", response_model=ExperimentComparisonJob)
    async def get_comparison(job_id: str) -> ExperimentComparisonJob:
        return service.get(job_id)

    @router.delete("/comparisons/{job_id}", response_model=ExperimentComparisonJob)
    async def cancel_comparison(job_id: str) -> ExperimentComparisonJob:
        return service.cancel(job_id)

    @router.get("/comparisons/{job_id}/events")
    async def comparison_events(job_id: str) -> StreamingResponse:
        service.get(job_id)

        async def stream():
            previous = ""
            while True:
                job = service.get(job_id)
                payload = job.model_dump_json(by_alias=True)
                if payload != previous:
                    yield f"data: {payload}\n\n"
                    previous = payload
                if job.stage in TERMINAL_STAGES:
                    break
                await asyncio.sleep(0.5)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return router
