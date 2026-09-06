"""Aggregate scheduler diagnostics without queued request content."""

from fastapi import APIRouter

from app.scheduling.service import ProcessingScheduler


def create_router(scheduler: ProcessingScheduler, *, git_sha: str) -> APIRouter:
    router = APIRouter(tags=["resources"])

    @router.get("/resources")
    async def resource_status() -> dict[str, object]:
        return {**scheduler.status(), "gitSha": git_sha}

    return router
