"""Thin HTTP controllers for browser-triggered benchmark jobs."""

from fastapi import APIRouter, status

from app.benchmark.service import BenchmarkJobService
from app.schemas.benchmark import BenchmarkConfig, BenchmarkJobStatus, BenchmarkRequest


def create_router(service: BenchmarkJobService) -> APIRouter:
    """Bind benchmark transport contracts to the job application service."""
    router = APIRouter(tags=["benchmarks"])

    @router.get("/benchmarks/config", response_model=BenchmarkConfig)
    async def get_benchmark_config() -> BenchmarkConfig:
        return service.describe()

    @router.get("/benchmarks/latest", response_model=BenchmarkJobStatus)
    async def get_latest_benchmark() -> BenchmarkJobStatus:
        return service.latest()

    @router.post(
        "/benchmarks",
        response_model=BenchmarkJobStatus,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def create_benchmark(request: BenchmarkRequest) -> BenchmarkJobStatus:
        return await service.start(request)

    @router.get("/benchmarks/{benchmark_id}", response_model=BenchmarkJobStatus)
    async def get_benchmark(benchmark_id: str) -> BenchmarkJobStatus:
        return service.get(benchmark_id)

    return router
