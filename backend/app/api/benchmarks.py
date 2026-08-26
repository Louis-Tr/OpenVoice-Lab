"""HTTP contract for benchmark requests."""

from fastapi import APIRouter, HTTPException, status

from app.schemas.benchmark import BenchmarkRequest, BenchmarkResult

router = APIRouter(tags=["benchmarks"])


@router.post(
    "/benchmarks",
    response_model=BenchmarkResult,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_benchmark(_request: BenchmarkRequest) -> BenchmarkResult:
    """Reserve the transport contract while HTTP job orchestration is deferred."""
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Benchmark HTTP job orchestration is not implemented yet.",
    )
