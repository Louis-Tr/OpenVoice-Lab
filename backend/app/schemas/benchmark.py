"""Benchmark request and response contracts."""

from pydantic import BaseModel, Field

from app.schemas.synthesis import ModelVariant


class BenchmarkRequest(BaseModel):
    """Request a run of the predefined benchmark corpus."""

    model_id: str = Field(min_length=1)
    variant: ModelVariant = "fp32"


class BenchmarkResult(BaseModel):
    """Aggregate result placeholder for a benchmark run."""

    benchmark_id: str
    status: str
    aggregates: dict[str, float]

