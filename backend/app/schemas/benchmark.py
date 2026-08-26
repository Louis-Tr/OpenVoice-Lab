"""Reproducible benchmark contracts."""

from datetime import datetime
from typing import Literal

from pydantic import Field

from app.schemas.base import ApiSchema
from app.schemas.synthesis import ModelVariant, SynthesisMetrics


class BenchmarkCase(ApiSchema):
    """One immutable evaluation input from the versioned corpus."""

    id: str = Field(min_length=1)
    category: str = Field(min_length=1)
    text: str = Field(min_length=1, max_length=5_000)


class BenchmarkCorpus(ApiSchema):
    """Versioned collection of evaluation inputs."""

    version: str = Field(min_length=1)
    cases: list[BenchmarkCase] = Field(min_length=1)


class BenchmarkRequest(ApiSchema):
    """Select model configurations and one shared voice for a benchmark run."""

    model_ids: list[str] | None = Field(default=None, min_length=1)
    voice_id: str = Field(default="af_heart", min_length=1)


class BenchmarkCaseResult(ApiSchema):
    """Raw outcome for one model and one corpus case."""

    case_id: str
    category: str
    text: str
    model_id: str
    precision: Literal["FP32", "INT8"]
    model_variant: ModelVariant
    voice_id: str
    status: Literal["success", "failure"]
    audio_url: str | None = None
    metrics: SynthesisMetrics | None = None
    error_type: str | None = None
    error_message: str | None = None


class BenchmarkAggregate(ApiSchema):
    """Comparable statistics for one model configuration."""

    model_id: str
    name: str
    precision: Literal["FP32", "INT8"]
    model_variant: ModelVariant
    total_cases: int = Field(ge=0)
    success_count: int = Field(ge=0)
    failure_count: int = Field(ge=0)
    average_latency_ms: float | None = Field(default=None, ge=0)
    median_latency_ms: float | None = Field(default=None, ge=0)
    p95_latency_ms: float | None = Field(default=None, ge=0)
    average_real_time_factor: float | None = Field(default=None, ge=0)
    average_memory_mb: float | None = Field(default=None, ge=0)
    peak_memory_mb: float | None = Field(default=None, ge=0)


class BenchmarkEnvironment(ApiSchema):
    """Execution context required to interpret a benchmark result."""

    python_version: str
    platform: str
    processor: str
    logical_cpu_count: int | None
    model_process_isolation: bool


class BenchmarkResult(ApiSchema):
    """Raw and aggregate evidence from one complete benchmark."""

    benchmark_id: str
    status: Literal["completed", "completed_with_failures"]
    started_at: datetime
    completed_at: datetime
    corpus_version: str
    corpus_sha256: str
    voice_id: str
    model_ids: list[str]
    environment: BenchmarkEnvironment
    raw_results: list[BenchmarkCaseResult]
    aggregates: list[BenchmarkAggregate]
    result_file: str | None = None
