"""Synthesis request and response contracts."""

from typing import Literal

from pydantic import BaseModel, Field

ModelVariant = Literal["fp32", "quantized"]


class SynthesisRequest(BaseModel):
    """Technology-neutral synthesis request."""

    text: str = Field(min_length=1, max_length=5_000)
    model_id: str = Field(min_length=1)
    voice_id: str = Field(min_length=1)
    variant: ModelVariant = "fp32"


class InferenceMetrics(BaseModel):
    """Metrics returned with a completed synthesis."""

    latency_ms: float = Field(ge=0)
    real_time_factor: float = Field(ge=0)
    memory_mb: float = Field(ge=0)
    cold_start: bool
    model_id: str
    variant: ModelVariant


class SynthesisResult(BaseModel):
    """Technology-neutral synthesis response."""

    audio_url: str
    duration_seconds: float = Field(ge=0)
    metrics: InferenceMetrics

