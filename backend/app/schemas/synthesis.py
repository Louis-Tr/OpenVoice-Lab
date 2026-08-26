"""Synthesis request and response contracts."""

from typing import Literal

from pydantic import Field

from app.schemas.base import ApiSchema

ModelVariant = Literal["fp32", "quantized"]


class SynthesisRequest(ApiSchema):
    """Technology-neutral synthesis request."""

    text: str = Field(min_length=1, max_length=5_000)
    model_id: str = Field(min_length=1)
    voice_id: str = Field(min_length=1)
    sanitize_text: bool = True
    normalize_text: bool = True


class SynthesisMetrics(ApiSchema):
    """Model-level measurements for one successful inference operation."""

    model_load_ms: float = Field(ge=0)
    inference_ms: float = Field(ge=0)
    audio_duration_ms: float = Field(gt=0)
    real_time_factor: float = Field(ge=0)
    memory_mb: float = Field(ge=0)
    warm: bool
    model_variant: ModelVariant


class SynthesisResult(ApiSchema):
    """Stable synthesis response with independent performance measurements."""

    status: Literal["mock", "ok"]
    model: str
    text: str
    normalized_text: str
    audio_url: str | None = None
    metrics: SynthesisMetrics
