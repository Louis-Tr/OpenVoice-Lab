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
    variant: ModelVariant = "fp32"


class SynthesisResult(ApiSchema):
    """Deterministic Stage 1 response without generated audio."""

    status: Literal["mock"] = "mock"
    model: str
    text: str
    audio_url: str | None = None
