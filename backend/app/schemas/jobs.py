"""Public synthesis job contracts."""

from datetime import datetime
from typing import Literal

from app.schemas.base import ApiSchema
from app.schemas.synthesis import SynthesisRequest, SynthesisResult

SynthesisJobState = Literal[
    "queued",
    "reserved",
    "loading",
    "running",
    "saving",
    "completed",
    "failed",
    "cancelled",
]


class SynthesisJobError(ApiSchema):
    code: str
    message: str


class SynthesisJobStatus(ApiSchema):
    id: str
    state: SynthesisJobState
    request: SynthesisRequest
    normalized_text: str
    queued_at: datetime
    updated_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    queue_wait_ms: float | None = None
    processing_ms: float | None = None
    waiting_reason: str | None = None
    cancellation_requested: bool = False
    result: SynthesisResult | None = None
    error: SynthesisJobError | None = None
