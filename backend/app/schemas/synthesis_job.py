"""Pollable synthesis job snapshots; completed results retain the existing schema."""

from datetime import datetime
from typing import Literal

from app.schemas.base import ApiSchema
from app.schemas.synthesis import SynthesisRequest, SynthesisResult

JobStatus = Literal["pending", "loading", "generating", "saving", "completed", "failed", "rejected"]


class SynthesisJobError(ApiSchema):
    status_code: int
    detail: str


class SynthesisJob(ApiSchema):
    job_id: str
    status: JobStatus = "pending"
    request: SynthesisRequest
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None
    expires_at: datetime | None = None
    result: SynthesisResult | None = None
    error: SynthesisJobError | None = None
