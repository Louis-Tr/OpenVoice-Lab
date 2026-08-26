"""Health endpoint contracts."""

from typing import Literal

from app.schemas.base import ApiSchema


class HealthResponse(ApiSchema):
    """Process liveness response."""

    status: Literal["healthy"] = "healthy"
