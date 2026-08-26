"""Application service for process health."""

from app.schemas.health import HealthResponse


class HealthService:
    """Report liveness independently of HTTP transport."""

    def status(self) -> HealthResponse:
        """Return the Stage 1 process health state."""
        return HealthResponse()
