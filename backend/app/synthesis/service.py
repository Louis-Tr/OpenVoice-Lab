"""Application-level owner of the synthesis workflow."""

from app.schemas.synthesis import SynthesisRequest, SynthesisResult


class SynthesisService:
    """Own synthesis orchestration outside the HTTP layer.

    Stage 1 returns a deterministic contract result. A later stage will inject
    the replaceable inference adapter here without changing the controller.
    """

    async def synthesize(self, request: SynthesisRequest) -> SynthesisResult:
        """Return the stable Stage 1 mock for a validated request."""
        return SynthesisResult(
            model=f"{request.model_id}-{request.variant}",
            text=request.text,
        )
