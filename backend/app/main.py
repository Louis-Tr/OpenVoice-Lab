"""FastAPI application composition root."""

from fastapi import FastAPI

from app.api import benchmarks, health, models, synthesis
from app.config.settings import Settings
from app.health.service import HealthService
from app.models.registry import ModelRegistry
from app.schemas.model import ModelSummary
from app.synthesis.service import SynthesisService


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create the API and compose controllers with application services."""
    resolved_settings = settings or Settings()
    health_service = HealthService()
    synthesis_service = SynthesisService()
    model_registry = ModelRegistry(
        models=(
            ModelSummary(
                id=resolved_settings.default_model_id,
                display_name=resolved_settings.default_model_display_name,
                voices=[resolved_settings.default_voice_id],
                variants=["fp32", "quantized"],
            ),
        )
    )

    application = FastAPI(
        title="OpenVoice Lab API",
        version="0.1.0",
        description="Executable Stage 1 API contract with deterministic synthesis output.",
    )
    application.include_router(synthesis.create_router(synthesis_service), prefix="/api")
    application.include_router(models.create_router(model_registry), prefix="/api")
    application.include_router(benchmarks.router, prefix="/api")
    application.include_router(health.create_router(health_service))
    return application


app = create_app()
