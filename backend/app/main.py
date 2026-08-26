"""FastAPI application composition root."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api import benchmarks, health, models, synthesis
from app.api.errors import register_error_handlers
from app.audio.service import AudioService
from app.config.settings import Settings
from app.health.service import HealthService
from app.models.loader import ModelLoader
from app.models.registry import ModelDefinition, ModelRegistry
from app.synthesis.service import SynthesisService

BACKEND_ROOT = Path(__file__).resolve().parents[1]


def resolve_backend_path(path: Path) -> Path:
    """Resolve configured relative paths consistently from the backend root."""
    return path if path.is_absolute() else BACKEND_ROOT / path


def create_app(
    settings: Settings | None = None,
    *,
    model_registry: ModelRegistry | None = None,
    model_loader: ModelLoader | None = None,
    audio_service: AudioService | None = None,
) -> FastAPI:
    """Create the API and compose controllers with application services."""
    resolved_settings = settings or Settings()
    health_service = HealthService()
    resolved_registry = model_registry or ModelRegistry(
        (
            ModelDefinition(
                model_id=resolved_settings.default_model_id,
                display_name=resolved_settings.default_model_display_name,
                variant="fp32",
                model_version=resolved_settings.kokoro_model_version,
                model_path=resolve_backend_path(resolved_settings.model_artifact_dir)
                / resolved_settings.kokoro_model_filename,
                voices_path=resolve_backend_path(resolved_settings.model_artifact_dir)
                / resolved_settings.kokoro_voices_filename,
                voices=(resolved_settings.default_voice_id,),
                language=resolved_settings.kokoro_language,
                speed=resolved_settings.kokoro_speed,
            ),
        )
    )
    resolved_loader = model_loader or ModelLoader()
    resolved_audio = audio_service or AudioService(
        resolve_backend_path(resolved_settings.generated_audio_dir),
        resolved_settings.audio_url_prefix,
    )
    synthesis_service = SynthesisService(
        resolved_registry,
        resolved_loader,
        resolved_audio,
    )

    application = FastAPI(
        title="OpenVoice Lab API",
        version="0.2.0",
        description="Self-hosted Kokoro ONNX synthesis behind a replaceable inference boundary.",
    )
    application.include_router(synthesis.create_router(synthesis_service), prefix="/api")
    application.include_router(models.create_router(resolved_registry), prefix="/api")
    application.include_router(benchmarks.router, prefix="/api")
    application.include_router(health.create_router(health_service))
    application.mount(
        resolved_settings.audio_url_prefix,
        StaticFiles(directory=resolved_audio.output_dir),
        name="generated-audio",
    )
    register_error_handlers(application)
    application.state.model_loader = resolved_loader
    application.state.model_registry = resolved_registry
    return application


app = create_app()
