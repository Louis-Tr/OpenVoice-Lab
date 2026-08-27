"""FastAPI application composition root."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api import benchmarks, experiments, health, models, synthesis
from app.api.errors import register_error_handlers
from app.audio.service import AudioService
from app.benchmark.service import BenchmarkJobService
from app.config.settings import Settings
from app.experiments.common import ExperimentEvidenceError
from app.experiments.service import ExperimentService, create_experiment_service
from app.health.service import HealthService
from app.metrics.collector import MetricsCollector
from app.models.loader import ModelLoader
from app.models.registry import ModelDefinition, ModelRegistry
from app.synthesis.service import SynthesisService
from app.text_processing.service import TextProcessingService

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
    metrics_collector: MetricsCollector | None = None,
    benchmark_job_service: BenchmarkJobService | None = None,
    text_processing_service: TextProcessingService | None = None,
    experiment_service: ExperimentService | None = None,
) -> FastAPI:
    """Create the API and compose controllers with application services."""
    resolved_settings = settings or Settings()
    health_service = HealthService()
    resolved_registry = model_registry or ModelRegistry(
        (
            ModelDefinition(
                model_id=resolved_settings.default_model_id,
                display_name=resolved_settings.default_model_display_name,
                precision="FP32",
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
            ModelDefinition(
                model_id=resolved_settings.quantized_model_id,
                display_name=resolved_settings.default_model_display_name,
                precision="INT8",
                variant="quantized",
                model_version=resolved_settings.kokoro_model_version,
                model_path=resolve_backend_path(resolved_settings.model_artifact_dir)
                / resolved_settings.kokoro_quantized_model_filename,
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
    resolved_metrics = metrics_collector or MetricsCollector()
    resolved_text_processing = text_processing_service or TextProcessingService()
    synthesis_service = SynthesisService(
        resolved_registry,
        resolved_loader,
        resolved_audio,
        resolved_metrics,
        resolved_text_processing,
    )
    resolved_benchmark_jobs = benchmark_job_service or BenchmarkJobService(
        resolved_registry,
        result_dir=resolve_backend_path(resolved_settings.benchmark_result_dir),
        default_voice_id=resolved_settings.default_voice_id,
    )
    stage12_root = resolve_backend_path(resolved_settings.stage12_artifact_root).resolve()
    stage12_root.mkdir(parents=True, exist_ok=True)
    resolved_experiments = experiment_service
    if resolved_experiments is None:
        try:
            resolved_experiments = create_experiment_service(
                artifact_root=resolve_backend_path(
                    resolved_settings.stage11_artifact_root
                ).resolve(),
                manifest_root=resolve_backend_path(
                    resolved_settings.stage11_manifest_root
                ).resolve(),
                stage12_root=stage12_root,
                model_cache_root=resolve_backend_path(
                    resolved_settings.experiment_model_cache_dir
                ).resolve(),
                speaker_profile_root=resolve_backend_path(
                    resolved_settings.experiment_speaker_profile_dir
                ).resolve(),
                text_processing=resolved_text_processing,
                audio_url_prefix=resolved_settings.experiment_audio_url_prefix,
                tts_revision=resolved_settings.speecht5_revision,
                vocoder_revision=resolved_settings.speecht5_vocoder_revision,
                maximum_queued_jobs=resolved_settings.experiment_maximum_queued_jobs,
                maximum_cached_models=resolved_settings.experiment_maximum_cached_models,
                cpu_threads=resolved_settings.experiment_cpu_threads,
            )
        except ExperimentEvidenceError:
            # Test/partial deployments may intentionally omit the ignored Stage 11 artifacts.
            # Existing synthesis and benchmark routes must remain available in that state.
            resolved_experiments = None

    application = FastAPI(
        title="OpenVoice Lab API",
        version="0.12.0",
        description=(
            "Measured self-hosted Kokoro ONNX variants with deterministic "
            "text processing and benchmarks, plus artifact-backed SpeechT5 "
            "training evidence and local CPU comparisons."
        ),
    )
    application.include_router(synthesis.create_router(synthesis_service), prefix="/api")
    application.include_router(models.create_router(resolved_registry), prefix="/api")
    application.include_router(
        benchmarks.create_router(resolved_benchmark_jobs),
        prefix="/api",
    )
    application.include_router(health.create_router(health_service))
    if resolved_experiments is not None:
        application.include_router(
            experiments.create_router(resolved_experiments),
            prefix="/api",
        )
        application.router.add_event_handler("startup", resolved_experiments.recover)
    application.mount(
        resolved_settings.audio_url_prefix,
        StaticFiles(directory=resolved_audio.output_dir),
        name="generated-audio",
    )
    experiment_audio_root = stage12_root / "comparisons"
    experiment_audio_root.mkdir(parents=True, exist_ok=True)
    application.mount(
        resolved_settings.experiment_audio_url_prefix,
        StaticFiles(directory=experiment_audio_root),
        name="experiment-audio",
    )
    register_error_handlers(application)
    application.state.model_loader = resolved_loader
    application.state.model_registry = resolved_registry
    application.state.metrics_collector = resolved_metrics
    application.state.synthesis_service = synthesis_service
    application.state.text_processing_service = resolved_text_processing
    application.state.benchmark_job_service = resolved_benchmark_jobs
    application.state.experiment_service = resolved_experiments
    return application


app = create_app()
