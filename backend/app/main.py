"""FastAPI application composition root."""

from importlib.util import find_spec
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api import benchmarks, experiments, health, models, synthesis
from app.api.errors import register_error_handlers
from app.audio.service import AudioService
from app.benchmark.service import BenchmarkJobService
from app.config.settings import Settings
from app.experiments.cloud import create_cloud_experiment_service
from app.experiments.common import ExperimentEvidenceError
from app.experiments.service import ExperimentService, create_experiment_service
from app.health.service import HealthService
from app.inference.cpu import configure_torch_threads
from app.metrics.collector import MetricsCollector
from app.models.loader import ModelLoader
from app.models.registry import ModelDefinition, ModelRegistry
from app.models.resources import INT8_UNAVAILABLE, MEMORY_PROFILES, ResourceManager
from app.models.scheduler import EngineScheduler
from app.synthesis.job_store import InMemoryJobStore
from app.synthesis.jobs import SynthesisJobService
from app.synthesis.service import SynthesisService
from app.text_processing.service import TextProcessingService
from app.web import AngularStaticFiles

BACKEND_ROOT = Path(__file__).resolve().parents[1]


def resolve_backend_path(path: Path) -> Path:
    """Resolve configured relative paths consistently from the backend root."""
    return path if path.is_absolute() else BACKEND_ROOT / path


def dependencies_available(*module_names: str) -> bool:
    """Check optional runtime modules without importing heavyweight packages."""
    try:
        return all(find_spec(name) is not None for name in module_names)
    except (ImportError, ModuleNotFoundError):
        return False


def create_app(
    settings: Settings | None = None,
    *,
    model_registry: ModelRegistry | None = None,
    model_loader: ModelLoader | None = None,
    audio_service: AudioService | None = None,
    metrics_collector: MetricsCollector | None = None,
    resource_manager: ResourceManager | None = None,
    benchmark_job_service: BenchmarkJobService | None = None,
    text_processing_service: TextProcessingService | None = None,
    experiment_service: ExperimentService | None = None,
) -> FastAPI:
    """Create the API and compose controllers with application services."""
    resolved_settings = settings or Settings()
    health_service = HealthService()
    model_artifact_root = resolve_backend_path(resolved_settings.model_artifact_dir)
    speecht5_dependencies_ready = dependencies_available("torch", "transformers", "sentencepiece")
    resolved_registry = model_registry or ModelRegistry(
        (
            ModelDefinition(
                model_id=resolved_settings.default_model_id,
                display_name=resolved_settings.default_model_display_name,
                precision="FP32",
                variant="fp32",
                model_version=resolved_settings.kokoro_model_version,
                model_path=model_artifact_root / resolved_settings.kokoro_model_filename,
                voices_path=model_artifact_root / resolved_settings.kokoro_voices_filename,
                voices=(resolved_settings.default_voice_id,),
                language=resolved_settings.kokoro_language,
                speed=resolved_settings.kokoro_speed,
                runtime="ONNX Runtime",
                description="Highest-fidelity Kokoro baseline with full-precision weights.",
            ),
            ModelDefinition(
                model_id=resolved_settings.fp16_model_id,
                display_name=resolved_settings.default_model_display_name,
                precision="FP16",
                variant="fp16",
                model_version=resolved_settings.kokoro_model_version,
                model_path=(model_artifact_root / resolved_settings.kokoro_fp16_model_filename),
                voices_path=model_artifact_root / resolved_settings.kokoro_voices_filename,
                voices=(resolved_settings.default_voice_id,),
                language=resolved_settings.kokoro_language,
                speed=resolved_settings.kokoro_speed,
                runtime="ONNX Runtime",
                description="Half-precision Kokoro for lower memory with the same voice set.",
            ),
            ModelDefinition(
                model_id=resolved_settings.quantized_model_id,
                display_name=resolved_settings.default_model_display_name,
                precision="INT8",
                variant="quantized",
                model_version=resolved_settings.kokoro_model_version,
                model_path=(
                    model_artifact_root / resolved_settings.kokoro_quantized_model_filename
                ),
                voices_path=model_artifact_root / resolved_settings.kokoro_voices_filename,
                voices=(resolved_settings.default_voice_id,),
                language=resolved_settings.kokoro_language,
                speed=resolved_settings.kokoro_speed,
                runtime="ONNX Runtime",
                description="Weight-quantized Kokoro for the smallest local footprint.",
                enabled=False,
                unavailable_reason=INT8_UNAVAILABLE,
            ),
            ModelDefinition(
                model_id=resolved_settings.speecht5_model_id,
                display_name="SpeechT5",
                precision="FP32",
                variant="pretrained",
                model_version=resolved_settings.speecht5_revision,
                model_path=model_artifact_root / resolved_settings.speecht5_model_dirname,
                voices_path=model_artifact_root / resolved_settings.speecht5_speaker_filename,
                voices=(resolved_settings.speecht5_voice_id,),
                runtime="PyTorch CPU",
                engine="speecht5-transformers",
                max_input_characters=5_000,
                max_input_tokens=600,
                additional_artifacts=(
                    model_artifact_root / resolved_settings.speecht5_vocoder_dirname,
                ),
                availability_markers=(
                    model_artifact_root
                    / resolved_settings.speecht5_model_dirname
                    / "pytorch_model.bin",
                    model_artifact_root / resolved_settings.speecht5_model_dirname / "config.json",
                    model_artifact_root
                    / resolved_settings.speecht5_vocoder_dirname
                    / "pytorch_model.bin",
                    model_artifact_root
                    / resolved_settings.speecht5_vocoder_dirname
                    / "config.json",
                ),
                enabled=speecht5_dependencies_ready,
                unavailable_reason=(
                    None
                    if speecht5_dependencies_ready
                    else "SpeechT5 CPU requires the pinned local PyTorch serving dependencies."
                ),
                description=(
                    "Pinned Microsoft SpeechT5 CPU baseline with a verified CMU speaker profile."
                ),
            ),
        )
    )
    resolved_resources = resource_manager or ResourceManager(
        profiles={
            resolved_settings.default_model_id: MEMORY_PROFILES["kokoro-fp32"],
            resolved_settings.fp16_model_id: MEMORY_PROFILES["kokoro-fp16"],
            resolved_settings.speecht5_model_id: MEMORY_PROFILES["speecht5-pretrained"],
        },
        cpu_units=resolved_settings.product_cpu_units,
        cpu_threads=resolved_settings.product_cpu_threads,
        memory_limit_mb=resolved_settings.product_memory_limit_mb,
        memory_headroom_mb=resolved_settings.product_memory_headroom_mb,
    )
    resolved_loader = model_loader or ModelLoader(cpu_threads=resolved_resources.cpu_threads)
    resolved_audio = audio_service or AudioService(
        resolve_backend_path(resolved_settings.generated_audio_dir),
        resolved_settings.audio_url_prefix,
    )
    resolved_metrics = metrics_collector or MetricsCollector()
    scheduler = EngineScheduler(
        resolved_loader,
        resolved_resources,
        resolved_metrics,
        maximum_cached_engines=resolved_settings.product_maximum_cached_models,
    )
    resolved_text_processing = text_processing_service or TextProcessingService()
    synthesis_service = SynthesisService(
        resolved_registry,
        scheduler,
        resolved_audio,
        resolved_metrics,
        resolved_text_processing,
    )
    synthesis_jobs = SynthesisJobService(
        synthesis_service,
        InMemoryJobStore(
            maximum_jobs=resolved_settings.synthesis_job_maximum_records,
            retention_seconds=resolved_settings.synthesis_job_retention_seconds,
        ),
    )
    resolved_benchmark_jobs = benchmark_job_service or BenchmarkJobService(
        resolved_registry,
        result_dir=resolve_backend_path(resolved_settings.benchmark_result_dir),
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
                approach_run_root=resolve_backend_path(
                    resolved_settings.stage11_approach_run_root
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
            # Cloud source builds retain verified evidence and can lazily provision the
            # adapted models from a private, checksum-verified artifact origin.
            try:
                access_token = resolved_settings.experiment_model_access_token
                resolved_experiments = create_cloud_experiment_service(
                    model_base_url=resolved_settings.experiment_model_base_url,
                    model_access_token=(
                        access_token.get_secret_value() if access_token is not None else None
                    ),
                    remote_manifest_path=resolve_backend_path(
                        resolved_settings.experiment_remote_manifest_path
                    ).resolve(),
                    pretrained_root=(
                        model_artifact_root / resolved_settings.speecht5_model_dirname
                    ).resolve(),
                    vocoder_root=(
                        model_artifact_root / resolved_settings.speecht5_vocoder_dirname
                    ).resolve(),
                    asr_root=(
                        model_artifact_root / resolved_settings.experiment_asr_dirname
                    ).resolve(),
                    speaker_embedding_path=(
                        model_artifact_root / resolved_settings.speecht5_speaker_filename
                    ).resolve(),
                    stage12_root=stage12_root,
                    remote_cache_root=resolve_backend_path(
                        resolved_settings.experiment_model_cache_dir
                    ).resolve()
                    / "adapted",
                    text_processing=resolved_text_processing,
                    audio_url_prefix=resolved_settings.experiment_audio_url_prefix,
                    vocoder_revision=resolved_settings.speecht5_vocoder_revision,
                    maximum_queued_jobs=resolved_settings.experiment_maximum_queued_jobs,
                    maximum_cached_models=resolved_settings.experiment_maximum_cached_models,
                    cpu_threads=resolved_settings.experiment_cpu_threads,
                )
            except ExperimentEvidenceError:
                resolved_experiments = None

    application = FastAPI(
        title="OpenVoice Lab API",
        version="0.12.0",
        description=(
            "Capability-aware local TTS serving with Kokoro ONNX variants, "
            "SpeechT5, deterministic text processing, and measured experiments."
        ),
    )

    def initialize_cpu_runtime() -> None:
        if speecht5_dependencies_ready:
            import torch

            configure_torch_threads(torch, resolved_resources.cpu_threads)

    application.router.add_event_handler("startup", initialize_cpu_runtime)
    application.include_router(
        synthesis.create_router(synthesis_service, synthesis_jobs), prefix="/api"
    )
    application.router.add_event_handler("shutdown", synthesis_jobs.close)
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
    frontend_dist_dir = resolved_settings.frontend_dist_dir
    if frontend_dist_dir is not None:
        resolved_frontend = resolve_backend_path(frontend_dist_dir).resolve()
        if not (resolved_frontend / "index.html").is_file():
            raise RuntimeError(
                f"Configured frontend build is missing index.html: {resolved_frontend}"
            )
        application.mount(
            "/",
            AngularStaticFiles(directory=resolved_frontend, html=True),
            name="frontend",
        )
    register_error_handlers(application)
    application.state.model_loader = resolved_loader
    application.state.engine_scheduler = scheduler
    application.state.resource_manager = resolved_resources
    application.state.model_registry = resolved_registry
    application.state.metrics_collector = resolved_metrics
    application.state.synthesis_service = synthesis_service
    application.state.synthesis_job_service = synthesis_jobs
    application.state.text_processing_service = resolved_text_processing
    application.state.benchmark_job_service = resolved_benchmark_jobs
    application.state.experiment_service = resolved_experiments
    application.state.frontend_dist_dir = frontend_dist_dir
    return application


app = create_app()
