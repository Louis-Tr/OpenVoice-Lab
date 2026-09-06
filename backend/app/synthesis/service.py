"""Application-level owner of the complete synthesis workflow."""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass

from app.audio.service import AudioService
from app.inference.base import InputTooLongError, UnsupportedVoiceError
from app.metrics.collector import MetricsCollector
from app.models.loader import ModelLoader
from app.models.registry import ModelDefinition, ModelRegistry
from app.schemas.synthesis import SynthesisMetrics, SynthesisRequest, SynthesisResult
from app.text_processing.service import TextProcessingService

ProgressCallback = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class PreparedSynthesis:
    """Validated immutable request captured before it enters the queue."""

    request: SynthesisRequest
    model: ModelDefinition
    normalized_text: str
    artifact_key: str


class SynthesisService:
    """Coordinate registry, leased inference, and audio storage."""

    def __init__(
        self,
        model_registry: ModelRegistry,
        model_loader: ModelLoader,
        audio_service: AudioService,
        metrics_collector: MetricsCollector,
        text_processing_service: TextProcessingService,
    ) -> None:
        self._model_registry = model_registry
        self._model_loader = model_loader
        self._audio_service = audio_service
        self._metrics_collector = metrics_collector
        self._text_processing_service = text_processing_service

    @property
    def model_loader(self) -> ModelLoader:
        return self._model_loader

    def prepare(self, request: SynthesisRequest) -> PreparedSynthesis:
        """Validate and normalize before the request consumes queue capacity."""
        model = self._model_registry.get(request.model_id)
        if request.voice_id not in model.voices:
            raise UnsupportedVoiceError(
                f"Voice '{request.voice_id}' is not available for {model.display_name}."
            )
        if len(request.text) > model.max_input_characters:
            raise InputTooLongError(
                f"{model.display_name} accepts at most {model.max_input_characters} "
                f"characters; received {len(request.text)}. Shorten the text or choose "
                "another model."
            )
        normalized_text = self._text_processing_service.process(
            request.text,
            sanitize_text=request.sanitize_text,
            normalize_text=request.normalize_text,
        )
        artifact_key = "\0".join(
            (
                model.label,
                request.voice_id,
                model.language,
                str(model.speed),
                normalized_text,
            )
        )
        return PreparedSynthesis(request, model, normalized_text, artifact_key)

    async def synthesize(self, request: SynthesisRequest) -> SynthesisResult:
        """Compatibility helper when no application scheduler is present."""
        prepared = self.prepare(request)
        return await asyncio.to_thread(self.execute, prepared)

    def execute(
        self,
        prepared: PreparedSynthesis,
        progress: ProgressCallback | None = None,
    ) -> SynthesisResult:
        request = prepared.request
        model = prepared.model
        normalized_text = prepared.normalized_text
        try:
            if progress is not None:
                progress("loading")
            loaded = self._metrics_collector.measure_model_load(
                lambda: self._model_loader.lease_with_state(model)
            )
            with loaded.value as lease:
                if progress is not None:
                    progress("running")
                measured = self._metrics_collector.measure(
                    lambda: lease.engine.synthesize(
                        normalized_text,
                        request.voice_id,
                        speed=model.speed,
                        language=model.language,
                    ),
                    model_load_ms=loaded.elapsed_ms,
                    warm=lease.warm,
                    model_variant=model.variant,
                )
                if progress is not None:
                    progress("saving")
                artifact = self._audio_service.create_artifact(
                    measured.audio,
                    model=model.label,
                    voice=request.voice_id,
                    artifact_key=prepared.artifact_key,
                )
        except Exception as error:
            error.normalized_text = normalized_text
            raise
        snapshot = measured.metrics
        return SynthesisResult(
            status="ok",
            model=model.label,
            text=request.text,
            normalized_text=normalized_text,
            audio_url=artifact.url,
            metrics=SynthesisMetrics(
                model_load_ms=snapshot.model_load_ms,
                inference_ms=snapshot.inference_ms,
                audio_duration_ms=snapshot.audio_duration_ms,
                real_time_factor=snapshot.real_time_factor,
                memory_mb=snapshot.memory_mb,
                warm=snapshot.warm,
                model_variant=model.variant,
            ),
        )
