"""Application-level owner of the complete synthesis workflow."""

import asyncio
from collections.abc import Callable

from app.audio.service import AudioService
from app.inference.base import InputTooLongError, UnsupportedVoiceError
from app.metrics.collector import MetricsCollector
from app.models.registry import ModelRegistry
from app.models.scheduler import EngineScheduler
from app.schemas.synthesis import SynthesisMetrics, SynthesisRequest, SynthesisResult
from app.text_processing.service import TextProcessingService


class SynthesisService:
    """Coordinate registry, cached inference, and audio storage."""

    def __init__(
        self,
        model_registry: ModelRegistry,
        engine_scheduler: EngineScheduler,
        audio_service: AudioService,
        metrics_collector: MetricsCollector,
        text_processing_service: TextProcessingService,
    ) -> None:
        self._model_registry = model_registry
        self._engine_scheduler = engine_scheduler
        self._audio_service = audio_service
        self._metrics_collector = metrics_collector
        self._text_processing_service = text_processing_service

    async def synthesize(
        self,
        request: SynthesisRequest,
        *,
        on_admitted: Callable[[], None] | None = None,
        on_stage: Callable[[str], None] | None = None,
    ) -> SynthesisResult:
        """Run blocking local inference away from the event-loop thread."""
        return await asyncio.to_thread(
            self._synthesize_sync, request, on_admitted=on_admitted, on_stage=on_stage
        )

    def _synthesize_sync(
        self,
        request: SynthesisRequest,
        *,
        on_admitted: Callable[[], None] | None = None,
        on_stage: Callable[[str], None] | None = None,
    ) -> SynthesisResult:
        normalized_text = self._text_processing_service.process(
            request.text,
            sanitize_text=request.sanitize_text,
            normalize_text=request.normalize_text,
        )
        try:
            model = self._model_registry.get(request.model_id)
            if on_admitted is not None and request.voice_id not in model.voices:
                raise UnsupportedVoiceError(f"Voice '{request.voice_id}' is not available.")
            if len(request.text) > model.max_input_characters:
                raise InputTooLongError(
                    f"{model.display_name} accepts at most {model.max_input_characters} "
                    f"characters; received {len(request.text)}. Shorten the text or choose "
                    "another model."
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
            with self._engine_scheduler.acquire(model, on_admitted=on_admitted) as lease:
                if on_stage is not None:
                    on_stage("generating")
                measured = self._metrics_collector.measure(
                    lambda: lease.loaded.value.synthesize(
                        normalized_text,
                        request.voice_id,
                        speed=model.speed,
                        language=model.language,
                    ),
                    model_id=model.model_id,
                    model_load=lease.loaded,
                    warm=lease.warm,
                    model_variant=model.variant,
                )
                if on_stage is not None:
                    on_stage("saving")
                artifact = self._audio_service.create_artifact(
                    measured.audio,
                    model=model.label,
                    voice=request.voice_id,
                    artifact_key=artifact_key,
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
