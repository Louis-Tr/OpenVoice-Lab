"""Application-level owner of the complete synthesis workflow."""

import asyncio

from app.audio.service import AudioService
from app.metrics.collector import MetricsCollector
from app.models.loader import ModelLoader
from app.models.registry import ModelRegistry
from app.schemas.synthesis import SynthesisMetrics, SynthesisRequest, SynthesisResult


class SynthesisService:
    """Coordinate registry, cached inference, and audio storage."""

    def __init__(
        self,
        model_registry: ModelRegistry,
        model_loader: ModelLoader,
        audio_service: AudioService,
        metrics_collector: MetricsCollector,
    ) -> None:
        self._model_registry = model_registry
        self._model_loader = model_loader
        self._audio_service = audio_service
        self._metrics_collector = metrics_collector

    async def synthesize(self, request: SynthesisRequest) -> SynthesisResult:
        """Run blocking local inference away from the event-loop thread."""
        return await asyncio.to_thread(self._synthesize_sync, request)

    def _synthesize_sync(self, request: SynthesisRequest) -> SynthesisResult:
        model = self._model_registry.get(request.model_id, request.variant)
        artifact_key = "\0".join(
            (
                model.label,
                request.voice_id,
                model.language,
                str(model.speed),
                request.text,
            )
        )
        loaded = self._metrics_collector.measure_model_load(
            lambda: self._model_loader.load_with_state(model)
        )
        measured = self._metrics_collector.measure(
            lambda: loaded.value.engine.synthesize(
                request.text,
                request.voice_id,
                speed=model.speed,
                language=model.language,
            ),
            model_load_ms=loaded.elapsed_ms,
            warm=loaded.value.warm,
            model_variant=model.variant,
        )
        artifact = self._audio_service.create_artifact(
            measured.audio,
            model=model.label,
            voice=request.voice_id,
            artifact_key=artifact_key,
        )
        snapshot = measured.metrics
        return SynthesisResult(
            status="ok",
            model=model.label,
            text=request.text,
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
