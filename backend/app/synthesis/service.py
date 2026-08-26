"""Application-level owner of the complete synthesis workflow."""

import asyncio
from threading import Lock

from app.audio.service import AudioService
from app.models.loader import ModelLoader
from app.models.registry import ModelRegistry
from app.schemas.synthesis import SynthesisRequest, SynthesisResult


class SynthesisService:
    """Coordinate registry, cached inference, and audio storage."""

    def __init__(
        self,
        model_registry: ModelRegistry,
        model_loader: ModelLoader,
        audio_service: AudioService,
    ) -> None:
        self._model_registry = model_registry
        self._model_loader = model_loader
        self._audio_service = audio_service
        self._results: dict[str, SynthesisResult] = {}
        self._result_lock = Lock()

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
        with self._result_lock:
            cached = self._results.get(artifact_key)
            if cached is not None:
                return cached.model_copy(deep=True)

            engine = self._model_loader.load(model)
            audio = engine.synthesize(
                request.text,
                request.voice_id,
                speed=model.speed,
                language=model.language,
            )
            artifact = self._audio_service.create_artifact(
                audio,
                model=model.label,
                voice=request.voice_id,
                artifact_key=artifact_key,
            )
            result = SynthesisResult(
                status="ok",
                model=model.label,
                text=request.text,
                audio_url=artifact.url,
            )
            self._results[artifact_key] = result
            return result.model_copy(deep=True)
