"""Application-level owner of the complete synthesis workflow."""

from app.audio.service import AudioService
from app.inference.base import TTSInferenceAdapter
from app.metrics.collector import MetricsCollector
from app.models.loader import ModelLoader
from app.models.registry import ModelRegistry
from app.schemas.synthesis import SynthesisRequest, SynthesisResult


class SynthesisService:
    """Coordinate model lookup, inference, metrics, and audio output.

    No HTTP concepts or concrete inference technology belong in this class.
    The workflow is deliberately left unimplemented in the architecture scaffold.
    """

    def __init__(
        self,
        model_registry: ModelRegistry,
        model_loader: ModelLoader,
        inference: TTSInferenceAdapter,
        metrics: MetricsCollector,
        audio: AudioService,
    ) -> None:
        self._model_registry = model_registry
        self._model_loader = model_loader
        self._inference = inference
        self._metrics = metrics
        self._audio = audio

    async def synthesize(self, _request: SynthesisRequest) -> SynthesisResult:
        """Run the synthesis workflow once concrete collaborators are wired."""
        raise NotImplementedError("Synthesis workflow is not implemented yet.")

