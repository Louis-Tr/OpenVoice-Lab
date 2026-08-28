"""Model loading and lifecycle boundary."""

from collections.abc import Callable
from dataclasses import dataclass
from threading import Lock

from app.inference.base import InferenceError, TTSInferenceEngine
from app.inference.kokoro_onnx import KokoroONNXEngine
from app.inference.speecht5 import SpeechT5InferenceEngine
from app.models.registry import ModelDefinition

EngineFactory = Callable[[ModelDefinition], TTSInferenceEngine]


class ModelLoadError(RuntimeError):
    """Raised when local model artifacts cannot become a runtime engine."""


@dataclass(frozen=True, slots=True)
class ModelLoadResult:
    """Engine plus lifecycle state for one load request."""

    engine: TTSInferenceEngine
    warm: bool


class ModelLoader:
    """Load each runtime engine once and reuse it for warm requests."""

    def __init__(self, engine_factory: EngineFactory | None = None) -> None:
        self._engine_factory = engine_factory or self._create_engine
        self._engines: dict[str, TTSInferenceEngine] = {}
        self._load_counts: dict[str, int] = {}
        self._lock = Lock()

    def load(self, model: ModelDefinition) -> TTSInferenceEngine:
        """Return a cached engine, loading its local artifacts at most once."""
        return self.load_with_state(model).engine

    def load_with_state(self, model: ModelDefinition) -> ModelLoadResult:
        """Return the engine and whether this request reused a loaded runtime."""
        key = model.key
        with self._lock:
            cached = self._engines.get(key)
            if cached is not None:
                return ModelLoadResult(engine=cached, warm=True)

            if not model.artifacts_available:
                raise ModelLoadError(
                    model.resolved_unavailable_reason
                    or f"Model '{model.label}' is not available."
                )

            try:
                engine = self._engine_factory(model)
            except InferenceError as error:
                raise ModelLoadError(str(error)) from error
            except Exception as error:
                raise ModelLoadError(f"Failed to load model '{model.label}': {error}") from error

            self._engines[key] = engine
            self._load_counts[key] = self._load_counts.get(key, 0) + 1
            return ModelLoadResult(engine=engine, warm=False)

    def load_count(self, model_id: str) -> int:
        """Expose lifecycle evidence for diagnostics and tests."""
        return self._load_counts.get(model_id, 0)

    @staticmethod
    def _create_engine(model: ModelDefinition) -> TTSInferenceEngine:
        if model.engine == "kokoro-onnx":
            if model.voices_path is None:
                raise ModelLoadError("Kokoro requires a local voice-vector artifact.")
            return KokoroONNXEngine(model.model_path, model.voices_path)
        if model.engine == "speecht5-transformers":
            if model.voices_path is None or len(model.additional_artifacts) != 1:
                raise ModelLoadError("SpeechT5 requires a speaker profile and vocoder.")
            return SpeechT5InferenceEngine(
                model_root=model.model_path,
                vocoder_root=model.additional_artifacts[0],
                speaker_embedding_path=model.voices_path,
                voice_id=model.voices[0],
            )
        raise ModelLoadError(
            model.resolved_unavailable_reason
            or f"Unsupported runtime '{model.runtime}'."
        )
