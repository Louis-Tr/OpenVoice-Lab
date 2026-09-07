"""Model loading and lifecycle boundary."""

from collections.abc import Callable
from threading import Lock

from app.inference.audio8_onnx import Audio8ONNXEngine
from app.inference.base import InferenceError, TTSInferenceEngine
from app.inference.kokoro_onnx import KokoroONNXEngine
from app.inference.speecht5 import SpeechT5InferenceEngine
from app.models.registry import ModelDefinition

EngineFactory = Callable[[ModelDefinition], TTSInferenceEngine]


class ModelLoadError(RuntimeError):
    """Raised when local model artifacts cannot become a runtime engine."""


class ModelLoader:
    """Construct engines; the scheduler exclusively owns caching and lifetime."""

    def __init__(
        self,
        engine_factory: EngineFactory | None = None,
        *,
        cpu_threads: int | None = None,
    ) -> None:
        self._engine_factory = engine_factory
        self._cpu_threads = cpu_threads
        self._load_counts: dict[str, int] = {}
        self._lock = Lock()

    def load(self, model: ModelDefinition) -> TTSInferenceEngine:
        """Validate artifacts and construct exactly one engine per invocation."""
        if not model.artifacts_available:
            raise ModelLoadError(model.resolved_unavailable_reason or "Model unavailable.")
        try:
            engine = (
                self._engine_factory(model)
                if self._engine_factory is not None
                else self._create_engine(model)
            )
        except InferenceError as error:
            raise ModelLoadError(str(error)) from error
        except Exception as error:
            raise ModelLoadError(f"Failed to load model '{model.label}': {error}") from error
        with self._lock:
            self._load_counts[model.key] = self._load_counts.get(model.key, 0) + 1
        return engine

    def load_count(self, model_id: str) -> int:
        """Expose lifecycle evidence for diagnostics and tests."""
        with self._lock:
            return self._load_counts.get(model_id, 0)

    def _create_engine(self, model: ModelDefinition) -> TTSInferenceEngine:
        if model.engine == "kokoro-onnx":
            if model.voices_path is None:
                raise ModelLoadError("Kokoro requires a local voice-vector artifact.")
            return KokoroONNXEngine(
                model.model_path, model.voices_path, cpu_threads=self._cpu_threads or 1
            )
        if model.engine == "speecht5-transformers":
            if model.voices_path is None or len(model.additional_artifacts) != 1:
                raise ModelLoadError("SpeechT5 requires a speaker profile and vocoder.")
            return SpeechT5InferenceEngine(
                model_root=model.model_path,
                vocoder_root=model.additional_artifacts[0],
                speaker_embedding_path=model.voices_path,
                voice_id=model.voices[0],
                cpu_threads=self._cpu_threads or 1,
            )
        if model.engine == "audio8-onnx":
            return Audio8ONNXEngine(
                model.model_path,
                voice_id=model.voices[0],
                threads=self._cpu_threads,
            )
        raise ModelLoadError(
            model.resolved_unavailable_reason or f"Unsupported runtime '{model.runtime}'."
        )
