"""Model loading and lifecycle boundary."""

from collections.abc import Callable
from dataclasses import dataclass
from threading import Lock

from app.inference.base import InferenceError, TTSInferenceEngine
from app.inference.kokoro_onnx import KokoroONNXEngine
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

            missing = [
                str(path)
                for path in (model.model_path, model.voices_path)
                if not path.is_file()
            ]
            if missing:
                raise ModelLoadError(
                    "Missing local model artifacts: " + ", ".join(missing)
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
        if model.runtime != "ONNX":
            raise ModelLoadError(f"Unsupported runtime '{model.runtime}'.")
        return KokoroONNXEngine(model.model_path, model.voices_path)
