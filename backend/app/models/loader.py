"""Lease-aware model loading and lifecycle boundary."""

import gc
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from threading import Event, Lock
from typing import Self

from app.inference.audio8_onnx import Audio8ONNXEngine
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


@dataclass(slots=True)
class _EngineEntry:
    engine: TTSInferenceEngine
    active_leases: int = 0


class ModelLease:
    """Keep an engine resident until a complete synthesis operation finishes."""

    def __init__(self, loader: "ModelLoader", key: str, result: ModelLoadResult) -> None:
        self._loader = loader
        self._key = key
        self.result = result
        self._released = False

    @property
    def engine(self) -> TTSInferenceEngine:
        return self.result.engine

    @property
    def warm(self) -> bool:
        return self.result.warm

    def release(self) -> None:
        if not self._released:
            self._released = True
            self._loader._release(self._key)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.release()


class ModelLoader:
    """Share loads, expose leases, and evict only idle least-recently-used engines."""

    def __init__(
        self,
        engine_factory: EngineFactory | None = None,
        *,
        maximum_cached_engines: int | None = None,
        cpu_threads: int | None = None,
    ) -> None:
        if maximum_cached_engines is not None and maximum_cached_engines < 1:
            raise ValueError("maximum_cached_engines must be at least one")
        self._engine_factory = engine_factory
        self._maximum_cached_engines = maximum_cached_engines
        self._cpu_threads = cpu_threads
        self._engines: OrderedDict[str, _EngineEntry] = OrderedDict()
        self._loading: dict[str, Event] = {}
        self._load_errors: dict[str, ModelLoadError] = {}
        self._load_counts: dict[str, int] = {}
        self._lock = Lock()

    def load(self, model: ModelDefinition) -> TTSInferenceEngine:
        """Compatibility helper for callers that do not need a retained lease."""
        return self.load_with_state(model).engine

    def load_with_state(self, model: ModelDefinition) -> ModelLoadResult:
        """Load without retaining a lease; synthesis should use ``lease_with_state``."""
        lease = self.lease_with_state(model)
        try:
            return lease.result
        finally:
            lease.release()

    def lease_with_state(self, model: ModelDefinition) -> ModelLease:
        """Return a lease, sharing an in-progress load for the same model."""
        key = model.key
        while True:
            with self._lock:
                cached = self._engines.get(key)
                if cached is not None:
                    cached.active_leases += 1
                    self._engines.move_to_end(key)
                    return ModelLease(
                        self,
                        key,
                        ModelLoadResult(engine=cached.engine, warm=True),
                    )
                loading = self._loading.get(key)
                if loading is None:
                    if not model.artifacts_available:
                        raise ModelLoadError(
                            model.resolved_unavailable_reason
                            or f"Model '{model.label}' is not available."
                        )
                    loading = Event()
                    self._loading[key] = loading
                    self._load_errors.pop(key, None)
                    evicted = self._evict_idle_locked(reserve_slot=True)
                    break
            loading.wait()
            with self._lock:
                error = self._load_errors.get(key)
            if error is not None:
                raise error

        self._close_engines(evicted)
        try:
            engine = (
                self._engine_factory(model)
                if self._engine_factory is not None
                else self._create_engine(model)
            )
        except InferenceError as error:
            failure = ModelLoadError(str(error))
        except Exception as error:  # noqa: BLE001 - model runtimes vary.
            failure = ModelLoadError(f"Failed to load model '{model.label}': {error}")
        else:
            failure = None

        with self._lock:
            signal = self._loading.pop(key)
            if failure is None:
                self._engines[key] = _EngineEntry(engine=engine, active_leases=1)
                self._load_counts[key] = self._load_counts.get(key, 0) + 1
            else:
                self._load_errors[key] = failure
            signal.set()
        if failure is not None:
            raise failure
        return ModelLease(self, key, ModelLoadResult(engine=engine, warm=False))

    def is_loaded(self, model_id: str) -> bool:
        with self._lock:
            return model_id in self._engines

    def engine_states(self) -> dict[str, str]:
        with self._lock:
            states = {
                key: "busy" if entry.active_leases else "idle"
                for key, entry in self._engines.items()
            }
            states.update({key: "loading" for key in self._loading})
            return states

    def load_count(self, model_id: str) -> int:
        """Expose lifecycle evidence for diagnostics and tests."""
        return self._load_counts.get(model_id, 0)

    def _release(self, key: str) -> None:
        with self._lock:
            entry = self._engines.get(key)
            if entry is None or entry.active_leases <= 0:
                raise RuntimeError(f"Model lease accounting failed for '{key}'.")
            entry.active_leases -= 1
            evicted = self._evict_idle_locked(reserve_slot=False)
        self._close_engines(evicted)

    def _evict_idle_locked(self, *, reserve_slot: bool) -> list[TTSInferenceEngine]:
        if self._maximum_cached_engines is None:
            return []
        target = self._maximum_cached_engines - (1 if reserve_slot else 0)
        evicted: list[TTSInferenceEngine] = []
        while len(self._engines) > max(0, target):
            idle_key = next(
                (key for key, entry in self._engines.items() if entry.active_leases == 0),
                None,
            )
            if idle_key is None:
                break
            evicted.append(self._engines.pop(idle_key).engine)
        return evicted

    @staticmethod
    def _close_engines(engines: list[TTSInferenceEngine]) -> None:
        for engine in engines:
            close = getattr(engine, "close", None)
            if callable(close):
                close()
        if engines:
            gc.collect()

    def _create_engine(self, model: ModelDefinition) -> TTSInferenceEngine:
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
        if model.engine == "audio8-onnx":
            return Audio8ONNXEngine(
                model.model_path,
                voice_id=model.voices[0],
                threads=self._cpu_threads,
            )
        raise ModelLoadError(
            model.resolved_unavailable_reason
            or f"Unsupported runtime '{model.runtime}'."
        )
