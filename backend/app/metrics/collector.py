"""Model-level performance measurement independent of inference execution."""

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Generic, TypeVar

import psutil

from app.inference.base import AudioResult

Clock = Callable[[], float]
MemoryReader = Callable[[], float]
LoadResult = TypeVar("LoadResult")


class MetricsCollectionError(RuntimeError):
    """Raised when a valid inference result cannot be measured."""


@dataclass(frozen=True, slots=True)
class MetricSnapshot:
    """Backend-internal metric values for one inference operation."""

    model_load_ms: float
    inference_ms: float
    audio_duration_ms: float
    real_time_factor: float
    memory_mb: float
    warm: bool
    model_variant: str


@dataclass(frozen=True, slots=True)
class MeasuredInference:
    """Raw inference output paired with its independent measurements."""

    audio: AudioResult
    metrics: MetricSnapshot


@dataclass(frozen=True, slots=True)
class MeasuredModelLoad(Generic[LoadResult]):
    """A model-loader result plus elapsed initialization boundary time."""

    value: LoadResult
    elapsed_ms: float


class MetricsCollector:
    """Collect process and inference metrics independently of HTTP handling."""

    def __init__(
        self,
        *,
        clock: Clock = perf_counter,
        memory_reader: MemoryReader | None = None,
    ) -> None:
        self._clock = clock
        if memory_reader is None:
            process = psutil.Process()
            self._memory_reader = lambda: process.memory_info().rss / (1024 * 1024)
        else:
            self._memory_reader = memory_reader

    def measure_model_load(
        self,
        load: Callable[[], LoadResult],
    ) -> MeasuredModelLoad[LoadResult]:
        """Measure the model-loader boundary without importing model implementations."""
        started_at = self._clock()
        value = load()
        elapsed_ms = max(0.0, (self._clock() - started_at) * 1_000)
        return MeasuredModelLoad(value=value, elapsed_ms=elapsed_ms)

    def measure(
        self,
        inference: Callable[[], AudioResult],
        *,
        model_load_ms: float,
        warm: bool,
        model_variant: str,
    ) -> MeasuredInference:
        """Execute inference once and calculate stable, serializable measurements."""
        started_at = self._clock()
        audio = inference()
        inference_ms = max(0.0, (self._clock() - started_at) * 1_000)
        audio_duration_ms = audio.duration_seconds * 1_000
        if audio_duration_ms <= 0:
            raise MetricsCollectionError("Cannot calculate RTF for zero-duration audio.")

        memory_mb = self._memory_reader()
        if memory_mb < 0:
            raise MetricsCollectionError("Process memory measurement cannot be negative.")

        rounded_inference_ms = round(inference_ms, 3)
        rounded_audio_duration_ms = round(audio_duration_ms, 3)
        snapshot = MetricSnapshot(
            model_load_ms=(0.0 if warm else round(max(0.0, model_load_ms), 3)),
            inference_ms=rounded_inference_ms,
            audio_duration_ms=rounded_audio_duration_ms,
            real_time_factor=round(
                rounded_inference_ms / rounded_audio_duration_ms,
                6,
            ),
            memory_mb=round(memory_mb, 3),
            warm=warm,
            model_variant=model_variant,
        )
        return MeasuredInference(audio=audio, metrics=snapshot)
