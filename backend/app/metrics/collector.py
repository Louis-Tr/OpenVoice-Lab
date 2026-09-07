"""Model-level timing and process RSS measurement around inference execution."""

import json
import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from threading import Event, Thread
from time import perf_counter
from typing import Generic, TypeVar

import psutil

from app.inference.base import AudioResult

Clock = Callable[[], float]
MemoryReader = Callable[[], float]
LoadResult_co = TypeVar("LoadResult_co", covariant=True)
DEFAULT_RSS_SAMPLE_INTERVAL_SECONDS = 0.075
LOGGER = logging.getLogger("uvicorn.error")


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
class MeasuredModelLoad(Generic[LoadResult_co]):
    """A model-loader result plus timing and process RSS boundary values."""

    value: LoadResult_co
    elapsed_ms: float
    pre_load_rss_mb: float
    post_load_rss_mb: float

    @property
    def rss_delta_mb(self) -> float:
        """Return the signed process RSS change across the loader boundary."""
        return self.post_load_rss_mb - self.pre_load_rss_mb


@dataclass(frozen=True, slots=True)
class InferenceRssMeasurement:
    """Process RSS boundary and sampled peak values for one inference call."""

    pre_inference_rss_mb: float
    post_inference_rss_mb: float
    peak_inference_rss_mb: float

    @property
    def peak_delta_mb(self) -> float:
        """Return peak process RSS above the pre-inference baseline."""
        return self.peak_inference_rss_mb - self.pre_inference_rss_mb


class ProcessRssSampler:
    """Sample process RSS on one daemon thread for a bounded operation."""

    def __init__(
        self,
        memory_reader: MemoryReader,
        *,
        interval_seconds: float = DEFAULT_RSS_SAMPLE_INTERVAL_SECONDS,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self._memory_reader = memory_reader
        self._interval_seconds = interval_seconds
        self._stop = Event()
        self._thread: Thread | None = None
        self._pre_inference_rss_mb: float | None = None
        self._peak_inference_rss_mb: float | None = None

    @property
    def running(self) -> bool:
        """Return whether the sampling thread is currently alive."""
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """Capture the pre-inference RSS and begin periodic peak sampling."""
        if self._thread is not None:
            raise RuntimeError("Process RSS sampler cannot be started more than once.")
        baseline = self._memory_reader()
        self._pre_inference_rss_mb = baseline
        self._peak_inference_rss_mb = baseline
        self._thread = Thread(
            target=self._sample,
            name="tts-process-rss-sampler",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> InferenceRssMeasurement:
        """Capture post-inference RSS, stop sampling, and return the observed peak."""
        if self._thread is None or self._pre_inference_rss_mb is None:
            raise RuntimeError("Process RSS sampler has not been started.")
        try:
            post_inference_rss_mb = self._memory_reader()
        finally:
            self._stop.set()
            self._thread.join()
        peak_inference_rss_mb = max(
            self._peak_inference_rss_mb or self._pre_inference_rss_mb,
            post_inference_rss_mb,
        )
        return InferenceRssMeasurement(
            pre_inference_rss_mb=self._pre_inference_rss_mb,
            post_inference_rss_mb=post_inference_rss_mb,
            peak_inference_rss_mb=peak_inference_rss_mb,
        )

    def _sample(self) -> None:
        while not self._stop.wait(self._interval_seconds):
            try:
                reading = self._memory_reader()
            except Exception:  # noqa: BLE001 - sampling must not interrupt inference.
                return
            self._peak_inference_rss_mb = max(
                self._peak_inference_rss_mb or reading,
                reading,
            )


class MetricsCollector:
    """Collect process and inference metrics independently of HTTP handling."""

    def __init__(
        self,
        *,
        clock: Clock = perf_counter,
        memory_reader: MemoryReader | None = None,
        rss_sample_interval_seconds: float = DEFAULT_RSS_SAMPLE_INTERVAL_SECONDS,
        logger: logging.Logger = LOGGER,
    ) -> None:
        self._clock = clock
        if memory_reader is None:
            process = psutil.Process()
            memory_reader = lambda: process.memory_info().rss / (1024 * 1024)
        self._memory_reader = lambda: self._validated_rss(memory_reader())
        if rss_sample_interval_seconds <= 0:
            raise ValueError("rss_sample_interval_seconds must be positive")
        self._rss_sample_interval_seconds = rss_sample_interval_seconds
        self._logger = logger

    def measure_model_load(
        self,
        load: Callable[[], LoadResult_co],
    ) -> MeasuredModelLoad[LoadResult_co]:
        """Measure the model-loader boundary without importing model implementations."""
        pre_load_rss_mb = self._memory_reader()
        started_at = self._clock()
        value = load()
        elapsed_ms = max(0.0, (self._clock() - started_at) * 1_000)
        post_load_rss_mb = self._memory_reader()
        return MeasuredModelLoad(
            value=value,
            elapsed_ms=elapsed_ms,
            pre_load_rss_mb=pre_load_rss_mb,
            post_load_rss_mb=post_load_rss_mb,
        )

    def measure(
        self,
        inference: Callable[[], AudioResult],
        *,
        model_id: str,
        model_load: MeasuredModelLoad[object],
        warm: bool,
        model_variant: str,
    ) -> MeasuredInference:
        """Execute inference once and calculate stable, serializable measurements."""
        sampler = ProcessRssSampler(
            self._memory_reader,
            interval_seconds=self._rss_sample_interval_seconds,
        )
        rss: InferenceRssMeasurement | None = None
        inference_ms = 0.0
        status = "failed"
        sampler.start()
        started_at = self._clock()
        try:
            try:
                audio = inference()
            finally:
                try:
                    inference_ms = max(0.0, (self._clock() - started_at) * 1_000)
                finally:
                    rss = sampler.stop()

            audio_duration_ms = audio.duration_seconds * 1_000
            if audio_duration_ms <= 0:
                raise MetricsCollectionError("Cannot calculate RTF for zero-duration audio.")

            rounded_inference_ms = round(inference_ms, 3)
            rounded_audio_duration_ms = round(audio_duration_ms, 3)
            snapshot = MetricSnapshot(
                model_load_ms=(0.0 if warm else round(max(0.0, model_load.elapsed_ms), 3)),
                inference_ms=rounded_inference_ms,
                audio_duration_ms=rounded_audio_duration_ms,
                real_time_factor=round(
                    rounded_inference_ms / rounded_audio_duration_ms,
                    6,
                ),
                memory_mb=round(rss.post_inference_rss_mb, 3),
                warm=warm,
                model_variant=model_variant,
            )
            status = "success"
        finally:
            if rss is not None:
                self._log_process_rss(
                    model_id=model_id,
                    model_variant=model_variant,
                    model_load=model_load,
                    inference=rss,
                    inference_ms=inference_ms,
                    warm=warm,
                    status=status,
                )
        return MeasuredInference(audio=audio, metrics=snapshot)

    @staticmethod
    def _validated_rss(reading: float) -> float:
        value = float(reading)
        if not math.isfinite(value) or value < 0:
            raise MetricsCollectionError("Process RSS measurement must be finite and non-negative.")
        return value

    def _log_process_rss(
        self,
        *,
        model_id: str,
        model_variant: str,
        model_load: MeasuredModelLoad[object],
        inference: InferenceRssMeasurement,
        inference_ms: float,
        warm: bool,
        status: str,
    ) -> None:
        """Emit process-level RSS values through the configured application logger."""
        fields: dict[str, str | bool | float] = {
            "event": "tts_process_rss_measurement",
            "status": status,
            "rss_scope": "process",
            "rss_unit": "MiB",
            "model_id": model_id,
            "model_variant": model_variant,
            "cache_state": "hit" if warm else "miss",
            "warm": warm,
            "pre_load_rss_mb": round(model_load.pre_load_rss_mb, 3),
            "post_load_rss_mb": round(model_load.post_load_rss_mb, 3),
            "model_load_rss_delta_mb": round(model_load.rss_delta_mb, 3),
            "model_load_duration_ms": round(max(0.0, model_load.elapsed_ms), 3),
            "pre_inference_rss_mb": round(inference.pre_inference_rss_mb, 3),
            "post_inference_rss_mb": round(inference.post_inference_rss_mb, 3),
            "peak_inference_rss_mb": round(inference.peak_inference_rss_mb, 3),
            "inference_peak_rss_delta_mb": round(inference.peak_delta_mb, 3),
            "inference_duration_ms": round(max(0.0, inference_ms), 3),
            "rss_sample_interval_ms": round(self._rss_sample_interval_seconds * 1_000, 3),
        }
        self._logger.info(
            json.dumps(fields, sort_keys=True, separators=(",", ":")),
            extra=fields,
        )
