"""Mathematical contract tests for inference performance instrumentation."""

import json
import logging
from threading import Event

import numpy as np
import pytest

from app.inference.base import AudioResult
from app.metrics import collector as collector_module
from app.metrics.collector import (
    MeasuredModelLoad,
    MetricsCollectionError,
    MetricsCollector,
    ProcessRssSampler,
)


def test_collector_measures_model_load_boundary() -> None:
    clock_values = iter((2.0, 2.8251254))
    rss_values = iter((100.0, 168.5))
    collector = MetricsCollector(
        clock=clock_values.__next__,
        memory_reader=rss_values.__next__,
    )

    measured = collector.measure_model_load(lambda: "loaded-engine")

    assert measured.value == "loaded-engine"
    assert measured.elapsed_ms == pytest.approx(825.1254)
    assert measured.pre_load_rss_mb == 100.0
    assert measured.post_load_rss_mb == 168.5
    assert measured.rss_delta_mb == 68.5


def test_collector_measures_inference_and_calculates_rtf() -> None:
    clock_values = iter((10.0, 10.4))
    audio = AudioResult(
        samples=np.zeros(48_000, dtype=np.float32),
        sample_rate_hz=24_000,
    )
    collector = MetricsCollector(
        clock=clock_values.__next__,
        memory_reader=lambda: 715.25,
    )

    measured = collector.measure(
        lambda: audio,
        model_id="kokoro-fp32",
        model_load=MeasuredModelLoad(
            value="loaded-engine",
            elapsed_ms=825.1254,
            pre_load_rss_mb=600.0,
            post_load_rss_mb=700.0,
        ),
        warm=False,
        model_variant="fp32",
    )

    assert measured.audio is audio
    assert measured.metrics.model_load_ms == 825.125
    assert measured.metrics.inference_ms == 400.0
    assert measured.metrics.audio_duration_ms == 2_000.0
    assert measured.metrics.real_time_factor == 0.2
    assert measured.metrics.memory_mb == 715.25
    assert measured.metrics.warm is False
    assert measured.metrics.model_variant == "fp32"
    assert measured.metrics.real_time_factor == pytest.approx(
        measured.metrics.inference_ms / measured.metrics.audio_duration_ms,
        abs=1e-6,
    )


def test_collector_rejects_zero_duration_audio() -> None:
    clock_values = iter((1.0, 1.1))
    collector = MetricsCollector(
        clock=clock_values.__next__,
        memory_reader=lambda: 100.0,
    )

    with pytest.raises(MetricsCollectionError, match="zero-duration"):
        collector.measure(
            lambda: AudioResult(
                samples=np.array([], dtype=np.float32),
                sample_rate_hz=24_000,
            ),
            model_id="kokoro-fp32",
            model_load=MeasuredModelLoad(
                value="loaded-engine",
                elapsed_ms=0.0,
                pre_load_rss_mb=100.0,
                post_load_rss_mb=100.0,
            ),
            warm=True,
            model_variant="fp32",
        )


def test_process_rss_sampler_captures_temporary_peak_and_stops() -> None:
    current_rss_mb = [100.0]
    peak_sampled = Event()

    def read_rss() -> float:
        reading = current_rss_mb[0]
        if reading == 145.0:
            peak_sampled.set()
        return reading

    sampler = ProcessRssSampler(read_rss, interval_seconds=0.001)
    sampler.start()
    current_rss_mb[0] = 145.0
    assert peak_sampled.wait(timeout=0.5)
    current_rss_mb[0] = 120.0

    measured = sampler.stop()

    assert measured.pre_inference_rss_mb == 100.0
    assert measured.post_inference_rss_mb == 120.0
    assert measured.peak_inference_rss_mb == 145.0
    assert measured.peak_delta_mb == 45.0
    assert sampler.running is False


def test_collector_stops_sampler_when_inference_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[ProcessRssSampler] = []
    sampler_type = ProcessRssSampler

    def recording_sampler(
        memory_reader: collector_module.MemoryReader,
        *,
        interval_seconds: float,
    ) -> ProcessRssSampler:
        sampler = sampler_type(memory_reader, interval_seconds=interval_seconds)
        created.append(sampler)
        return sampler

    monkeypatch.setattr(collector_module, "ProcessRssSampler", recording_sampler)
    clock_values = iter((1.0, 1.2))
    collector = MetricsCollector(
        clock=clock_values.__next__,
        memory_reader=lambda: 100.0,
    )

    def fail() -> AudioResult:
        raise RuntimeError("inference failed")

    with pytest.raises(RuntimeError, match="inference failed"):
        collector.measure(
            fail,
            model_id="kokoro-fp32",
            model_load=MeasuredModelLoad(
                value="loaded-engine",
                elapsed_ms=50.0,
                pre_load_rss_mb=75.0,
                post_load_rss_mb=100.0,
            ),
            warm=False,
            model_variant="fp32",
        )

    assert len(created) == 1
    assert created[0].running is False


def test_collector_emits_structured_process_rss_fields(
    caplog: pytest.LogCaptureFixture,
) -> None:
    current_rss_mb = [100.0]
    clock_values = iter((1.0, 1.25, 2.0, 2.4))
    logger = logging.getLogger("tests.tts-process-rss")
    collector = MetricsCollector(
        clock=clock_values.__next__,
        memory_reader=lambda: current_rss_mb[0],
        logger=logger,
    )

    def load() -> str:
        current_rss_mb[0] = 125.0
        return "loaded-engine"

    loaded = collector.measure_model_load(load)
    audio = AudioResult(
        samples=np.zeros(24_000, dtype=np.float32),
        sample_rate_hz=24_000,
    )

    def infer() -> AudioResult:
        current_rss_mb[0] = 140.0
        return audio

    with caplog.at_level(logging.INFO, logger=logger.name):
        collector.measure(
            infer,
            model_id="kokoro-fp32",
            model_load=loaded,
            warm=False,
            model_variant="fp32",
        )

    record = next(item for item in caplog.records if item.event == "tts_process_rss_measurement")
    fields = json.loads(record.getMessage())
    assert fields == {
        "cache_state": "miss",
        "event": "tts_process_rss_measurement",
        "inference_duration_ms": 400.0,
        "inference_peak_rss_delta_mb": 15.0,
        "model_id": "kokoro-fp32",
        "model_load_duration_ms": 250.0,
        "model_load_rss_delta_mb": 25.0,
        "model_variant": "fp32",
        "peak_inference_rss_mb": 140.0,
        "post_inference_rss_mb": 140.0,
        "post_load_rss_mb": 125.0,
        "pre_inference_rss_mb": 125.0,
        "pre_load_rss_mb": 100.0,
        "rss_sample_interval_ms": 75.0,
        "rss_scope": "process",
        "rss_unit": "MiB",
        "status": "success",
        "warm": False,
    }
    assert record.model_id == "kokoro-fp32"
    assert record.cache_state == "miss"
