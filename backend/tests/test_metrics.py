"""Mathematical contract tests for inference performance instrumentation."""

import numpy as np
import pytest

from app.inference.base import AudioResult
from app.metrics.collector import MetricsCollectionError, MetricsCollector


def test_collector_measures_model_load_boundary() -> None:
    clock_values = iter((2.0, 2.8251254))
    collector = MetricsCollector(
        clock=clock_values.__next__,
        memory_reader=lambda: 100.0,
    )

    measured = collector.measure_model_load(lambda: "loaded-engine")

    assert measured.value == "loaded-engine"
    assert measured.elapsed_ms == pytest.approx(825.1254)


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
        model_load_ms=825.1254,
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
            model_load_ms=0.0,
            warm=True,
            model_variant="fp32",
        )
