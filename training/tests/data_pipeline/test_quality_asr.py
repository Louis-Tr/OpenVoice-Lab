import numpy as np

from training.data_pipeline.asr import word_error_rate
from training.data_pipeline.quality import calculate_quality


def test_quality_metrics_are_actual_signal_measurements() -> None:
    values = np.array([0.0, 0.0, 0.25, -0.25, 1.0, -1.0], dtype=np.float32)
    metrics = calculate_quality(values, -40.0)

    assert metrics["peak"] == 1.0
    assert metrics["clipping_ratio"] == 2 / 6
    assert metrics["silence_ratio"] == 2 / 6
    assert metrics["rms_dbfs"] < 0


def test_wer_is_deterministic_and_handles_empty_text() -> None:
    assert word_error_rate("chest pain", "chest pain") == 0.0
    assert word_error_rate("chest pain", "chest") == 0.5
    assert word_error_rate("", "") == 0.0
    assert word_error_rate("", "unexpected") == 1.0
