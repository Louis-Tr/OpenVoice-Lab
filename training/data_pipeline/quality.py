from __future__ import annotations

import math
from typing import Any

import numpy as np

from .audio import read_pcm_mono
from .config import PipelineConfig


def _append_unique(record: dict[str, Any], field: str, value: str) -> None:
    if value not in record[field]:
        record[field].append(value)


def calculate_quality(
    values: np.ndarray, silence_dbfs: float
) -> dict[str, float | None]:
    absolute = np.abs(values.astype(np.float64, copy=False))
    rms = (
        float(np.sqrt(np.mean(np.square(values, dtype=np.float64))))
        if values.size
        else 0.0
    )
    rms_dbfs = 20.0 * math.log10(max(rms, 1e-12))
    peak = float(absolute.max(initial=0.0))
    silence_threshold = 10.0 ** (silence_dbfs / 20.0)
    silence_ratio = (
        float(np.mean(absolute <= silence_threshold)) if values.size else 1.0
    )
    clipping_ratio = (
        float(np.mean(absolute >= (32766.0 / 32767.0))) if values.size else 0.0
    )
    active = absolute[absolute > silence_threshold]
    quiet = absolute[absolute <= silence_threshold]
    snr_proxy: float | None = None
    if active.size and quiet.size:
        signal_rms = float(np.sqrt(np.mean(np.square(active))))
        noise_rms = float(np.sqrt(np.mean(np.square(quiet))))
        snr_proxy = 20.0 * math.log10(max(signal_rms, 1e-12) / max(noise_rms, 1e-12))
    return {
        "rms": rms,
        "rms_dbfs": rms_dbfs,
        "peak": peak,
        "silence_ratio": silence_ratio,
        "clipping_ratio": clipping_ratio,
        "snr_proxy_db": snr_proxy,
    }


def screen_quality(
    records: list[dict[str, Any]], config: PipelineConfig
) -> list[dict[str, Any]]:
    quality = config.section("quality")
    hard = quality["hard_reject"]
    review = quality["manual_review"]
    silence_dbfs = float(quality["silence_amplitude_dbfs"])
    for record in records:
        clean_relative = record.get("clean_audio")
        if not clean_relative:
            continue
        clean_path = config.repository_root / clean_relative
        try:
            values, sample_rate, channels, width = read_pcm_mono(clean_path)
            metrics = calculate_quality(values, silence_dbfs)
            metrics["duration_seconds"] = len(values) / sample_rate
            record["quality_metrics"] = metrics
            if metrics["rms_dbfs"] < float(hard["minimum_rms_dbfs"]):
                _append_unique(record, "exclusion_reasons", "QUALITY_RMS_TOO_LOW")
            if metrics["peak"] < float(hard["minimum_peak_amplitude"]):
                _append_unique(record, "exclusion_reasons", "QUALITY_PEAK_TOO_LOW")
            if metrics["silence_ratio"] > float(hard["maximum_silence_ratio"]):
                _append_unique(record, "exclusion_reasons", "QUALITY_SILENCE_EXCESSIVE")
            if metrics["clipping_ratio"] > float(hard["maximum_clipping_ratio"]):
                _append_unique(
                    record, "exclusion_reasons", "QUALITY_CLIPPING_EXCESSIVE"
                )

            if metrics["rms_dbfs"] < float(review["minimum_rms_dbfs"]):
                _append_unique(record, "review_flags", "QUALITY_RMS_LOW_REVIEW")
            if metrics["silence_ratio"] > float(review["maximum_silence_ratio"]):
                _append_unique(record, "review_flags", "QUALITY_SILENCE_HIGH_REVIEW")
            if metrics["clipping_ratio"] > float(review["maximum_clipping_ratio"]):
                _append_unique(record, "review_flags", "QUALITY_CLIPPING_REVIEW")

            metadata = record.get("source_metadata", {})
            record["source_quality_signals"] = {
                "audio_clipping": metadata.get("audio_clipping"),
                "background_noise_audible": metadata.get("background_noise_audible"),
                "quiet_speaker": metadata.get("quiet_speaker"),
                "overall_quality": record.get("parsed_metadata", {}).get(
                    "overall_quality_of_the_audio"
                ),
                "policy": "signals_only_not_automatic_rejection",
            }
            record["quality_status"] = "screened"
        except Exception as exc:
            _append_unique(record, "exclusion_reasons", "QUALITY_ANALYSIS_FAILED")
            record["quality_status"] = "failed"
            record["quality_error"] = f"{type(exc).__name__}: {exc}"
        record["exclusion_reasons"].sort()
        record["review_flags"].sort()
    return records
