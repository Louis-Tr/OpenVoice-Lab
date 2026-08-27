from __future__ import annotations

import math
import wave
from typing import Any

from .config import PipelineConfig


def _append_reason(record: dict[str, Any], reason: str) -> None:
    if reason not in record["exclusion_reasons"]:
        record["exclusion_reasons"].append(reason)


def validate_records(
    records: list[dict[str, Any]], config: PipelineConfig
) -> list[dict[str, Any]]:
    required_numeric = (
        "audio_clipping:confidence",
        "background_noise_audible:confidence",
        "overall_quality_of_the_audio",
        "quiet_speaker:confidence",
    )
    for record in records:
        for issue in record.get("intake_issues", []):
            _append_reason(record, issue)
        transcript = record.get("original_transcript", "")
        if not isinstance(transcript, str) or not transcript.strip():
            _append_reason(record, "TRANSCRIPT_EMPTY")
        if not record.get("speaker_id"):
            _append_reason(record, "SPEAKER_ID_EMPTY")
        if not record.get("prompt"):
            _append_reason(record, "PROMPT_EMPTY")

        metadata_values: dict[str, float] = {}
        for field in required_numeric:
            try:
                value = float(record["source_metadata"].get(field, ""))
                if not math.isfinite(value):
                    raise ValueError
                metadata_values[field] = value
            except (TypeError, ValueError):
                _append_reason(record, f"METADATA_UNPARSEABLE:{field}")
        record["parsed_metadata"] = metadata_values

        source_relative = record.get("source_wav")
        if not source_relative:
            continue
        source = config.repository_root / source_relative
        try:
            if not source.is_file() or source.stat().st_size <= 44:
                _append_reason(record, "WAV_MISSING_OR_EMPTY")
                continue
            with wave.open(str(source), "rb") as audio:
                frames = audio.getnframes()
                sample_rate = audio.getframerate()
                channels = audio.getnchannels()
                sample_width = audio.getsampwidth()
                compression = audio.getcomptype()
            if frames <= 0 or sample_rate <= 0 or channels <= 0 or sample_width <= 0:
                _append_reason(record, "WAV_INVALID_DIMENSIONS")
            elif compression != "NONE":
                _append_reason(record, "WAV_UNSUPPORTED_COMPRESSION")
            record["source_audio"] = {
                "frames": frames,
                "sample_rate": sample_rate,
                "channels": channels,
                "sample_width_bytes": sample_width,
                "compression": compression,
                "duration_seconds": frames / sample_rate if sample_rate else None,
            }
        except (OSError, EOFError, wave.Error) as exc:
            _append_reason(record, "WAV_UNREADABLE")
            record["structural_error"] = f"{type(exc).__name__}: {exc}"
        record["exclusion_reasons"].sort()
    return records
