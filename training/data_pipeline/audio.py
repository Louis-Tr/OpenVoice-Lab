from __future__ import annotations

import math
import os
import tempfile
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import resample_poly

from .config import PipelineConfig
from .io_utils import relative_posix, sha256_file


def _append_reason(record: dict[str, Any], reason: str) -> None:
    if reason not in record["exclusion_reasons"]:
        record["exclusion_reasons"].append(reason)


def _decode_pcm(raw: bytes, sample_width: int) -> np.ndarray:
    if sample_width == 1:
        return (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    if sample_width == 2:
        return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if sample_width == 3:
        packed = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        values = (
            packed[:, 0].astype(np.int32)
            | (packed[:, 1].astype(np.int32) << 8)
            | (packed[:, 2].astype(np.int32) << 16)
        )
        values = np.where(values & 0x800000, values - 0x1000000, values)
        return values.astype(np.float32) / 8388608.0
    if sample_width == 4:
        return np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    raise ValueError(f"Unsupported PCM sample width: {sample_width}")


def read_pcm_mono(path: Path) -> tuple[np.ndarray, int, int, int]:
    with wave.open(str(path), "rb") as audio:
        if audio.getcomptype() != "NONE":
            raise ValueError(f"Unsupported WAV compression: {audio.getcomptype()}")
        frames = audio.getnframes()
        sample_rate = audio.getframerate()
        channels = audio.getnchannels()
        sample_width = audio.getsampwidth()
        raw = audio.readframes(frames)
    values = _decode_pcm(raw, sample_width)
    if channels > 1:
        if values.size % channels:
            raise ValueError("PCM frame data is not divisible by channel count")
        values = values.reshape(-1, channels).mean(axis=1, dtype=np.float32)
    return values.astype(np.float32, copy=False), sample_rate, channels, sample_width


def _write_pcm16_atomic(path: Path, values: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".wav.tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        pcm = np.rint(np.clip(values, -1.0, 1.0) * 32767.0).astype("<i2")
        with wave.open(str(temporary), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(sample_rate)
            output.writeframes(pcm.tobytes())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _valid_standardized(path: Path, sample_rate: int) -> bool:
    try:
        with wave.open(str(path), "rb") as audio:
            return (
                audio.getframerate() == sample_rate
                and audio.getnchannels() == 1
                and audio.getsampwidth() == 2
                and audio.getcomptype() == "NONE"
                and audio.getnframes() > 0
            )
    except (OSError, EOFError, wave.Error):
        return False


def _standardize_one(record: dict[str, Any], config: PipelineConfig) -> dict[str, Any]:
    if record["exclusion_reasons"]:
        record["audio_standardization_status"] = "skipped_structural_rejection"
        return record
    source = config.repository_root / record["source_wav"]
    target_rate = int(config.section("audio")["sample_rate"])
    target = config.path("clean_audio") / f"{record['sample_id']}.wav"
    try:
        reused = _valid_standardized(target, target_rate)
        if not reused:
            values, source_rate, source_channels, source_width = read_pcm_mono(source)
            if source_rate != target_rate:
                divisor = math.gcd(source_rate, target_rate)
                values = resample_poly(
                    values, target_rate // divisor, source_rate // divisor
                )
            _write_pcm16_atomic(target, values, target_rate)
        with wave.open(str(target), "rb") as standardized:
            frames = standardized.getnframes()
            output_rate = standardized.getframerate()
            output_channels = standardized.getnchannels()
            output_width = standardized.getsampwidth()
        source_audio = record.get("source_audio", {})
        record["clean_audio"] = relative_posix(target, config.repository_root)
        record["standardized_audio_sha256"] = sha256_file(target)
        # Cache use is intentionally not persisted: a cold and warm run must
        # produce byte-identical manifests for the same source/configuration.
        record["audio_standardization_status"] = "standardized"
        record["standardized_audio"] = {
            "sample_rate": output_rate,
            "channels": output_channels,
            "sample_width_bytes": output_width,
            "frames": frames,
            "duration_seconds": frames / output_rate,
            "source_duration_seconds": source_audio.get("duration_seconds"),
            "source_sample_rate": source_audio.get("sample_rate"),
            "source_channels": source_audio.get("channels"),
        }
    except Exception as exc:  # Per-record isolation is intentional for corrupt audio.
        _append_reason(record, "AUDIO_CONVERSION_FAILED")
        record["audio_standardization_status"] = "failed"
        record["audio_conversion_error"] = f"{type(exc).__name__}: {exc}"
    record["exclusion_reasons"].sort()
    return record


def standardize_audio(
    records: list[dict[str, Any]], config: PipelineConfig
) -> list[dict[str, Any]]:
    workers = max(1, int(config.section("audio")["workers"]))
    config.path("clean_audio").mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        output = list(executor.map(lambda row: _standardize_one(row, config), records))
    return output


def apply_duration_filter(
    records: list[dict[str, Any]], config: PipelineConfig
) -> list[dict[str, Any]]:
    duration_config = config.section("audio")["duration"]
    minimum = float(duration_config["minimum_seconds"])
    maximum = float(duration_config["maximum_seconds"])
    for record in records:
        duration = record.get("standardized_audio", {}).get("duration_seconds")
        if duration is None:
            continue
        if duration < minimum:
            _append_reason(record, "DURATION_BELOW_MINIMUM")
        elif duration > maximum:
            _append_reason(record, "DURATION_ABOVE_MAXIMUM")
        record["duration_filter"] = {
            "minimum_seconds": minimum,
            "maximum_seconds": maximum,
            "accepted": minimum <= duration <= maximum,
        }
        record["exclusion_reasons"].sort()
    return records
