from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Protocol

from .config import PipelineConfig
from .io_utils import atomic_write_json


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.casefold())


def word_error_rate(reference: str, hypothesis: str) -> float:
    reference_words = _words(reference)
    hypothesis_words = _words(hypothesis)
    if not reference_words:
        return 0.0 if not hypothesis_words else 1.0
    previous = list(range(len(hypothesis_words) + 1))
    for reference_index, reference_word in enumerate(reference_words, 1):
        current = [reference_index]
        for hypothesis_index, hypothesis_word in enumerate(hypothesis_words, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[hypothesis_index] + 1,
                    previous[hypothesis_index - 1]
                    + (reference_word != hypothesis_word),
                )
            )
        previous = current
    return previous[-1] / len(reference_words)


class AsrAdapter(Protocol):
    def transcribe(self, audio_path: Path) -> str: ...


class TransformersWhisperAdapter:
    def __init__(self, model_id: str, revision: str, device: str):
        from transformers import pipeline

        device_index = -1 if device == "cpu" else 0
        self.pipeline = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            revision=revision,
            device=device_index,
        )

    def transcribe(self, audio_path: Path) -> str:
        result = self.pipeline(str(audio_path))
        return str(result["text"]).strip()


def _apply_alignment(record: dict[str, Any], transcript: str, threshold: float) -> None:
    wer = word_error_rate(record.get("normalized_transcript", ""), transcript)
    record["asr_transcript"] = transcript
    record["asr_wer"] = wer
    record["asr_failure"] = None
    record["asr_status"] = "aligned"
    record["alignment_status"] = "high_mismatch" if wer > threshold else "aligned"
    if wer > threshold and "ASR_HIGH_MISMATCH_REVIEW" not in record["review_flags"]:
        record["review_flags"].append("ASR_HIGH_MISMATCH_REVIEW")


def screen_asr(
    records: list[dict[str, Any]],
    config: PipelineConfig,
    mode_override: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    section = config.section("asr")
    mode = mode_override or str(section["mode"])
    if mode not in {"disabled", "cache_only", "full"}:
        raise ValueError(f"Unsupported ASR mode: {mode}")
    cache_dir = config.path("cache") / "asr"
    cache_dir.mkdir(parents=True, exist_ok=True)
    threshold = float(section["high_wer_threshold"])
    adapter: AsrAdapter | None = None
    adapter_error: str | None = None
    if mode == "full":
        try:
            adapter = TransformersWhisperAdapter(
                str(section["model_id"]),
                str(section["model_revision"]),
                str(section["device"]),
            )
        except Exception as exc:
            adapter_error = f"{type(exc).__name__}: {exc}"

    counts = {
        "eligible": 0,
        "aligned": 0,
        "high_mismatch": 0,
        "cache_entries": 0,
        "not_run": 0,
        "failures": 0,
        "skipped_prior_exclusion": 0,
    }
    for record in records:
        if record.get("exclusion_reasons"):
            record.update(
                {
                    "asr_transcript": None,
                    "asr_wer": None,
                    "asr_failure": None,
                    "asr_status": "skipped_prior_exclusion",
                    "alignment_status": "not_run",
                }
            )
            counts["skipped_prior_exclusion"] += 1
            continue
        counts["eligible"] += 1
        cache_path = cache_dir / f"{record['sample_id']}.json"
        if cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if (
                    cached.get("source_audio_sha256")
                    == record.get("standardized_audio_sha256")
                    and cached.get("model_revision") == section["model_revision"]
                    and isinstance(cached.get("transcript"), str)
                ):
                    _apply_alignment(record, cached["transcript"], threshold)
                    counts["cache_entries"] += 1
                    counts[
                        "high_mismatch"
                        if record["alignment_status"] == "high_mismatch"
                        else "aligned"
                    ] += 1
                    continue
            except (OSError, ValueError, json.JSONDecodeError):
                pass

        if mode == "disabled":
            record.update(
                {
                    "asr_transcript": None,
                    "asr_wer": None,
                    "asr_failure": "disabled_by_configuration",
                    "asr_status": "not_run_disabled",
                    "alignment_status": "not_run",
                }
            )
            counts["not_run"] += 1
        elif mode == "cache_only":
            record.update(
                {
                    "asr_transcript": None,
                    "asr_wer": None,
                    "asr_failure": "cache_miss_full_asr_not_requested",
                    "asr_status": "not_run_cache_miss",
                    "alignment_status": "not_run",
                }
            )
            counts["not_run"] += 1
        elif adapter is None:
            record.update(
                {
                    "asr_transcript": None,
                    "asr_wer": None,
                    "asr_failure": adapter_error,
                    "asr_status": "not_run_dependency_unavailable",
                    "alignment_status": "not_run",
                }
            )
            counts["not_run"] += 1
        else:
            try:
                transcript = adapter.transcribe(
                    config.repository_root / record["clean_audio"]
                )
                _apply_alignment(record, transcript, threshold)
                atomic_write_json(
                    cache_path,
                    {
                        "sample_id": record["sample_id"],
                        "source_audio_sha256": record["standardized_audio_sha256"],
                        "model_id": section["model_id"],
                        "model_revision": section["model_revision"],
                        "transcript": transcript,
                    },
                )
                counts["cache_entries"] += 1
                counts[
                    "high_mismatch"
                    if record["alignment_status"] == "high_mismatch"
                    else "aligned"
                ] += 1
            except Exception as exc:
                record.update(
                    {
                        "asr_transcript": None,
                        "asr_wer": None,
                        "asr_failure": f"{type(exc).__name__}: {exc}",
                        "asr_status": "failed",
                        "alignment_status": "not_run",
                    }
                )
                counts["failures"] += 1
        record["review_flags"].sort()
    return records, {
        "mode": mode,
        "adapter": section["adapter"],
        "model_id": section["model_id"],
        "model_revision": section["model_revision"],
        "adapter_initialization_error": adapter_error,
        "high_wer_threshold": threshold,
        "counts": counts,
        "interpretation": "not_run records do not contain fabricated ASR transcripts or WER values",
    }
