from __future__ import annotations

import os
import tempfile
import wave
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .audio import read_pcm_mono
from .config import PipelineConfig
from .io_utils import atomic_write_csv, atomic_write_json, atomic_write_jsonl
from .splitting import SPLIT_NAMES
from .text import audit_model_text


def _load_cached_embedding(path: Path, expected_size: int) -> list[float] | None:
    if not path.exists():
        return None
    try:
        values = np.load(path, allow_pickle=False).astype(np.float32).reshape(-1)
    except (OSError, ValueError):
        return None
    if values.size != expected_size or not np.isfinite(values).all():
        return None
    return values.tolist()


def _save_embedding(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as handle:
            np.save(handle, values.astype(np.float32), allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class SpeakerEncoder:
    def __init__(self, config: PipelineConfig):
        section = config.section("speecht5")
        from huggingface_hub import snapshot_download
        from speechbrain.inference.speaker import EncoderClassifier

        snapshot = snapshot_download(
            repo_id=section["speaker_encoder_model_id"],
            revision=section["speaker_encoder_revision"],
            local_dir=config.path("cache") / "models" / "speaker_encoder",
        )
        self.encoder = EncoderClassifier.from_hparams(
            source=snapshot, run_opts={"device": "cpu"}
        )

    def encode(self, path: Path) -> np.ndarray:
        import torch

        values, _, _, _ = read_pcm_mono(path)
        waveform = torch.from_numpy(values).unsqueeze(0)
        with torch.no_grad():
            embedding = self.encoder.encode_batch(waveform).squeeze().cpu().numpy()
        embedding = embedding.astype(np.float32).reshape(-1)
        norm = float(np.linalg.norm(embedding))
        return embedding / norm if norm else embedding


def prepare_speecht5(
    records: list[dict[str, Any]], config: PipelineConfig
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    section = config.section("speecht5")
    mode = str(section["mode"])
    if mode not in {"cache_only", "full", "disabled"}:
        raise ValueError(f"Unsupported SpeechT5 mode: {mode}")
    expected_size = int(section["speaker_embedding_size"])
    embedding_dir = config.path("cache") / "speaker_embeddings"
    encoder = None
    encoder_error = None
    if mode == "full":
        try:
            encoder = SpeakerEncoder(config)
        except Exception as exc:
            encoder_error = f"{type(exc).__name__}: {exc}"

    approved = [record for record in records if record.get("split") in SPLIT_NAMES]
    unknown_counter: Counter[str] = Counter()
    audio_validation_failures: list[dict[str, str]] = []
    embedding_counts = Counter()
    for record in approved:
        audit = audit_model_text(record["model_input_text"])
        for character in audit["unsupported_characters"]:
            unknown_counter[character] += 1
        if audit["contains_digit"]:
            unknown_counter["<digit>"] += 1
        record["unsupported_character_audit"] = audit
        path = config.repository_root / record["clean_audio"]
        try:
            with wave.open(str(path), "rb") as audio:
                valid_audio = (
                    audio.getframerate() == 16000
                    and audio.getnchannels() == 1
                    and audio.getsampwidth() == 2
                    and audio.getcomptype() == "NONE"
                    and audio.getnframes() > 0
                )
            if not valid_audio:
                raise ValueError("not readable 16 kHz mono PCM16 WAV")
            record["speecht5_audio_loader_valid"] = True
        except Exception as exc:
            record["speecht5_audio_loader_valid"] = False
            audio_validation_failures.append(
                {
                    "sample_id": record["sample_id"],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

        cache_path = embedding_dir / f"{record['sample_id']}.npy"
        embedding = _load_cached_embedding(cache_path, expected_size)
        status = "available" if embedding is not None else "not_available"
        if embedding is None and encoder is not None:
            try:
                values = encoder.encode(path)
                if values.size != expected_size:
                    raise ValueError(
                        f"expected {expected_size} values, received {values.size}"
                    )
                _save_embedding(cache_path, values)
                embedding = values.tolist()
                status = "available"
            except Exception as exc:
                status = "failed"
                record["speaker_embedding_error"] = f"{type(exc).__name__}: {exc}"
        elif embedding is None and mode == "disabled":
            status = "not_run_disabled"
        elif embedding is None and mode == "cache_only":
            status = "not_run_cache_miss"
        elif embedding is None and encoder_error:
            status = "not_run_dependency_unavailable"
        record["speaker_embedding"] = embedding
        record["speaker_embedding_status"] = status
        record["speaker_embedding_size"] = (
            len(embedding) if embedding is not None else None
        )
        embedding_counts[status] += 1

    processor, processor_error = _load_processor(config, local_only=mode != "full")
    tokenizer_unknowns: Counter[str] = Counter()
    if processor is not None:
        tokenizer = processor.tokenizer
        unk_id = tokenizer.unk_token_id
        for record in approved:
            token_ids = tokenizer(
                record["model_input_text"], add_special_tokens=False
            ).input_ids
            unknown_count = sum(token_id == unk_id for token_id in token_ids)
            record["speecht5_unknown_token_count"] = unknown_count
            if unknown_count:
                tokenizer_unknowns[record["sample_id"]] = unknown_count
    else:
        for record in approved:
            record["speecht5_unknown_token_count"] = None

    batch_validation = _validate_batch(approved, config, processor, processor_error)
    _write_manifests(records, config)
    report = {
        "mode": mode,
        "approved_records": len(approved),
        "processor_model_id": section["processor_model_id"],
        "processor_revision": section["processor_revision"],
        "speaker_encoder_model_id": section["speaker_encoder_model_id"],
        "speaker_encoder_revision": section["speaker_encoder_revision"],
        "speaker_embedding_size": expected_size,
        "speaker_embedding_status_counts": dict(sorted(embedding_counts.items())),
        "speaker_encoder_error": encoder_error,
        "audio_loader_failures": audio_validation_failures,
        "conservative_character_audit": dict(sorted(unknown_counter.items())),
        "processor_audit_status": "completed" if processor is not None else "blocked",
        "processor_error": processor_error,
        "actual_unknown_token_samples": dict(tokenizer_unknowns),
        "batch_validation": batch_validation,
        "training_ready": bool(approved)
        and not audio_validation_failures
        and processor is not None
        and all(
            record.get("speaker_embedding_size") == expected_size for record in approved
        )
        and batch_validation["status"] == "completed",
    }
    atomic_write_json(
        config.path("manifests") / "speecht5_preparation_report.json", report
    )
    return records, report


def _load_processor(config: PipelineConfig, local_only: bool):
    section = config.section("speecht5")
    try:
        from transformers import SpeechT5Processor

        processor = SpeechT5Processor.from_pretrained(
            section["processor_model_id"],
            revision=section["processor_revision"],
            local_files_only=local_only,
        )
        return processor, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _validate_batch(
    approved: list[dict[str, Any]],
    config: PipelineConfig,
    processor: Any,
    processor_error: str | None,
) -> dict[str, Any]:
    batch_size = int(config.section("speecht5")["validation_batch_size"])
    batch = sorted(approved, key=lambda record: record["sample_id"])[:batch_size]
    if not batch:
        return {"status": "blocked", "reason": "no_approved_records"}
    if processor is None:
        return {
            "status": "blocked",
            "reason": "processor_unavailable",
            "detail": processor_error,
        }
    if any(record.get("speaker_embedding") is None for record in batch):
        return {"status": "blocked", "reason": "speaker_embeddings_unavailable"}
    try:
        arrays = [
            read_pcm_mono(config.repository_root / record["clean_audio"])[0]
            for record in batch
        ]
        processed = processor(
            text=[record["model_input_text"] for record in batch],
            audio_target=arrays,
            sampling_rate=16000,
            return_attention_mask=False,
        )
        return {
            "status": "completed",
            "batch_size": len(batch),
            "fields": sorted(processed.keys()),
        }
    except Exception as exc:
        return {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}


def _manifest_row(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": record["sample_id"],
        "audio": record["clean_audio"],
        "text": record["model_input_text"],
        "original_transcript": record["original_transcript"],
        "normalized_transcript": record["normalized_transcript"],
        "speaker_id": record["speaker_id"],
        "speaker_embedding": record.get("speaker_embedding"),
        "speaker_embedding_status": record.get("speaker_embedding_status"),
        "prompt": record["prompt"],
        "medical_terms": record.get("medical_terms", []),
        "duration_seconds": record.get("standardized_audio", {}).get(
            "duration_seconds"
        ),
        "source_split": record.get("source_split"),
        "split": record.get("split"),
        "audio_sha256": record.get("standardized_audio_sha256"),
        "leakage_group_id": record.get("leakage_group_id"),
    }


def _write_manifests(records: list[dict[str, Any]], config: PipelineConfig) -> None:
    manifest_dir = config.path("manifests")
    speech_dir = manifest_dir / "speecht5"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    speech_dir.mkdir(parents=True, exist_ok=True)
    approved_rows = [
        _manifest_row(record)
        for record in records
        if record.get("split") in SPLIT_NAMES
    ]
    atomic_write_jsonl(manifest_dir / "approved.jsonl", approved_rows)
    atomic_write_jsonl(
        manifest_dir / "rejections.jsonl",
        [record for record in records if record.get("exclusion_reasons")],
    )
    atomic_write_jsonl(manifest_dir / "all_records.jsonl", records)
    for split in SPLIT_NAMES:
        rows = [row for row in approved_rows if row["split"] == split]
        atomic_write_jsonl(manifest_dir / f"{split}.jsonl", rows)
        atomic_write_jsonl(speech_dir / f"{split}.jsonl", rows)
        csv_rows = [
            {
                "sample_id": row["sample_id"],
                "audio": row["audio"],
                "text": row["text"],
                "speaker_id": row["speaker_id"],
                "prompt": row["prompt"],
                "duration_seconds": row["duration_seconds"],
                "source_split": row["source_split"],
                "audio_sha256": row["audio_sha256"],
                "leakage_group_id": row["leakage_group_id"],
                "speaker_embedding_status": row["speaker_embedding_status"],
            }
            for row in rows
        ]
        atomic_write_csv(
            manifest_dir / f"{split}.csv",
            csv_rows,
            [
                "sample_id",
                "audio",
                "text",
                "speaker_id",
                "prompt",
                "duration_seconds",
                "source_split",
                "audio_sha256",
                "leakage_group_id",
                "speaker_embedding_status",
            ],
        )
