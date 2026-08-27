from __future__ import annotations

import html
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .audio import read_pcm_mono
from .config import PipelineConfig
from .inventory import current_metadata_snapshot
from .io_utils import (
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_text,
    stable_hash,
)
from .splitting import SPLIT_NAMES


def _quality_level(record: dict[str, Any]) -> str:
    rms = record.get("quality_metrics", {}).get("rms_dbfs")
    if rms is None:
        return "unavailable"
    if rms < -30:
        return "quiet"
    if rms > -16:
        return "loud"
    return "typical"


def _duration_bin(duration: float | None, bins: list[float]) -> str:
    if duration is None:
        return "unavailable"
    for lower, upper in zip(bins, bins[1:], strict=True):
        if lower <= duration < upper:
            return f"{lower:g}-{upper:g}s"
    return f">={bins[-1]:g}s"


def _inspection_sample(
    approved: list[dict[str, Any]], size: int, seed: int, bins: list[float]
) -> list[dict[str, Any]]:
    remaining = sorted(
        approved,
        key=lambda record: stable_hash(f"{seed}:{record['sample_id']}:inspection"),
    )
    selected: list[dict[str, Any]] = []
    seen_prompts: set[str] = set()
    seen_speakers: set[str] = set()
    seen_durations: set[str] = set()
    seen_quality: set[str] = set()
    while remaining and len(selected) < size:

        def score(record: dict[str, Any]) -> tuple[int, str]:
            duration = record.get("standardized_audio", {}).get("duration_seconds")
            duration_label = _duration_bin(duration, bins)
            quality_label = _quality_level(record)
            coverage = (
                8 * (record.get("prompt") not in seen_prompts)
                + 4 * (record.get("speaker_id") not in seen_speakers)
                + 3 * (duration_label not in seen_durations)
                + 3 * (quality_label not in seen_quality)
            )
            return coverage, stable_hash(f"{seed}:{record['sample_id']}")

        candidate = max(remaining, key=score)
        remaining.remove(candidate)
        selected.append(candidate)
        seen_prompts.add(candidate.get("prompt", ""))
        seen_speakers.add(candidate.get("speaker_id", ""))
        duration = candidate.get("standardized_audio", {}).get("duration_seconds")
        seen_durations.add(_duration_bin(duration, bins))
        seen_quality.add(_quality_level(candidate))
    return selected


def _rejection_examples(
    records: list[dict[str, Any]], per_category: int, seed: int
) -> dict[str, list[dict[str, Any]]]:
    by_reason: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        for reason in record.get("exclusion_reasons", []):
            by_reason[reason].append(record)
    return {
        reason: sorted(
            grouped,
            key=lambda record: stable_hash(f"{seed}:{reason}:{record['sample_id']}"),
        )[:per_category]
        for reason, grouped in sorted(by_reason.items())
    }


def _distribution(records: list[dict[str, Any]], bins: list[float]) -> dict[str, Any]:
    return {
        "prompts": dict(
            sorted(Counter(record.get("prompt", "") for record in records).items())
        ),
        "speakers": dict(
            sorted(Counter(record.get("speaker_id", "") for record in records).items())
        ),
        "medical_term_categories": dict(
            sorted(
                Counter(
                    category
                    for record in records
                    for category in record.get("medical_term_categories", [])
                ).items()
            )
        ),
        "duration_bins": dict(
            sorted(
                Counter(
                    _duration_bin(
                        record.get("standardized_audio", {}).get("duration_seconds"),
                        bins,
                    )
                    for record in records
                ).items()
            )
        ),
        "quality_levels": dict(
            sorted(Counter(_quality_level(record) for record in records).items())
        ),
    }


def generate_reports(
    records: list[dict[str, Any]],
    config: PipelineConfig,
    stage_reports: dict[str, Any],
) -> dict[str, Any]:
    report_dir = config.path("reports")
    asset_dir = report_dir / "inspection_assets"
    asset_dir.mkdir(parents=True, exist_ok=True)
    inspection = config.section("inspection")
    bins = [float(value) for value in inspection["duration_bins_seconds"]]
    approved = [record for record in records if record.get("split") in SPLIT_NAMES]
    rejected = [record for record in records if record.get("exclusion_reasons")]
    sample = _inspection_sample(
        approved, int(inspection["accepted_sample_size"]), config.seed, bins
    )
    rejection_examples = _rejection_examples(
        records, int(inspection["rejection_examples_per_category"]), config.seed
    )
    preview_records = {record["sample_id"]: record for record in sample}
    for examples in rejection_examples.values():
        for record in examples:
            if record.get("clean_audio"):
                preview_records[record["sample_id"]] = record
    preview_paths: dict[str, str] = {}
    for sample_id, record in sorted(preview_records.items()):
        preview_path = asset_dir / f"{sample_id}.svg"
        try:
            values, sample_rate, _, _ = read_pcm_mono(
                config.repository_root / record["clean_audio"]
            )
            atomic_write_text(
                preview_path, _waveform_spectrogram_svg(values, sample_rate)
            )
            preview_paths[sample_id] = os.path.relpath(
                preview_path, report_dir
            ).replace("\\", "/")
        except Exception:
            continue

    rejection_counts = Counter(
        reason for record in rejected for reason in record.get("exclusion_reasons", [])
    )
    inventory = _load_json(
        config.path("intermediate") / "00_inventory" / "intake_inventory.json"
    )
    canonical_unchanged = inventory.get("canonical", {}).get(
        "metadata_snapshot_sha256"
    ) == current_metadata_snapshot(config, "raw_root")
    duplicate_unchanged = inventory.get("duplicate_copy", {}).get(
        "metadata_snapshot_sha256"
    ) == current_metadata_snapshot(config, "duplicate_raw_root")
    report = {
        "schema_version": 1,
        "config_sha256": config.digest,
        "seed": config.seed,
        "counts": {
            "input_records": len(records),
            "standardized_audio": sum(
                bool(record.get("clean_audio")) for record in records
            ),
            "approved": len(approved),
            "rejected_or_pending_review": len(rejected),
            "pending_manual_review": sum(
                record.get("review_status") == "pending" for record in records
            ),
            "asr_aligned": sum(
                record.get("alignment_status") == "aligned" for record in records
            ),
            "asr_high_mismatch": sum(
                record.get("alignment_status") == "high_mismatch" for record in records
            ),
        },
        "rejection_reason_counts": dict(sorted(rejection_counts.items())),
        "split_counts": dict(
            sorted(Counter(record["split"] for record in approved).items())
        ),
        "distributions": _distribution(approved, bins),
        "inspection": {
            "accepted_sample_size": len(sample),
            "accepted_sample_ids": [record["sample_id"] for record in sample],
            "rejection_examples": {
                reason: [record["sample_id"] for record in examples]
                for reason, examples in rejection_examples.items()
            },
            "preview_paths": preview_paths,
        },
        "intake": inventory,
        "immutability_check": {
            "method": "path/size/mtime-nanosecond metadata snapshot after byte hashes were inventoried",
            "canonical_metadata_unchanged": canonical_unchanged,
            "duplicate_metadata_unchanged": duplicate_unchanged,
        },
        "stage_reports": stage_reports,
        "limitations": _limitations(stage_reports),
        "fabricated_results": False,
    }
    atomic_write_json(report_dir / "cleaning_report.json", report)
    atomic_write_jsonl(
        report_dir / "inspection_sample.jsonl", [_report_record(row) for row in sample]
    )
    atomic_write_json(
        report_dir / "rejection_examples.json",
        {
            reason: [_report_record(record) for record in examples]
            for reason, examples in rejection_examples.items()
        },
    )
    atomic_write_text(report_dir / "cleaning_report.md", _markdown_report(report))
    atomic_write_text(
        report_dir / "inspection_report.html",
        _inspection_html(
            report, sample, rejection_examples, preview_paths, report_dir, config
        ),
    )
    atomic_write_text(
        config.path("documentation"), _dataset_documentation(report, config)
    )
    return report


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _limitations(stage_reports: dict[str, Any]) -> list[str]:
    limitations = []
    asr = stage_reports.get("asr", {})
    if asr.get("counts", {}).get("not_run", 0):
        limitations.append(
            "ASR alignment was not run for eligible cache misses; ASR transcript and WER remain null."
        )
    speecht5 = stage_reports.get("speecht5", {})
    if not speecht5.get("training_ready", False):
        limitations.append(
            "SpeechT5 model preprocessing is not training-ready; see the processor, embedding, and batch statuses."
        )
    review = stage_reports.get("review", {})
    if review.get("accepted_without_review_count", 0):
        limitations.append(
            f"Manual review was explicitly skipped; {review['accepted_without_review_count']} "
            "flagged records were accepted without reviewer decisions while hard exclusions remained active."
        )
    if review.get("pending_count", 0):
        limitations.append(
            "Pending manual-review records are excluded from approved training manifests."
        )
    return limitations


def _report_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": record["sample_id"],
        "audio": record.get("clean_audio"),
        "speaker_id": record.get("speaker_id"),
        "prompt": record.get("prompt"),
        "source_split": record.get("source_split"),
        "split": record.get("split"),
        "duration_seconds": record.get("standardized_audio", {}).get(
            "duration_seconds"
        ),
        "quality_metrics": record.get("quality_metrics"),
        "original_transcript": record.get("original_transcript"),
        "normalized_transcript": record.get("normalized_transcript"),
        "model_input_text": record.get("model_input_text"),
        "asr_transcript": record.get("asr_transcript"),
        "asr_wer": record.get("asr_wer"),
        "review_flags": record.get("review_flags"),
        "review_status": record.get("review_status"),
        "exclusion_reasons": record.get("exclusion_reasons"),
        "medical_terms": record.get("medical_terms"),
    }


def _waveform_spectrogram_svg(values: np.ndarray, sample_rate: int) -> str:
    width, height = 900, 420
    waveform_height = 150
    if values.size == 0:
        raise ValueError("empty audio")
    points_count = min(width, values.size)
    edges = np.linspace(0, values.size, points_count + 1, dtype=int)
    peaks = [
        float(
            np.max(
                np.abs(values[edges[index] : max(edges[index + 1], edges[index] + 1)])
            )
        )
        for index in range(points_count)
    ]
    points = " ".join(
        f"{index * width / max(points_count - 1, 1):.1f},{75 - peak * 68:.1f} "
        f"{index * width / max(points_count - 1, 1):.1f},{75 + peak * 68:.1f}"
        for index, peak in enumerate(peaks)
    )
    window_size = 512
    hop = 256
    if values.size < window_size:
        padded = np.pad(values, (0, window_size - values.size))
    else:
        padded = values
    frame_starts = np.arange(0, max(1, padded.size - window_size + 1), hop)
    if frame_starts.size > 120:
        frame_starts = frame_starts[
            np.linspace(0, frame_starts.size - 1, 120).astype(int)
        ]
    frames = np.stack([padded[start : start + window_size] for start in frame_starts])
    spectrum = np.abs(np.fft.rfft(frames * np.hanning(window_size), axis=1)).T
    spectrum = 20 * np.log10(np.maximum(spectrum, 1e-8))
    spectrum = spectrum[: min(96, spectrum.shape[0])]
    floor, ceiling = np.percentile(spectrum, [5, 99])
    normalized = np.clip((spectrum - floor) / max(ceiling - floor, 1e-6), 0, 1)
    rectangles = []
    cell_width = width / normalized.shape[1]
    cell_height = (height - waveform_height - 30) / normalized.shape[0]
    for frequency_index in range(normalized.shape[0]):
        y = height - (frequency_index + 1) * cell_height
        for time_index in range(normalized.shape[1]):
            value = float(normalized[frequency_index, time_index])
            hue = 250 - 220 * value
            light = 8 + 58 * value
            rectangles.append(
                f"<rect x='{time_index * cell_width:.2f}' y='{y:.2f}' width='{cell_width + 0.2:.2f}' height='{cell_height + 0.2:.2f}' fill='hsl({hue:.0f} 85% {light:.0f}%)'/>"
            )
    duration = values.size / sample_rate
    return (
        f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {width} {height}' role='img'>"
        f"<rect width='{width}' height='{height}' fill='white'/><text x='8' y='18' font-family='sans-serif' font-size='14'>"
        f"Waveform and spectrogram · {duration:.3f}s · {sample_rate} Hz</text>"
        f"<polyline points='{points}' stroke='#1769aa' stroke-width='1' fill='none'/><g>{''.join(rectangles)}</g></svg>"
    )


def _markdown_report(report: dict[str, Any]) -> str:
    counts = report["counts"]
    rejection_rows = (
        "\n".join(
            f"| `{reason}` | {count} |"
            for reason, count in report["rejection_reason_counts"].items()
        )
        or "| None | 0 |"
    )
    split_rows = "\n".join(
        f"| {name} | {report['split_counts'].get(name, 0)} |" for name in SPLIT_NAMES
    )
    limitation_rows = (
        "\n".join(f"- {item}" for item in report["limitations"]) or "- None."
    )
    return f"""# Medical TTS cleaning report

This report contains deterministic measurements from the local pipeline run. No ASR, reviewer decision, speaker embedding, or training result is inferred when it did not run.

## Counts

| Measure | Count |
| --- | ---: |
| Canonical input records | {counts["input_records"]} |
| Standardized WAV files | {counts["standardized_audio"]} |
| Approved records | {counts["approved"]} |
| Rejected or pending review | {counts["rejected_or_pending_review"]} |
| Pending manual review | {counts["pending_manual_review"]} |
| ASR aligned | {counts["asr_aligned"]} |

## Approved splits

| Split | Records |
| --- | ---: |
{split_rows}

## Exclusion provenance

| Reason code | Records |
| --- | ---: |
{rejection_rows}

## Leakage assertions

```json
{json.dumps(report["stage_reports"].get("split", {}).get("leakage_assertions", {}), indent=2)}
```

## Honest limitations

{limitation_rows}

Open `inspection_report.html` beside this report for the seeded audio inspection with waveform/spectrogram previews.
"""


def _inspection_html(
    report: dict[str, Any],
    sample: list[dict[str, Any]],
    rejection_examples: dict[str, list[dict[str, Any]]],
    previews: dict[str, str],
    report_dir: Path,
    config: PipelineConfig,
) -> str:
    def card(record: dict[str, Any]) -> str:
        audio = ""
        if record.get("clean_audio"):
            audio_path = os.path.relpath(
                config.repository_root / record["clean_audio"], report_dir
            ).replace("\\", "/")
            audio = f"<audio controls preload='none' src='{html.escape(audio_path)}'></audio>"
        preview = previews.get(record["sample_id"])
        image = (
            f"<img loading='lazy' src='{html.escape(preview)}' alt='waveform and spectrogram'>"
            if preview
            else ""
        )
        return (
            f"<article><h3>{html.escape(record['sample_id'])}</h3>{audio}{image}"
            f"<p><b>Original:</b> {html.escape(record.get('original_transcript') or '')}</p>"
            f"<p><b>Normalized:</b> {html.escape(record.get('normalized_transcript') or '')}</p>"
            f"<p><b>Model input:</b> {html.escape(record.get('model_input_text') or '')}</p>"
            f"<p><b>Prompt / speaker / duration:</b> {html.escape(record.get('prompt') or '')} / "
            f"{html.escape(record.get('speaker_id') or '')} / {record.get('standardized_audio', {}).get('duration_seconds')}</p>"
            f"<pre>{html.escape(json.dumps({'quality': record.get('quality_metrics'), 'ASR': {'status': record.get('asr_status'), 'text': record.get('asr_transcript'), 'WER': record.get('asr_wer')}, 'terms': record.get('medical_terms'), 'exclusions': record.get('exclusion_reasons')}, indent=2))}</pre></article>"
        )

    accepted_html = "\n".join(card(record) for record in sample)
    rejected_html = "\n".join(
        f"<section><h2>{html.escape(reason)} ({len(examples)} examples)</h2>"
        + "\n".join(card(record) for record in examples)
        + "</section>"
        for reason, examples in rejection_examples.items()
    )
    return f"""<!doctype html><meta charset='utf-8'><title>Medical TTS inspection</title>
<style>body{{font:15px system-ui;max-width:1200px;margin:2rem auto;color:#17202a}}article{{border:1px solid #ccd1d1;border-radius:8px;padding:1rem;margin:1rem 0}}img{{width:100%;max-height:420px}}pre{{white-space:pre-wrap;background:#f4f6f7;padding:.7rem;overflow:auto}}table{{border-collapse:collapse}}td,th{{border:1px solid #bbb;padding:.35rem}}</style>
<h1>Medical TTS genuine light inspection</h1>
<p>Generated from {report["counts"]["input_records"]} actual canonical records with seed {report["seed"]}. Approved: {report["counts"]["approved"]}; rejected/pending: {report["counts"]["rejected_or_pending_review"]}.</p>
<h2>Run summary</h2><pre>{html.escape(json.dumps({"counts": report["counts"], "splits": report["split_counts"], "rejections": report["rejection_reason_counts"], "leakage": report["stage_reports"].get("split", {}).get("leakage_assertions"), "unknown_token_audit": report["stage_reports"].get("speecht5", {})}, indent=2))}</pre>
<h2>Seeded accepted sample ({len(sample)})</h2>{accepted_html}
<h1>Rejection/review examples</h1>{rejected_html}
"""


def _dataset_documentation(report: dict[str, Any], config: PipelineConfig) -> str:
    counts = report["counts"]
    canonical = report["intake"].get("canonical", {})
    duplicate = report["intake"].get("duplicate_copy", {})
    split = report["stage_reports"].get("split", {})
    asr = report["stage_reports"].get("asr", {})
    speech = report["stage_reports"].get("speecht5", {})
    review = report["stage_reports"].get("review", {})
    if review.get("accepted_without_review_count", 0):
        review_policy = (
            f"Manual review was explicitly skipped for this run. "
            f"{review['accepted_without_review_count']} flagged records were accepted "
            "without fabricated reviewer actions; hard validation and audio-quality "
            "exclusions remained active."
        )
    elif review.get("pending_count", 0):
        review_policy = "Pending high-risk records are excluded from approved manifests."
    else:
        review_policy = "No unresolved manual-review records remain."
    rejection_lines = (
        "\n".join(
            f"- `{reason}`: {count}"
            for reason, count in report["rejection_reason_counts"].items()
        )
        or "- None"
    )
    return f"""# Dataset preparation

This document is generated from the actual Stage 11 Part 1 medical-speech pipeline results. Rerun it from the repository root with:

```powershell
python -m training.data_pipeline.run --config training/config/dataset.yaml
```

## Immutable intake

- Processing input: `{report["intake"].get("processing_input")}`
- Canonical WAV inventory: {canonical.get("wav_count")} files, {canonical.get("total_wav_bytes")} bytes
- Canonical tree SHA-256: `{canonical.get("tree_sha256")}`
- Canonical CSV SHA-256: `{canonical.get("csv_sha256")}`
- Duplicate raw root detected: {duplicate.get("exists")}; WAV files: {duplicate.get("wav_count")}
- Duplicate tree SHA-256: `{duplicate.get("tree_sha256")}`
- Archive SHA-256: `{report["intake"].get("archive", {}).get("sha256")}`
- Canonical metadata unchanged during execution: {report["immutability_check"]["canonical_metadata_unchanged"]}
- Duplicate metadata unchanged during execution: {report["immutability_check"]["duplicate_metadata_unchanged"]}

The pipeline reads only the configured canonical upper-case root. It does not move, rewrite, delete, or deduplicate either raw copy or the archive.

## Measured outcome

- Input records: {counts["input_records"]}
- Readable standardized 16 kHz mono PCM WAVs: {counts["standardized_audio"]}
- Approved records: {counts["approved"]}
- Rejected or pending review: {counts["rejected_or_pending_review"]}
- Pending manual review: {counts["pending_manual_review"]}
- Split counts: `{json.dumps(report["split_counts"], sort_keys=True)}`
- Leakage assertions: `{json.dumps(split.get("leakage_assertions", {}), sort_keys=True)}`

## Exclusion provenance

{rejection_lines}

Every non-approved row remains in `data-processing/manifests/medical_tts/rejections.jsonl` and `all_records.jsonl` with reason codes and source provenance.

## ASR and review status

- ASR mode: `{asr.get("mode")}`
- Pinned ASR revision: `{asr.get("model_revision")}`
- ASR counts: `{json.dumps(asr.get("counts", {}), sort_keys=True)}`
- Manual actions fabricated: `false`
- Accepted without review: `{review.get("accepted_without_review_count", 0)}`

No ASR transcript or WER is populated when ASR did not execute. High WER is a review flag only. {review_policy} Reviewer actions are append-only in the ignored review directory.

## SpeechT5 preparation status

- Processor revision: `{speech.get("processor_revision")}`
- Speaker encoder revision: `{speech.get("speaker_encoder_revision")}`
- Processor audit: `{speech.get("processor_audit_status")}`
- Speaker embedding statuses: `{json.dumps(speech.get("speaker_embedding_status_counts", {}), sort_keys=True)}`
- One-batch validation: `{json.dumps(speech.get("batch_validation", {}), sort_keys=True)}`
- Training-ready: `{speech.get("training_ready")}`

The split JSONL/CSV files and `speecht5/` JSONL representation are generated even when external processor or 512-element embeddings are unavailable. A blocked status is not represented as a successful model-preprocessing result.

## Generated artifacts

- `data-processing/intermediate/medical_tts/`: stage manifests and intake hashes
- `data-processing/clean_audio/medical_16khz/`: standardized WAVs
- `data-processing/manifests/medical_tts/`: approved splits, complete records, rejections, and SpeechT5 representation
- `data-processing/review/medical_tts/`: append-only review interface and queue
- `data-processing/reports/medical_tts/`: JSON, Markdown, and HTML inspection reports

All generated data paths are Git-ignored. Source, configuration, tests, this measured document, and the thin inspection notebook remain trackable.
"""
