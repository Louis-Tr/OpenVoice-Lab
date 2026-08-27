from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .asr import screen_asr
from .audio import apply_duration_filter, standardize_audio
from .config import PipelineConfig
from .duplicates import detect_duplicates
from .inventory import inventory_intake
from .io_utils import atomic_write_json, atomic_write_jsonl, read_jsonl
from .quality import screen_quality
from .reporting import generate_reports
from .review import apply_review
from .speecht5 import prepare_speecht5
from .splitting import assign_splits
from .terms import annotate_medical_terms
from .text import process_transcripts
from .validation import validate_records


STAGES = (
    "inventory",
    "validation",
    "standardization",
    "duration",
    "text",
    "quality",
    "duplicates",
    "asr",
    "review",
    "terms",
    "split",
    "speecht5",
    "report",
)


def with_run_name(config: PipelineConfig, run_name: str | None) -> PipelineConfig:
    if not run_name:
        return config
    if not run_name.replace("-", "").replace("_", "").isalnum():
        raise ValueError(
            "run-name must contain only letters, digits, hyphens, and underscores"
        )
    values = copy.deepcopy(config.values)
    for key in (
        "clean_audio",
        "intermediate",
        "manifests",
        "reports",
        "review",
        "cache",
    ):
        values["paths"][key] = str(Path(values["paths"][key]) / run_name).replace(
            "\\", "/"
        )
    values["paths"]["documentation"] = str(
        Path(values["paths"]["reports"]) / "DATASET_PREPARATION.md"
    ).replace("\\", "/")
    return PipelineConfig(config.repository_root, config.config_path, values)


def _stage_dir(config: PipelineConfig, index: int, name: str) -> Path:
    return config.path("intermediate") / f"{index:02d}_{name}"


def _summary(
    records: list[dict[str, Any]], extra: dict[str, Any] | None = None
) -> dict[str, Any]:
    summary = {
        "record_count": len(records),
        "records_with_clean_audio": sum(
            bool(record.get("clean_audio")) for record in records
        ),
        "records_with_exclusions": sum(
            bool(record.get("exclusion_reasons")) for record in records
        ),
        "exclusion_reason_counts": dict(
            sorted(
                Counter(
                    reason
                    for record in records
                    for reason in record.get("exclusion_reasons", [])
                ).items()
            )
        ),
        "review_flag_counts": dict(
            sorted(
                Counter(
                    flag
                    for record in records
                    for flag in record.get("review_flags", [])
                ).items()
            )
        ),
    }
    if extra:
        summary.update(extra)
    return summary


def _write_stage(
    config: PipelineConfig,
    index: int,
    name: str,
    records: list[dict[str, Any]],
    report: dict[str, Any],
) -> None:
    directory = _stage_dir(config, index, name)
    atomic_write_jsonl(directory / "records.jsonl", records)
    atomic_write_json(directory / "stage_report.json", report)
    atomic_write_json(
        directory / "stage_manifest.json",
        {
            "schema_version": 1,
            "stage_index": index,
            "stage": name,
            "config_sha256": config.digest,
            "record_count": len(records),
        },
    )


def _load_previous(config: PipelineConfig, start_index: int) -> list[dict[str, Any]]:
    previous_index = start_index - 1
    previous_name = STAGES[previous_index]
    path = _stage_dir(config, previous_index, previous_name) / "records.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot start at {STAGES[start_index]!r}; prior stage output is missing: {path}"
        )
    return list(read_jsonl(path))


def _load_stage_reports(config: PipelineConfig, before_index: int) -> dict[str, Any]:
    reports = {}
    for index, name in enumerate(STAGES[:before_index]):
        path = _stage_dir(config, index, name) / "stage_report.json"
        if path.exists():
            reports[name] = json.loads(path.read_text(encoding="utf-8"))
    return reports


def run_pipeline(
    config: PipelineConfig,
    *,
    start_stage: str = "inventory",
    end_stage: str = "report",
    limit: int | None = None,
    asr_mode: str | None = None,
) -> dict[str, Any]:
    if start_stage not in STAGES or end_stage not in STAGES:
        raise ValueError(f"Stages must be one of: {', '.join(STAGES)}")
    start_index = STAGES.index(start_stage)
    end_index = STAGES.index(end_stage)
    if start_index > end_index:
        raise ValueError("start-stage must not come after end-stage")
    if limit is not None and start_index != 0:
        raise ValueError("--limit is only valid when starting from inventory")
    records = [] if start_index == 0 else _load_previous(config, start_index)
    stage_reports = _load_stage_reports(config, start_index)

    for index in range(start_index, end_index + 1):
        name = STAGES[index]
        extra: dict[str, Any] = {}
        if name == "inventory":
            records = inventory_intake(config, limit=limit)
        elif name == "validation":
            records = validate_records(records, config)
        elif name == "standardization":
            records = standardize_audio(records, config)
            extra = {
                "status_counts": dict(
                    sorted(
                        Counter(
                            record.get("audio_standardization_status")
                            for record in records
                        ).items()
                    )
                )
            }
        elif name == "duration":
            records = apply_duration_filter(records, config)
        elif name == "text":
            records = process_transcripts(records, config)
        elif name == "quality":
            records = screen_quality(records, config)
        elif name == "duplicates":
            records, extra = detect_duplicates(records, config)
        elif name == "asr":
            records, extra = screen_asr(records, config, mode_override=asr_mode)
        elif name == "review":
            records, extra = apply_review(records, config)
        elif name == "terms":
            records = annotate_medical_terms(records, config)
        elif name == "split":
            records, extra = assign_splits(records, config)
        elif name == "speecht5":
            records, extra = prepare_speecht5(records, config)
        elif name == "report":
            generated_report = generate_reports(records, config, stage_reports)
            extra = {
                "report_counts": generated_report["counts"],
                "limitations": generated_report["limitations"],
                "cleaning_report": str(config.path("reports") / "cleaning_report.json"),
                "inspection_report": str(
                    config.path("reports") / "inspection_report.html"
                ),
            }
        stage_report = _summary(records, extra)
        stage_reports[name] = stage_report
        _write_stage(config, index, name, records, stage_report)

    return {
        "config_sha256": config.digest,
        "start_stage": start_stage,
        "end_stage": end_stage,
        "limit": limit,
        "final": stage_reports[end_stage],
        "generated_paths": {
            key: str(config.path(key))
            for key in ("clean_audio", "intermediate", "manifests", "reports", "review")
        },
    }
