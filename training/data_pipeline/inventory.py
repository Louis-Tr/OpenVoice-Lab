from __future__ import annotations

import csv
import hashlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

from .config import PipelineConfig
from .io_utils import (
    atomic_write_json,
    atomic_write_jsonl,
    relative_posix,
    sha256_file,
    stable_hash,
)


REQUIRED_COLUMNS = {
    "audio_clipping",
    "audio_clipping:confidence",
    "background_noise_audible",
    "background_noise_audible:confidence",
    "overall_quality_of_the_audio",
    "quiet_speaker",
    "quiet_speaker:confidence",
    "speaker_id",
    "file_name",
    "phrase",
    "prompt",
    "writer_id",
}


def _hash_paths(paths: Iterable[Path], workers: int) -> dict[Path, str]:
    ordered = sorted(paths, key=lambda item: item.as_posix().casefold())
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        hashes = executor.map(sha256_file, ordered)
        return dict(zip(ordered, hashes, strict=True))


def _tree_digest(root: Path, paths: list[Path], hashes: dict[Path, str]) -> str:
    digest = hashlib.sha256()
    for path in sorted(
        paths, key=lambda item: item.relative_to(root).as_posix().casefold()
    ):
        relative = path.relative_to(root).as_posix()
        digest.update(
            f"{relative}\0{path.stat().st_size}\0{hashes[path]}\n".encode("utf-8")
        )
    return digest.hexdigest()


def _metadata_snapshot(root: Path, paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(
        paths, key=lambda item: item.relative_to(root).as_posix().casefold()
    ):
        stat = path.stat()
        digest.update(
            f"{path.relative_to(root).as_posix()}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode()
        )
    return digest.hexdigest()


def _read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        missing = sorted(REQUIRED_COLUMNS - set(columns))
        if missing:
            raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")
        return [dict(row) for row in reader], columns


def _copy_inventory(
    root: Path,
    csv_name: str,
    recordings_directory: str,
    workers: int,
    hash_files: bool,
) -> tuple[dict[str, Any], dict[Path, str]]:
    csv_path = root / csv_name
    wav_root = root / recordings_directory
    wav_paths = sorted(
        wav_root.rglob("*.wav"), key=lambda path: path.as_posix().casefold()
    )
    all_paths = ([csv_path] if csv_path.exists() else []) + wav_paths
    hashes = _hash_paths(all_paths, workers) if hash_files else {}
    split_counts = Counter(path.parent.name for path in wav_paths)
    inventory = {
        "root": str(root),
        "exists": root.exists(),
        "csv_exists": csv_path.is_file(),
        "csv_sha256": hashes.get(csv_path),
        "wav_count": len(wav_paths),
        "total_wav_bytes": sum(path.stat().st_size for path in wav_paths),
        "split_counts": dict(sorted(split_counts.items())),
        "tree_sha256": _tree_digest(root, all_paths, hashes)
        if hash_files and all_paths
        else None,
        "metadata_snapshot_sha256": _metadata_snapshot(root, all_paths)
        if all_paths
        else None,
    }
    return inventory, hashes


def inventory_intake(
    config: PipelineConfig, limit: int | None = None
) -> list[dict[str, Any]]:
    intake = config.section("intake")
    workers = int(config.section("audio")["workers"])
    raw_root = config.path("raw_root")
    duplicate_root = config.path("duplicate_raw_root")
    csv_name = intake["csv_name"]
    recordings_directory = intake["recordings_directory"]
    if not raw_root.is_dir():
        raise FileNotFoundError(f"Canonical raw root does not exist: {raw_root}")

    canonical, canonical_hashes = _copy_inventory(
        raw_root, csv_name, recordings_directory, workers, hash_files=True
    )
    rows, columns = _read_csv(raw_root / csv_name)
    wav_paths = sorted(
        (raw_root / recordings_directory).rglob("*.wav"),
        key=lambda path: path.as_posix().casefold(),
    )
    wav_by_name: dict[str, list[Path]] = {}
    for wav_path in wav_paths:
        wav_by_name.setdefault(wav_path.name, []).append(wav_path)

    filename_counts = Counter(row.get("file_name", "") for row in rows)
    records: list[dict[str, Any]] = []
    for row in rows:
        filename = row.get("file_name", "").strip()
        candidates = wav_by_name.get(filename, [])
        source_wav = candidates[0] if len(candidates) == 1 else None
        identity = "\0".join(
            [filename.casefold(), row.get("speaker_id", ""), row.get("phrase", "")]
        )
        record = {
            "sample_id": f"medical-{stable_hash(identity, 20)}",
            "source_csv": relative_posix(raw_root / csv_name, config.repository_root),
            "source_wav": relative_posix(source_wav, config.repository_root)
            if source_wav
            else None,
            "source_wav_sha256": canonical_hashes.get(source_wav)
            if source_wav
            else None,
            "source_wav_bytes": source_wav.stat().st_size if source_wav else None,
            "source_split": source_wav.parent.name if source_wav else None,
            "file_name": filename,
            "speaker_id": row.get("speaker_id", "").strip(),
            "writer_id": row.get("writer_id", "").strip(),
            "prompt": row.get("prompt", "").strip(),
            "original_transcript": row.get("phrase", ""),
            "source_metadata": {column: row.get(column, "") for column in columns},
            "intake_issues": [],
            "exclusion_reasons": [],
            "review_flags": [],
        }
        if not filename:
            record["intake_issues"].append("CSV_FILENAME_EMPTY")
        elif filename_counts[filename] > 1:
            record["intake_issues"].append("CSV_FILENAME_DUPLICATE")
        if not candidates:
            record["intake_issues"].append("WAV_MISSING")
        elif len(candidates) > 1:
            record["intake_issues"].append("WAV_FILENAME_AMBIGUOUS")
        records.append(record)

    selected = (
        records
        if limit is None
        else sorted(records, key=lambda row: row["sample_id"])[:limit]
    )
    duplicate: dict[str, Any] = {
        "root": str(duplicate_root),
        "exists": duplicate_root.exists(),
    }
    duplicate_hashes: dict[Path, str] = {}
    if duplicate_root.exists():
        duplicate, duplicate_hashes = _copy_inventory(
            duplicate_root,
            csv_name,
            recordings_directory,
            workers,
            hash_files=bool(intake.get("hash_duplicate_copy", True)) and limit is None,
        )
        duplicate_wavs = sorted(
            (duplicate_root / recordings_directory).rglob("*.wav"),
            key=lambda path: path.as_posix().casefold(),
        )
        duplicate_by_relative = {
            path.relative_to(duplicate_root / recordings_directory).as_posix(): path
            for path in duplicate_wavs
        }
        canonical_by_relative = {
            path.relative_to(raw_root / recordings_directory).as_posix(): path
            for path in wav_paths
        }
        common = sorted(set(canonical_by_relative) & set(duplicate_by_relative))
        hash_mismatches = []
        if duplicate_hashes:
            hash_mismatches = [
                relative
                for relative in common
                if canonical_hashes[canonical_by_relative[relative]]
                != duplicate_hashes[duplicate_by_relative[relative]]
            ]
        duplicate["comparison"] = {
            "canonical_only_paths": len(
                set(canonical_by_relative) - set(duplicate_by_relative)
            ),
            "duplicate_only_paths": len(
                set(duplicate_by_relative) - set(canonical_by_relative)
            ),
            "common_paths": len(common),
            "hash_comparison_performed": bool(duplicate_hashes),
            "hash_mismatch_count": len(hash_mismatches) if duplicate_hashes else None,
            "hash_mismatch_examples": hash_mismatches[:20],
        }

    csv_names = {row.get("file_name", "").strip() for row in rows}
    wav_names = set(wav_by_name)
    inventory = {
        "schema_version": 1,
        "config_sha256": config.digest,
        "processing_input": str(raw_root),
        "canonical_root_selected": True,
        "canonical": canonical,
        "duplicate_copy": duplicate,
        "archive": _archive_inventory(config, workers, limit),
        "csv": {
            "row_count": len(rows),
            "selected_row_count": len(selected),
            "expected_row_count": int(intake["expected_rows"]),
            "columns": columns,
            "unique_filenames": len(csv_names),
            "unique_speakers": len({row.get("speaker_id", "") for row in rows}),
            "unique_writers": len({row.get("writer_id", "") for row in rows}),
            "unique_phrases": len({row.get("phrase", "") for row in rows}),
            "unique_prompts": len({row.get("prompt", "") for row in rows}),
        },
        "relationships": {
            "csv_without_wav": len(csv_names - wav_names),
            "wav_without_csv": len(wav_names - csv_names),
            "ambiguous_wav_filenames": sum(
                len(paths) > 1 for paths in wav_by_name.values()
            ),
            "duplicate_csv_filenames": sum(
                count > 1 for count in filename_counts.values()
            ),
        },
    }
    output_dir = config.path("intermediate") / "00_inventory"
    atomic_write_json(output_dir / "intake_inventory.json", inventory)
    atomic_write_jsonl(output_dir / "records.jsonl", selected)
    return selected


def _archive_inventory(
    config: PipelineConfig, workers: int, limit: int | None
) -> dict[str, Any]:
    archive = config.path("archive")
    exists = archive.is_file()
    should_hash = (
        bool(config.section("intake").get("hash_archive", True)) and limit is None
    )
    return {
        "path": str(archive),
        "exists": exists,
        "bytes": archive.stat().st_size if exists else None,
        "sha256": sha256_file(archive) if exists and should_hash else None,
        "hash_skipped_for_subset": exists and not should_hash,
    }


def current_metadata_snapshot(config: PipelineConfig, root_key: str) -> str | None:
    root = config.path(root_key)
    if not root.exists():
        return None
    intake = config.section("intake")
    paths = [root / intake["csv_name"]]
    paths.extend(sorted((root / intake["recordings_directory"]).rglob("*.wav")))
    paths = [path for path in paths if path.exists()]
    return _metadata_snapshot(root, paths)
