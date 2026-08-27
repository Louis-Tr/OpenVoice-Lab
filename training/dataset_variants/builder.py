from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from collections import Counter
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from training.data_pipeline.io_utils import (
    atomic_write_json,
    atomic_write_jsonl,
    read_jsonl,
    sha256_file,
)

SPLITS = ("train", "validation", "test")
LEAKAGE_FIELDS = (
    "sample_id",
    "audio_sha256",
    "leakage_group_id",
    "normalized_transcript",
)


@dataclass(frozen=True)
class VariantConfig:
    repository_root: Path
    config_path: Path
    values: dict[str, Any]

    def path(self, key: str) -> Path:
        value = Path(str(self.values[key]))
        return value if value.is_absolute() else self.repository_root / value

    @property
    def seed(self) -> int:
        return int(self.values["seed"])

    @property
    def digest(self) -> str:
        canonical = json.dumps(self.values, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_variant_config(path: str | Path) -> VariantConfig:
    config_path = Path(path).resolve()
    values = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(values, dict) or values.get("schema_version") != 1:
        raise ValueError("variant config must be a schema_version: 1 mapping")
    repository_root = _repository_root(config_path)
    config = VariantConfig(repository_root, config_path, values)
    _validate_config(config)
    return config


def _repository_root(path: Path) -> Path:
    for parent in (path.parent, *path.parents):
        if (parent / ".git").exists():
            return parent
    raise ValueError(f"Could not find repository root above {path}")


def _validate_config(config: VariantConfig) -> None:
    batch_size = int(config.values["batch_size"])
    replay = config.values["replay"]
    if int(replay["term_examples_per_batch"]) + int(
        replay["general_examples_per_batch"]
    ) != batch_size:
        raise ValueError("replay batch composition must equal batch_size")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    share = float(config.values["term_balance"]["maximum_speaker_share"])
    if not 0.0 < share <= 1.0:
        raise ValueError("maximum_speaker_share must be in (0, 1]")
    variants = config.values.get("variants", [])
    strategies = [item.get("strategy") for item in variants]
    if strategies != ["uniform", "term_balance", "replay"]:
        raise ValueError(
            "variants must define uniform, term_balance, and replay in that order"
        )


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required source manifest is missing: {path}")
    return list(read_jsonl(path))


def _canonical_term(annotation: dict[str, Any]) -> str:
    return str(annotation.get("canonical") or annotation.get("text") or "").strip().casefold()


def _term_set(row: dict[str, Any]) -> set[str]:
    return {
        term
        for annotation in row.get("medical_terms", [])
        if (term := _canonical_term(annotation))
    }


def _source_audit(
    config: VariantConfig, split_rows: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    errors: list[str] = []
    required = {
        "sample_id",
        "audio",
        "text",
        "speaker_id",
        "audio_sha256",
        "leakage_group_id",
        "normalized_transcript",
    }
    for split, rows in split_rows.items():
        ids = [str(row.get("sample_id")) for row in rows]
        if len(ids) != len(set(ids)):
            errors.append(f"{split}: duplicate sample_id values")
        for index, row in enumerate(rows):
            missing = sorted(key for key in required if not row.get(key))
            if missing:
                errors.append(
                    f"{split}:{index}: missing required fields {', '.join(missing)}"
                )

    leakage: dict[str, dict[str, int]] = {}
    for field in LEAKAGE_FIELDS:
        values = {
            split: {str(row[field]) for row in rows if row.get(field)}
            for split, rows in split_rows.items()
        }
        leakage[field] = {
            "train_validation": len(values["train"] & values["validation"]),
            "train_test": len(values["train"] & values["test"]),
            "validation_test": len(values["validation"] & values["test"]),
        }
        if any(leakage[field].values()):
            errors.append(f"cross-split leakage detected for {field}")

    unique_audio: dict[str, dict[str, Any]] = {}
    for rows in split_rows.values():
        for row in rows:
            unique_audio.setdefault(str(row.get("audio")), row)

    def verify_audio(item: tuple[str, dict[str, Any]]) -> tuple[str, str | None]:
        relative, row = item
        path = config.repository_root / relative
        if not path.is_file():
            return relative, "missing"
        actual = sha256_file(path)
        if actual != row.get("audio_sha256"):
            return relative, "checksum_mismatch"
        return relative, None

    audio_failures: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        for relative, error in executor.map(verify_audio, unique_audio.items()):
            if error:
                audio_failures.append({"audio": relative, "error": error})
    if audio_failures:
        errors.append(f"{len(audio_failures)} audio files failed verification")

    return {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "split_counts": {split: len(rows) for split, rows in split_rows.items()},
        "leakage_intersections": leakage,
        "unique_audio_files": len(unique_audio),
        "audio_verification_failures": audio_failures,
    }


def _support(train: list[dict[str, Any]]) -> tuple[Counter[str], dict[str, set[str]]]:
    occurrences: Counter[str] = Counter()
    speakers: dict[str, set[str]] = {}
    for row in train:
        for term in _term_set(row):
            occurrences[term] += 1
            speakers.setdefault(term, set()).add(str(row["speaker_id"]))
    return occurrences, speakers


def _weighted_candidates(
    train: list[dict[str, Any]], config: VariantConfig
) -> tuple[list[dict[str, Any]], list[float], dict[str, Any]]:
    section = config.values["term_balance"]
    minimum_rows = int(section["minimum_term_utterances"])
    minimum_speakers = int(section["minimum_term_speakers"])
    maximum_weight = float(section["maximum_row_weight"])
    occurrences, speakers = _support(train)
    eligible = {
        term
        for term, count in occurrences.items()
        if count >= minimum_rows and len(speakers.get(term, set())) >= minimum_speakers
    }
    if not eligible:
        raise ValueError("No terms satisfy the configured balancing support gate")
    maximum_frequency = max(occurrences[term] for term in eligible)
    candidates: list[dict[str, Any]] = []
    weights: list[float] = []
    for row in train:
        row_terms = _term_set(row) & eligible
        if not row_terms:
            continue
        weight = max(
            min(maximum_weight, math.sqrt(maximum_frequency / occurrences[term]))
            for term in row_terms
        )
        candidates.append(row)
        weights.append(weight)
    support = {
        "minimum_term_utterances": minimum_rows,
        "minimum_term_speakers": minimum_speakers,
        "eligible_terms": {
            term: {
                "utterances": occurrences[term],
                "speakers": len(speakers[term]),
            }
            for term in sorted(eligible)
        },
        "insufficient_support_terms": {
            term: {
                "utterances": occurrences[term],
                "speakers": len(speakers.get(term, set())),
            }
            for term in sorted(set(occurrences) - eligible)
        },
        "candidate_rows": len(candidates),
    }
    return candidates, weights, support


def _select_weighted(
    candidates: list[dict[str, Any]],
    weights: list[float],
    count: int,
    *,
    seed: int,
    maximum_sample_exposures: int,
    maximum_speaker_share: float,
    ensure_candidate_coverage: bool = False,
) -> list[tuple[dict[str, Any], float]]:
    rng = random.Random(seed)
    selected: list[tuple[dict[str, Any], float]] = []
    sample_counts: Counter[str] = Counter()
    speaker_counts: Counter[str] = Counter()
    speaker_limit = max(1, math.ceil(count * maximum_speaker_share))
    total_weight = sum(weights)
    if ensure_candidate_coverage:
        if count < len(candidates):
            raise ValueError(
                "Exposure budget is too small to include every balancing candidate"
            )
        initial_indices = list(range(len(candidates)))
        rng.shuffle(initial_indices)
        for index in initial_indices:
            row = candidates[index]
            speaker = str(row["speaker_id"])
            if speaker_counts[speaker] >= speaker_limit:
                raise ValueError(
                    "Candidate coverage cannot satisfy maximum_speaker_share"
                )
            sample_counts[str(row["sample_id"])] += 1
            speaker_counts[speaker] += 1
            selected.append((row, weights[index]))

    for _ in range(count - len(selected)):
        accepted_index = None
        for _attempt in range(max(100, len(candidates) * 2)):
            threshold = rng.random() * total_weight
            running = 0.0
            index = len(candidates) - 1
            for candidate_index, weight in enumerate(weights):
                running += weight
                if running >= threshold:
                    index = candidate_index
                    break
            candidate = candidates[index]
            if (
                sample_counts[str(candidate["sample_id"])]
                < maximum_sample_exposures
                and speaker_counts[str(candidate["speaker_id"])] < speaker_limit
            ):
                accepted_index = index
                break
        if accepted_index is None:
            eligible_indices = [
                index
                for index, candidate in enumerate(candidates)
                if sample_counts[str(candidate["sample_id"])]
                < maximum_sample_exposures
                and speaker_counts[str(candidate["speaker_id"])] < speaker_limit
            ]
            if not eligible_indices:
                raise ValueError(
                    "Term-balanced schedule cannot satisfy sample/speaker exposure caps"
                )
            accepted_index = min(
                eligible_indices,
                key=lambda index: (
                    sample_counts[str(candidates[index]["sample_id"])],
                    speaker_counts[str(candidates[index]["speaker_id"])],
                    str(candidates[index]["sample_id"]),
                ),
            )
        row = candidates[accepted_index]
        sample_counts[str(row["sample_id"])] += 1
        speaker_counts[str(row["speaker_id"])] += 1
        selected.append((row, weights[accepted_index]))
    return selected


def _uniform_schedule(
    rows: list[dict[str, Any]], count: int, *, seed: int
) -> list[tuple[dict[str, Any], float]]:
    rng = random.Random(seed)
    ordered = list(rows)
    rng.shuffle(ordered)
    selected: list[tuple[dict[str, Any], float]] = []
    while len(selected) < count:
        cycle = list(ordered)
        if selected:
            rng.shuffle(cycle)
        selected.extend((row, 1.0) for row in cycle[: count - len(selected)])
    return selected


def _materialize(
    selected: Iterable[tuple[dict[str, Any], float]],
    *,
    variant_id: str,
    source_pool: str | list[str],
    batch_size: int,
) -> list[dict[str, Any]]:
    repeat_counts: Counter[str] = Counter()
    materialized = []
    selected_list = list(selected)
    pools = (
        source_pool
        if isinstance(source_pool, list)
        else [source_pool] * len(selected_list)
    )
    for index, ((source, weight), pool) in enumerate(zip(selected_list, pools, strict=True)):
        sample_id = str(source["sample_id"])
        repeat_index = repeat_counts[sample_id]
        repeat_counts[sample_id] += 1
        row = copy.deepcopy(source)
        row["source_sample_id"] = sample_id
        row["schedule_id"] = f"{variant_id}-{index:07d}"
        row["schedule_index"] = index
        row["batch_index"] = index // batch_size
        row["repeat_index"] = repeat_index
        row["source_pool"] = pool
        row["sampling_weight"] = round(float(weight), 8)
        row["training_variant"] = variant_id
        materialized.append(row)
    return materialized


def _build_variant(
    variant: dict[str, Any],
    train: list[dict[str, Any]],
    exposure_rows: int,
    config: VariantConfig,
    weighted: tuple[list[dict[str, Any]], list[float], dict[str, Any]],
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    variant_id = str(variant["id"])
    strategy = str(variant["strategy"])
    batch_size = int(config.values["batch_size"])
    balance = config.values["term_balance"]
    candidates, weights, support = weighted
    if strategy == "uniform":
        selected = _uniform_schedule(train, exposure_rows, seed=config.seed + 101)
        schedule = _materialize(
            selected,
            variant_id=variant_id,
            source_pool="canonical",
            batch_size=batch_size,
        )
    elif strategy == "term_balance":
        selected = _select_weighted(
            candidates,
            weights,
            exposure_rows,
            seed=config.seed + 202,
            maximum_sample_exposures=int(balance["maximum_sample_exposures"]),
            maximum_speaker_share=float(balance["maximum_speaker_share"]),
            ensure_candidate_coverage=True,
        )
        schedule = _materialize(
            selected,
            variant_id=variant_id,
            source_pool="term-balanced",
            batch_size=batch_size,
        )
    elif strategy == "replay":
        replay = config.values["replay"]
        term_per_batch = int(replay["term_examples_per_batch"])
        general_per_batch = int(replay["general_examples_per_batch"])
        batches = exposure_rows // batch_size
        target = _select_weighted(
            candidates,
            weights,
            batches * term_per_batch,
            seed=config.seed + 303,
            maximum_sample_exposures=int(balance["maximum_sample_exposures"]),
            maximum_speaker_share=float(balance["maximum_speaker_share"]),
        )
        general_pool = [row for row in train if not _term_set(row)]
        if not general_pool:
            raise ValueError("Replay strategy requires general rows without medical terms")
        general = _uniform_schedule(
            general_pool, batches * general_per_batch, seed=config.seed + 404
        )
        combined: list[tuple[dict[str, Any], float]] = []
        pools: list[str] = []
        rng = random.Random(config.seed + 505)
        for batch in range(batches):
            items = [
                (item, "term-balanced")
                for item in target[
                    batch * term_per_batch : (batch + 1) * term_per_batch
                ]
            ] + [
                (item, "replay")
                for item in general[
                    batch * general_per_batch : (batch + 1) * general_per_batch
                ]
            ]
            rng.shuffle(items)
            combined.extend(item for item, _ in items)
            pools.extend(pool for _, pool in items)
        schedule = _materialize(
            combined,
            variant_id=variant_id,
            source_pool=pools,
            batch_size=batch_size,
        )
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")
    return variant_id, schedule, support


def _distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    term_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    speaker_counts: Counter[str] = Counter()
    pool_counts: Counter[str] = Counter()
    unique_ids: set[str] = set()
    for row in rows:
        unique_ids.add(str(row["source_sample_id"]))
        speaker_counts[str(row["speaker_id"])] += 1
        pool_counts[str(row["source_pool"])] += 1
        for annotation in row.get("medical_terms", []):
            term = _canonical_term(annotation)
            if term:
                term_counts[term] += 1
            category = str(annotation.get("category") or "").strip()
            if category:
                category_counts[category] += 1
    maximum_speaker = speaker_counts.most_common(1)[0] if speaker_counts else (None, 0)
    return {
        "scheduled_rows": len(rows),
        "unique_source_rows": len(unique_ids),
        "repeated_exposures": len(rows) - len(unique_ids),
        "source_pool_counts": dict(sorted(pool_counts.items())),
        "term_occurrences": dict(sorted(term_counts.items())),
        "term_category_occurrences": dict(sorted(category_counts.items())),
        "rows_with_terms": sum(bool(_term_set(row)) for row in rows),
        "unique_speakers": len(speaker_counts),
        "maximum_speaker": maximum_speaker[0],
        "maximum_speaker_exposures": maximum_speaker[1],
        "maximum_speaker_share": round(
            maximum_speaker[1] / len(rows) if rows else 0.0, 8
        ),
        "duration_hours": round(
            sum(float(row.get("duration_seconds") or 0.0) for row in rows) / 3600.0,
            6,
        ),
    }


def _variant_audit(
    variant_id: str,
    rows: list[dict[str, Any]],
    split_rows: dict[str, list[dict[str, Any]]],
    config: VariantConfig,
    support: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    source_train_ids = {str(row["sample_id"]) for row in split_rows["train"]}
    schedule_source_ids = {str(row["source_sample_id"]) for row in rows}
    if not schedule_source_ids <= source_train_ids:
        errors.append("schedule contains samples outside the canonical train split")
    if len(rows) % int(config.values["batch_size"]):
        errors.append("schedule length is not divisible by batch_size")
    validation_ids = {str(row["sample_id"]) for row in split_rows["validation"]}
    test_ids = {str(row["sample_id"]) for row in split_rows["test"]}
    if schedule_source_ids & (validation_ids | test_ids):
        errors.append("schedule contains validation or test sample IDs")
    if variant_id == "v3-replay":
        term_expected = int(config.values["replay"]["term_examples_per_batch"])
        general_expected = int(config.values["replay"]["general_examples_per_batch"])
        batch_size = int(config.values["batch_size"])
        for start in range(0, len(rows), batch_size):
            pools = Counter(row["source_pool"] for row in rows[start : start + batch_size])
            if pools != Counter(
                {"term-balanced": term_expected, "replay": general_expected}
            ):
                errors.append(f"batch {start // batch_size} violates replay composition")
                break
    distribution = _distribution(rows)
    if variant_id in {"v2-term-balance", "v3-replay"} and (
        distribution["maximum_speaker_share"]
        > float(config.values["term_balance"]["maximum_speaker_share"])
        + (1 / max(len(rows), 1))
    ):
        errors.append("schedule exceeds maximum_speaker_share")
    return {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "variant": variant_id,
        "strategy": next(
            item["strategy"]
            for item in config.values["variants"]
            if item["id"] == variant_id
        ),
        "distribution": distribution,
        "term_support_gate": support,
    }


def build_all_variants(config_path: str | Path) -> dict[str, Any]:
    config = load_variant_config(config_path)
    source_root = config.path("source_manifest_root")
    output_root = config.path("output_root")
    split_paths = {split: source_root / f"{split}.jsonl" for split in SPLITS}
    split_rows = {split: _rows(path) for split, path in split_paths.items()}
    source_audit = _source_audit(config, split_rows)
    if source_audit["status"] != "passed":
        raise ValueError(f"Source preflight failed: {source_audit['errors']}")

    batch_size = int(config.values["batch_size"])
    exposure_rows = math.ceil(len(split_rows["train"]) / batch_size) * batch_size
    weighted = _weighted_candidates(split_rows["train"], config)
    built: dict[str, tuple[list[dict[str, Any]], dict[str, Any]]] = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(
                _build_variant,
                variant,
                split_rows["train"],
                exposure_rows,
                config,
                weighted,
            )
            for variant in config.values["variants"]
        ]
        for future in futures:
            variant_id, schedule, support = future.result()
            built[variant_id] = (schedule, support)

    output_root.mkdir(parents=True, exist_ok=True)
    variants_report: dict[str, Any] = {}
    validation_bytes = split_paths["validation"].read_bytes()
    test_bytes = split_paths["test"].read_bytes()
    for variant in config.values["variants"]:
        variant_id = str(variant["id"])
        schedule, support = built[variant_id]
        directory = output_root / variant_id
        atomic_write_jsonl(directory / "train.jsonl", schedule)
        (directory / "validation.jsonl").parent.mkdir(parents=True, exist_ok=True)
        (directory / "validation.jsonl").write_bytes(validation_bytes)
        (directory / "test.jsonl").write_bytes(test_bytes)
        audit = _variant_audit(variant_id, schedule, split_rows, config, support)
        audit["manifest_sha256"] = {
            split: sha256_file(directory / f"{split}.jsonl") for split in SPLITS
        }
        atomic_write_json(directory / "audit.json", audit)
        variants_report[variant_id] = audit

    validation_hashes = {
        report["manifest_sha256"]["validation"]
        for report in variants_report.values()
    }
    test_hashes = {
        report["manifest_sha256"]["test"] for report in variants_report.values()
    }
    errors = [
        f"{variant_id}: {error}"
        for variant_id, report in variants_report.items()
        for error in report["errors"]
    ]
    if len(validation_hashes) != 1:
        errors.append("validation manifests differ across variants")
    if len(test_hashes) != 1:
        errors.append("test manifests differ across variants")
    lock = {
        "schema_version": 1,
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "seed": config.seed,
        "config_sha256": config.digest,
        "builder_sha256": sha256_file(Path(__file__)),
        "batch_size": batch_size,
        "source_manifest_sha256": {
            split: sha256_file(path) for split, path in split_paths.items()
        },
        "source_audit": source_audit,
        "exposure_rows_per_variant": exposure_rows,
        "variants": {
            variant_id: {
                "strategy": report["strategy"],
                "manifest_sha256": report["manifest_sha256"],
                "distribution": report["distribution"],
                "audit_status": report["status"],
            }
            for variant_id, report in variants_report.items()
        },
        "shared_evaluation_manifests": len(validation_hashes) == 1
        and len(test_hashes) == 1,
    }
    atomic_write_json(output_root / "dataset-lock.json", lock)
    atomic_write_json(
        output_root / "audit-report.json",
        {"source": source_audit, "variants": variants_report},
    )
    if errors:
        raise ValueError(f"Variant preflight failed: {errors}")
    return lock
