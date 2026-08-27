from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from .config import PipelineConfig
from .io_utils import stable_hash


SPLIT_NAMES = ("train", "validation", "test")


def assign_splits(
    records: list[dict[str, Any]], config: PipelineConfig
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    approved = [record for record in records if not record.get("exclusion_reasons")]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in approved:
        groups[record["leakage_group_id"]].append(record)
    ratios = {name: float(config.section("splits")[name]) for name in SPLIT_NAMES}
    target = {name: ratios[name] * len(approved) for name in SPLIT_NAMES}
    counts = Counter({name: 0 for name in SPLIT_NAMES})
    covered_prompts: dict[str, set[str]] = {name: set() for name in SPLIT_NAMES}
    covered_speakers: dict[str, set[str]] = {name: set() for name in SPLIT_NAMES}
    covered_terms: dict[str, set[str]] = {name: set() for name in SPLIT_NAMES}

    ordered_groups = sorted(
        groups.items(),
        key=lambda item: (
            -len(item[1]),
            stable_hash(f"{config.seed}:{item[0]}"),
            item[0],
        ),
    )
    for group_id, group in ordered_groups:
        group_prompts = {record.get("prompt", "") for record in group}
        group_speakers = {record.get("speaker_id", "") for record in group}
        group_terms = {
            category
            for record in group
            for category in record.get("medical_term_categories", [])
        }
        scored = []
        for name in SPLIT_NAMES:
            load = (counts[name] + len(group)) / max(target[name], 1.0)
            overflow = max(0.0, counts[name] + len(group) - target[name]) / max(
                target[name], 1.0
            )
            prompt_gain = len(group_prompts - covered_prompts[name])
            speaker_gain = len(group_speakers - covered_speakers[name])
            term_gain = len(group_terms - covered_terms[name])
            coverage_bonus = min(
                0.08, prompt_gain * 0.008 + speaker_gain * 0.001 + term_gain * 0.004
            )
            score = load + overflow * 0.5 - coverage_bonus
            scored.append(
                (score, stable_hash(f"{config.seed}:{group_id}:{name}"), name)
            )
        _, _, selected = min(scored)
        for record in group:
            record["split"] = selected
        counts[selected] += len(group)
        covered_prompts[selected].update(group_prompts)
        covered_speakers[selected].update(group_speakers)
        covered_terms[selected].update(group_terms)

    leakage_assertions = _leakage_assertions(approved)
    if not all(leakage_assertions.values()):
        raise AssertionError(f"Split leakage detected: {leakage_assertions}")
    approved_ids = {record["sample_id"] for record in approved}
    for record in records:
        if record["sample_id"] not in approved_ids:
            record["split"] = None
    return records, {
        "seed": config.seed,
        "ratios": ratios,
        "approved_count": len(approved),
        "group_count": len(groups),
        "actual_counts": {name: counts[name] for name in SPLIT_NAMES},
        "target_counts": target,
        "coverage": {
            name: {
                "prompts": len(covered_prompts[name]),
                "speakers": len(covered_speakers[name]),
                "medical_term_categories": len(covered_terms[name]),
            }
            for name in SPLIT_NAMES
        },
        "leakage_assertions": leakage_assertions,
    }


def _leakage_assertions(records: list[dict[str, Any]]) -> dict[str, bool]:
    fields = (
        "leakage_group_id",
        "standardized_audio_sha256",
        "normalized_transcript",
        "near_transcript_group",
    )
    assertions: dict[str, bool] = {}
    for field in fields:
        seen: dict[str, str] = {}
        safe = True
        for record in records:
            value = record.get(field)
            if not value:
                continue
            previous = seen.setdefault(str(value), record["split"])
            if previous != record["split"]:
                safe = False
                break
        assertions[f"no_{field}_crosses_splits"] = safe
    return assertions
