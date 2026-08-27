from __future__ import annotations

import re
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Any, Hashable

from .config import PipelineConfig
from .io_utils import stable_hash


class UnionFind:
    def __init__(self, values: list[Hashable]):
        self.parent = {value: value for value in values}

    def find(self, value: Hashable) -> Hashable:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: Hashable, right: Hashable) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        first, second = sorted((left_root, right_root), key=str)
        self.parent[second] = first


def _comparison_text(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))


def _group_id(prefix: str, values: list[str]) -> str:
    joined = "\n".join(sorted(values))
    return f"{prefix}-{stable_hash(joined, 16)}"


def detect_duplicates(
    records: list[dict[str, Any]], config: PipelineConfig
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    usable = [record for record in records if record.get("standardized_audio_sha256")]
    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_phrase: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_phrase_speaker: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in usable:
        by_hash[record["standardized_audio_sha256"]].append(record)
        phrase = _comparison_text(record.get("normalized_transcript", ""))
        by_phrase[phrase].append(record)
        by_phrase_speaker[(phrase, record.get("speaker_id", ""))].append(record)

    phrases = sorted(phrase for phrase in by_phrase if phrase)
    phrase_union = UnionFind(phrases)
    threshold = float(config.section("duplicates")["near_transcript_similarity"])
    buckets: dict[tuple[str, int], list[str]] = defaultdict(list)
    for phrase in phrases:
        tokens = phrase.split()
        prefix = tokens[0] if tokens else ""
        buckets[(prefix, len(tokens))].append(phrase)
    comparisons = 0
    for phrase in phrases:
        tokens = phrase.split()
        candidate_phrases: set[str] = set()
        for token_count in range(max(1, len(tokens) - 2), len(tokens) + 3):
            candidate_phrases.update(
                buckets.get((tokens[0] if tokens else "", token_count), [])
            )
        for candidate in sorted(candidate_phrases):
            if candidate <= phrase:
                continue
            comparisons += 1
            left_tokens = set(tokens)
            right_tokens = set(candidate.split())
            union = left_tokens | right_tokens
            jaccard = len(left_tokens & right_tokens) / len(union) if union else 1.0
            ratio = SequenceMatcher(None, phrase, candidate, autojunk=False).ratio()
            if ratio >= threshold or (
                jaccard >= threshold and abs(len(phrase) - len(candidate)) <= 8
            ):
                phrase_union.union(phrase, candidate)

    near_members: dict[str, list[str]] = defaultdict(list)
    for phrase in phrases:
        near_members[str(phrase_union.find(phrase))].append(phrase)
    near_ids = {
        phrase: _group_id("near", members)
        for members in near_members.values()
        for phrase in members
    }

    sample_union = UnionFind([record["sample_id"] for record in usable])
    for groups in (by_hash.values(), by_phrase.values()):
        for group in groups:
            anchor = min(record["sample_id"] for record in group)
            for record in group:
                sample_union.union(anchor, record["sample_id"])
    records_by_near: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for phrase, grouped_records in by_phrase.items():
        for record in grouped_records:
            records_by_near[near_ids.get(phrase, _group_id("near", [phrase]))].append(
                record
            )
    for group in records_by_near.values():
        anchor = min(record["sample_id"] for record in group)
        for record in group:
            sample_union.union(anchor, record["sample_id"])

    leakage_members: dict[str, list[str]] = defaultdict(list)
    for record in usable:
        leakage_members[str(sample_union.find(record["sample_id"]))].append(
            record["sample_id"]
        )
    leakage_ids = {
        sample_id: _group_id("leak", members)
        for members in leakage_members.values()
        for sample_id in members
    }

    redundant_exact = 0
    for audio_hash, group in sorted(by_hash.items()):
        members = sorted(record["sample_id"] for record in group)
        group_id = _group_id("audio", members)
        winner = members[0]
        for record in group:
            record["exact_audio_group"] = group_id
            record["exact_audio_duplicate_count"] = len(group)
            record["exact_audio_canonical_sample_id"] = winner
            if (
                len(group) > 1
                and record["sample_id"] != winner
                and bool(config.section("duplicates")["reject_redundant_exact_audio"])
            ):
                if "EXACT_AUDIO_DUPLICATE" not in record["exclusion_reasons"]:
                    record["exclusion_reasons"].append("EXACT_AUDIO_DUPLICATE")
                    redundant_exact += 1

    for phrase, group in by_phrase.items():
        exact_phrase_group = _group_id(
            "phrase", [record["sample_id"] for record in group]
        )
        for record in group:
            record["normalized_phrase_group"] = exact_phrase_group
            record["near_transcript_group"] = near_ids.get(
                phrase, _group_id("near", [phrase])
            )
            record["leakage_group_id"] = leakage_ids[record["sample_id"]]
    for (_, _), group in by_phrase_speaker.items():
        group_id = _group_id("speaker-text", [record["sample_id"] for record in group])
        for record in group:
            record["same_transcript_speaker_group"] = group_id
            record["same_transcript_speaker_count"] = len(group)
    for record in records:
        if not record.get("leakage_group_id"):
            record["leakage_group_id"] = _group_id("leak", [record["sample_id"]])
        record["exclusion_reasons"].sort()

    report = {
        "records_with_standardized_audio": len(usable),
        "unique_standardized_audio_hashes": len(by_hash),
        "exact_audio_duplicate_groups": sum(
            len(group) > 1 for group in by_hash.values()
        ),
        "redundant_exact_audio_exclusions": redundant_exact,
        "unique_normalized_phrases": len(by_phrase),
        "exact_phrase_groups_with_multiple_samples": sum(
            len(group) > 1 for group in by_phrase.values()
        ),
        "same_transcript_speaker_groups_with_multiple_samples": sum(
            len(group) > 1 for group in by_phrase_speaker.values()
        ),
        "near_transcript_groups": len(near_members),
        "near_transcript_pair_comparisons": comparisons,
        "leakage_groups": len(leakage_members),
    }
    return records, report
