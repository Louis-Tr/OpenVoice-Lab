from __future__ import annotations

import re
from typing import Any

import yaml

from .config import PipelineConfig
from .io_utils import sha256_file


CATEGORIES = {
    "symptom",
    "condition",
    "anatomy",
    "medication",
    "procedure",
    "abbreviation",
    "dosage",
    "measurement",
}


def _term_pattern(term: str) -> re.Pattern[str]:
    return re.compile(rf"(?<!\w){re.escape(term)}(?!\w)", re.IGNORECASE)


def annotate_medical_terms(
    records: list[dict[str, Any]], config: PipelineConfig
) -> list[dict[str, Any]]:
    path = config.path("medical_terms")
    with path.open("r", encoding="utf-8") as handle:
        terms_config = yaml.safe_load(handle)
    lexicon = terms_config.get("lexicon", {})
    unknown = set(lexicon) - CATEGORIES
    if unknown:
        raise ValueError(f"Unknown medical term categories: {sorted(unknown)}")
    compiled_terms = [
        (category, term, _term_pattern(term))
        for category in sorted(lexicon)
        for term in sorted(set(lexicon[category]), key=str.casefold)
    ]
    compiled_rules = []
    for rule in terms_config.get("regex_rules", []):
        if rule["category"] not in CATEGORIES:
            raise ValueError(f"Unknown category in regex rule {rule['name']}")
        compiled_rules.append(
            (rule["name"], rule["category"], re.compile(rule["pattern"], re.IGNORECASE))
        )
    config_sha256 = sha256_file(path)

    for record in records:
        text = record.get("normalized_transcript", "")
        annotations: list[dict[str, Any]] = []
        seen: set[tuple[int, int, str, str]] = set()
        for category, term, pattern in compiled_terms:
            for match in pattern.finditer(text):
                key = (match.start(), match.end(), category, term.casefold())
                if key in seen:
                    continue
                seen.add(key)
                annotations.append(
                    {
                        "start": match.start(),
                        "end": match.end(),
                        "text": match.group(0),
                        "canonical": term,
                        "category": category,
                        "rule": "lexicon",
                    }
                )
        for name, category, pattern in compiled_rules:
            for match in pattern.finditer(text):
                key = (match.start(), match.end(), category, match.group(0).casefold())
                if key in seen:
                    continue
                seen.add(key)
                annotations.append(
                    {
                        "start": match.start(),
                        "end": match.end(),
                        "text": match.group(0),
                        "canonical": match.group(0),
                        "category": category,
                        "rule": name,
                    }
                )
        annotations.sort(
            key=lambda item: (item["start"], item["end"], item["category"])
        )
        record["medical_terms"] = annotations
        record["medical_term_categories"] = sorted(
            {item["category"] for item in annotations}
        )
        record["medical_term_config_sha256"] = config_sha256
    return records
