from __future__ import annotations

from collections import Counter

from training.dataset_variants.builder import (
    VariantConfig,
    _build_variant,
    _variant_audit,
    _weighted_candidates,
)


def _row(index: int, *, term: bool, split: str = "train") -> dict:
    annotations = (
        [
            {
                "canonical": "aspirin" if index % 2 else "hypertension",
                "text": "aspirin" if index % 2 else "hypertension",
                "category": "medication" if index % 2 else "condition",
            }
        ]
        if term
        else []
    )
    return {
        "sample_id": f"{split}-{index}",
        "audio": f"audio/{split}-{index}.wav",
        "text": f"sentence {index}",
        "normalized_transcript": f"sentence {index}",
        "speaker_id": f"speaker-{index % 4}",
        "medical_terms": annotations,
        "duration_seconds": 2.0,
        "audio_sha256": f"audio-sha-{split}-{index}",
        "leakage_group_id": f"group-{split}-{index}",
    }


def _config(tmp_path) -> VariantConfig:
    values = {
        "schema_version": 1,
        "seed": 42,
        "source_manifest_root": "source",
        "output_root": "output",
        "batch_size": 8,
        "term_balance": {
            "minimum_term_utterances": 2,
            "minimum_term_speakers": 2,
            "maximum_row_weight": 4.0,
            "maximum_sample_exposures": 8,
            "maximum_speaker_share": 1.0,
        },
        "replay": {
            "term_examples_per_batch": 4,
            "general_examples_per_batch": 4,
        },
        "variants": [
            {"id": "v1-baseline", "strategy": "uniform"},
            {"id": "v2-term-balance", "strategy": "term_balance"},
            {"id": "v3-replay", "strategy": "replay"},
        ],
    }
    return VariantConfig(tmp_path, tmp_path / "variants.yaml", values)


def test_three_schedules_have_equal_exposure_and_exact_replay_batches(tmp_path) -> None:
    config = _config(tmp_path)
    train = [_row(index, term=index < 8) for index in range(16)]
    split_rows = {
        "train": train,
        "validation": [_row(100, term=True, split="validation")],
        "test": [_row(200, term=True, split="test")],
    }
    weighted = _weighted_candidates(train, config)

    schedules = {}
    for variant in config.values["variants"]:
        variant_id, rows, support = _build_variant(
            variant, train, 16, config, weighted
        )
        audit = _variant_audit(variant_id, rows, split_rows, config, support)
        assert audit["status"] == "passed", audit["errors"]
        schedules[variant_id] = rows

    assert {len(rows) for rows in schedules.values()} == {16}
    assert all(row["medical_terms"] for row in schedules["v2-term-balance"])
    assert {
        row["source_sample_id"] for row in schedules["v2-term-balance"]
    } == {row["sample_id"] for row in train if row["medical_terms"]}
    for start in range(0, 16, 8):
        pools = Counter(
            row["source_pool"]
            for row in schedules["v3-replay"][start : start + 8]
        )
        assert pools == {"term-balanced": 4, "replay": 4}
