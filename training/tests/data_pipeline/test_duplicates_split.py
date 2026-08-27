from training.data_pipeline.duplicates import detect_duplicates
from training.data_pipeline.splitting import assign_splits


def _record(index: int, phrase: str, audio_hash: str) -> dict:
    return {
        "sample_id": f"sample-{index}",
        "standardized_audio_sha256": audio_hash,
        "normalized_transcript": phrase,
        "speaker_id": f"speaker-{index % 3}",
        "prompt": f"prompt-{index % 2}",
        "medical_term_categories": ["symptom"],
        "exclusion_reasons": [],
        "review_flags": [],
    }


def test_exact_audio_is_rejected_and_phrases_never_cross_splits(fixture_config) -> None:
    records = [
        _record(0, "I have chest pain", "hash-a"),
        _record(1, "I have chest pain", "hash-b"),
        _record(2, "I have severe chest pain", "hash-c"),
        _record(3, "My knee hurts", "hash-d"),
        _record(4, "My head hurts", "hash-e"),
        _record(5, "Another phrase", "hash-a"),
    ]

    records, duplicate_report = detect_duplicates(records, fixture_config)
    records, split_report = assign_splits(records, fixture_config)

    assert duplicate_report["exact_audio_duplicate_groups"] == 1
    assert (
        sum("EXACT_AUDIO_DUPLICATE" in row["exclusion_reasons"] for row in records) == 1
    )
    assert all(split_report["leakage_assertions"].values())
    phrase_splits = {
        row["split"]
        for row in records
        if row["normalized_transcript"] == "I have chest pain"
    }
    assert len(phrase_splits) == 1
