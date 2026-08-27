from __future__ import annotations

import csv

from training.data_pipeline.inventory import inventory_intake
from training.data_pipeline.validation import validate_records

from .conftest import write_csv


def test_sample_ids_do_not_depend_on_csv_row_order(fixture_config) -> None:
    csv_path = fixture_config.path("raw_root") / "overview-of-recordings.csv"
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    first = inventory_intake(fixture_config)

    write_csv(csv_path, reversed(rows))
    second = inventory_intake(fixture_config)

    assert {row["file_name"]: row["sample_id"] for row in first} == {
        row["file_name"]: row["sample_id"] for row in second
    }


def test_corrupt_wav_is_rejected_without_crashing(fixture_config) -> None:
    corrupt = fixture_config.repository_root / "corrupt.wav"
    corrupt.write_bytes(b"this is not a wav" * 8)
    record = {
        "sample_id": "corrupt-sample",
        "source_wav": "corrupt.wav",
        "original_transcript": "I have chest pain",
        "speaker_id": "speaker",
        "prompt": "Chest pain",
        "source_metadata": {
            "audio_clipping:confidence": "1",
            "background_noise_audible:confidence": "1",
            "overall_quality_of_the_audio": "3",
            "quiet_speaker:confidence": "1",
        },
        "intake_issues": [],
        "exclusion_reasons": [],
        "review_flags": [],
    }

    result = validate_records([record], fixture_config)

    assert result[0]["exclusion_reasons"] == ["WAV_UNREADABLE"]
    assert result[0]["structural_error"]
