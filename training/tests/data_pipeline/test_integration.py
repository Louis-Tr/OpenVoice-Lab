from __future__ import annotations

import json
import wave

from training.data_pipeline.io_utils import read_jsonl
from training.data_pipeline.pipeline import run_pipeline

from .conftest import sha256


def test_full_fixture_pipeline_is_immutable_idempotent_and_leak_free(
    fixture_config,
) -> None:
    raw_root = fixture_config.path("raw_root")
    raw_files = sorted(path for path in raw_root.rglob("*") if path.is_file())
    before = {path.relative_to(raw_root).as_posix(): sha256(path) for path in raw_files}

    first = run_pipeline(fixture_config)
    manifest_dir = fixture_config.path("manifests")
    first_approved = (manifest_dir / "approved.jsonl").read_bytes()
    first_all_records = (manifest_dir / "all_records.jsonl").read_bytes()
    cleaning_report_path = fixture_config.path("reports") / "cleaning_report.json"
    first_report = cleaning_report_path.read_bytes()
    report = json.loads(first_report)

    assert report["counts"]["input_records"] == 12
    assert report["counts"]["standardized_audio"] == 12
    assert report["rejection_reason_counts"]["DURATION_BELOW_MINIMUM"] == 1
    assert report["rejection_reason_counts"]["QUALITY_RMS_TOO_LOW"] == 1
    assert all(report["stage_reports"]["split"]["leakage_assertions"].values())
    assert report["stage_reports"]["asr"]["counts"]["aligned"] == 0
    assert report["stage_reports"]["asr"]["counts"]["not_run"] > 0
    assert report["stage_reports"]["speecht5"]["training_ready"] is False
    assert (fixture_config.path("review") / "review_queue.html").is_file()
    assert (fixture_config.path("reports") / "inspection_report.html").is_file()

    approved = list(read_jsonl(manifest_dir / "approved.jsonl"))
    assert approved
    for row in approved:
        with wave.open(
            str(fixture_config.repository_root / row["audio"]), "rb"
        ) as audio:
            assert (
                audio.getframerate(),
                audio.getnchannels(),
                audio.getsampwidth(),
            ) == (16000, 1, 2)

    second = run_pipeline(fixture_config)
    assert first["config_sha256"] == second["config_sha256"]
    assert (manifest_dir / "approved.jsonl").read_bytes() == first_approved
    assert (manifest_dir / "all_records.jsonl").read_bytes() == first_all_records
    assert cleaning_report_path.read_bytes() == first_report
    after = {path.relative_to(raw_root).as_posix(): sha256(path) for path in raw_files}
    assert before == after


def test_review_bypass_is_explicit_and_limited_to_smoke_runs(fixture_config) -> None:
    result = run_pipeline(
        fixture_config,
        limit=12,
        accept_unreviewed_for_smoke=True,
    )

    assert result["accept_unreviewed_for_smoke"] is True
    review_report = json.loads(
        (
            fixture_config.path("intermediate") / "08_review" / "stage_report.json"
        ).read_text(encoding="utf-8")
    )
    assert review_report["accept_unreviewed_for_smoke"] is True
    assert review_report["accepted_unreviewed_for_smoke_count"] > 0
    assert review_report["pending_count"] == 0


def test_review_bypass_cannot_run_without_a_limit(fixture_config) -> None:
    try:
        run_pipeline(fixture_config, accept_unreviewed_for_smoke=True)
    except ValueError as error:
        assert "restricted to limited smoke runs" in str(error)
    else:
        raise AssertionError("full-corpus review bypass unexpectedly succeeded")
