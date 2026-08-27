# Dataset preparation

This document is generated from the actual Stage 11 Part 1 medical-speech pipeline results. Rerun it from the repository root with:

```powershell
python -m training.data_pipeline.run --config training/config/dataset.yaml
```

## Immutable intake

- Processing input: `D:\OpenVoice-Lab\data-processing\raw_data\Medical Speech, Transcription, and Intent`
- Canonical WAV inventory: 6661 files, 5914687708 bytes
- Canonical tree SHA-256: `0e339ad12e393b224266b8fb6fe2349d676e3ffab010b02be720f8a31fb7a231`
- Canonical CSV SHA-256: `2312ebac9c09e62c32c9dca95a9b4f1e9695ff667605f99b39c4c6598d2c9939`
- Duplicate raw root detected: True; WAV files: 6661
- Duplicate tree SHA-256: `0e339ad12e393b224266b8fb6fe2349d676e3ffab010b02be720f8a31fb7a231`
- Archive SHA-256: `0ab646a205ca2516f87a6595982600de0043a3472e8aacbb24f5d080d6ec323d`
- Canonical metadata unchanged during execution: True
- Duplicate metadata unchanged during execution: True

The pipeline reads only the configured canonical upper-case root. It does not move, rewrite, delete, or deduplicate either raw copy or the archive.

## Measured outcome

- Input records: 6661
- Readable standardized 16 kHz mono PCM WAVs: 6661
- Approved records: 4811
- Rejected or pending review: 1850
- Pending manual review: 1843
- Split counts: `{"test": 480, "train": 3850, "validation": 481}`
- Leakage assertions: `{"no_leakage_group_id_crosses_splits": true, "no_near_transcript_group_crosses_splits": true, "no_normalized_transcript_crosses_splits": true, "no_standardized_audio_sha256_crosses_splits": true}`

## Exclusion provenance

- `DURATION_ABOVE_MAXIMUM`: 8
- `DURATION_BELOW_MINIMUM`: 4
- `MANUAL_REVIEW_PENDING`: 1843
- `QUALITY_CLIPPING_EXCESSIVE`: 3
- `QUALITY_RMS_TOO_LOW`: 18
- `QUALITY_SILENCE_EXCESSIVE`: 18

Every non-approved row remains in `data-processing/manifests/medical_tts/rejections.jsonl` and `all_records.jsonl` with reason codes and source provenance.

## ASR and review status

- ASR mode: `disabled`
- Pinned ASR revision: `e8727524f962ee844a7319d92be39ac1bd25655a`
- ASR counts: `{"aligned": 0, "cache_entries": 0, "eligible": 6628, "failures": 0, "high_mismatch": 0, "not_run": 6628, "skipped_prior_exclusion": 33}`
- Manual actions fabricated: `false`

No ASR transcript or WER is populated when ASR did not execute. High WER is a review flag only. Pending high-risk records are excluded from approved manifests. Reviewer actions are append-only in the ignored review directory.

## SpeechT5 preparation status

- Processor revision: `30fcde30f19b87502b8435427b5f5068e401d5f6`
- Speaker encoder revision: `56895a2df401be4150a159f3a1c653f00051d477`
- Processor audit: `blocked`
- Speaker embedding statuses: `{"not_run_cache_miss": 4811}`
- One-batch validation: `{"detail": "ModuleNotFoundError: No module named 'transformers'", "reason": "processor_unavailable", "status": "blocked"}`
- Training-ready: `False`

The split JSONL/CSV files and `speecht5/` JSONL representation are generated even when external processor or 512-element embeddings are unavailable. A blocked status is not represented as a successful model-preprocessing result.

## Generated artifacts

- `data-processing/intermediate/medical_tts/`: stage manifests and intake hashes
- `data-processing/clean_audio/medical_16khz/`: standardized WAVs
- `data-processing/manifests/medical_tts/`: approved splits, complete records, rejections, and SpeechT5 representation
- `data-processing/review/medical_tts/`: append-only review interface and queue
- `data-processing/reports/medical_tts/`: JSON, Markdown, and HTML inspection reports

All generated data paths are Git-ignored. Source, configuration, tests, this measured document, and the thin inspection notebook remain trackable.
