from __future__ import annotations

import csv
import hashlib
import math
import wave
from pathlib import Path
from typing import Iterable

import numpy as np
import pytest
import yaml

from training.data_pipeline.config import PipelineConfig


COLUMNS = [
    "audio_clipping",
    "audio_clipping:confidence",
    "background_noise_audible",
    "background_noise_audible:confidence",
    "overall_quality_of_the_audio",
    "quiet_speaker",
    "quiet_speaker:confidence",
    "speaker_id",
    "file_download",
    "file_name",
    "phrase",
    "prompt",
    "writer_id",
]


def write_wav(
    path: Path, *, duration: float, frequency: float, silent: bool = False
) -> None:
    sample_rate = 22050
    time = np.arange(int(duration * sample_rate), dtype=np.float64) / sample_rate
    values = (
        np.zeros_like(time) if silent else 0.25 * np.sin(2 * math.pi * frequency * time)
    )
    stereo = np.column_stack([values, values * 0.9])
    pcm = np.rint(stereo * 32767).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(2)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(pcm.tobytes())


def write_csv(path: Path, rows: Iterable[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def fixture_config(tmp_path: Path) -> PipelineConfig:
    repository_root = tmp_path
    (repository_root / ".git").mkdir()
    raw = (
        repository_root
        / "data-processing/raw_data/Medical Speech, Transcription, and Intent"
    )
    rows = []
    phrases = [
        "My chest hurts",
        "My chest hurts",
        "I have a headache",
        "I have severe headache",
        "BP is 120/80 mmHg",
        "Take 5 mg aspirin",
        "My left arm hurts",
        "My right arm hurts",
        "I feel dizzy",
        "I have a sore throat",
        "My knee is swollen",
        "SpO2 is 97%",
    ]
    for index, phrase in enumerate(phrases):
        filename = f"sample-{index:02d}.wav"
        split = ("train", "test", "validate")[index % 3]
        duration = 0.6 if index == 0 else 1.1 + index * 0.05
        write_wav(
            raw / "recordings" / split / filename,
            duration=duration,
            frequency=220 + index * 17,
            silent=index == 10,
        )
        rows.append(
            {
                "audio_clipping": "no_clipping",
                "audio_clipping:confidence": "1",
                "background_noise_audible": "no_noise",
                "background_noise_audible:confidence": "1",
                "overall_quality_of_the_audio": "4.0",
                "quiet_speaker": "audible_speaker",
                "quiet_speaker:confidence": "1",
                "speaker_id": f"speaker-{index % 5}",
                "file_download": "",
                "file_name": filename,
                "phrase": phrase,
                "prompt": f"prompt-{index % 4}",
                "writer_id": f"writer-{index % 3}",
            }
        )
    write_csv(raw / "overview-of-recordings.csv", rows)
    terms_source = Path(__file__).parents[2] / "config" / "medical_terms.yaml"
    terms_target = repository_root / "training/config/medical_terms.yaml"
    terms_target.parent.mkdir(parents=True, exist_ok=True)
    terms_target.write_text(terms_source.read_text(encoding="utf-8"), encoding="utf-8")
    values = yaml.safe_load(
        (Path(__file__).parents[2] / "config" / "dataset.yaml").read_text(
            encoding="utf-8"
        )
    )
    values["intake"]["expected_rows"] = len(rows)
    values["intake"]["hash_archive"] = False
    values["intake"]["hash_duplicate_copy"] = False
    values["audio"]["workers"] = 2
    values["inspection"]["accepted_sample_size"] = 5
    values["inspection"]["rejection_examples_per_category"] = 5
    config_path = repository_root / "training/config/dataset.yaml"
    config_path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
    return PipelineConfig(repository_root, config_path, values)
