"""Export verified Stage 11 API projections for read-only cloud deployments."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

from app.experiments.service import create_experiment_service
from app.text_processing.service import TextProcessingService

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = REPOSITORY_ROOT / "backend" / "app" / "experiments" / "snapshots"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def source_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        text=True,
    ).strip()


def main() -> None:
    service = create_experiment_service(
        artifact_root=REPOSITORY_ROOT / "artifacts" / "stage11" / "full-training",
        approach_run_root=REPOSITORY_ROOT / "artifacts" / "stage11" / "agent-runs",
        manifest_root=REPOSITORY_ROOT / "data-processing" / "manifests" / "stage11",
        stage12_root=REPOSITORY_ROOT / "artifacts" / "stage12",
        model_cache_root=REPOSITORY_ROOT / "artifacts" / "stage12" / "model-cache",
        speaker_profile_root=(
            REPOSITORY_ROOT / "artifacts" / "stage12" / "serving-profile"
        ),
        text_processing=TextProcessingService(),
        audio_url_prefix="/experiment-audio",
        tts_revision="30fcde30f19b87502b8435427b5f5068e401d5f6",
        vocoder_revision="bb6f429406e86a9992357a972c0698b22043307d",
        maximum_queued_jobs=2,
        maximum_cached_models=2,
        cpu_threads=None,
    )
    report_path = OUTPUT_ROOT / "report.json"
    fixtures_path = OUTPUT_ROOT / "fixtures.json"
    models_path = OUTPUT_ROOT / "models.json"
    write_json(report_path, service.report().model_dump(mode="json", by_alias=True))
    write_json(
        fixtures_path,
        service.fixtures(
            query=None,
            term=None,
            category=None,
            offset=0,
            limit=100_000,
        ).model_dump(mode="json", by_alias=True),
    )
    write_json(
        models_path,
        [item.model_dump(mode="json", by_alias=True) for item in service.models()],
    )
    write_json(
        OUTPUT_ROOT / "manifest.json",
        {
            "schema_version": 1,
            "generated_at": datetime.now(UTC).isoformat(),
            "source_commit": source_commit(),
            "files": {
                path.name: file_sha256(path)
                for path in (report_path, fixtures_path, models_path)
            },
        },
    )
    print(f"Exported verified experiment snapshot to {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
