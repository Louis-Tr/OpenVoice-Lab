"""Build the verified remote inventory and optional upload tree for Stage 11 models."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from hashlib import sha256
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))

from app.experiments.approach_catalog import V1_APPROACHES


def digest(path: Path) -> str:
    result = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            result.update(chunk)
    return result.hexdigest()


def export(
    *,
    run_root: Path,
    manifest_output: Path,
    staging_root: Path,
    stage_files: bool,
) -> None:
    models = []
    for approach in V1_APPROACHES:
        final_root = run_root / approach.run_id / "final"
        source_root = final_root / "selected-model"
        artifact_manifest = json.loads(
            (final_root / "run_artifact_manifest.json").read_text(encoding="utf-8")
        )
        entries = []
        for item in artifact_manifest.get("files", []):
            raw_path = str(item.get("path", ""))
            if not raw_path.startswith("selected-model/"):
                continue
            relative = Path(raw_path).relative_to("selected-model")
            source = source_root / relative
            if (
                not source.is_file()
                or source.stat().st_size != int(item["bytes"])
                or digest(source) != item["sha256"]
            ):
                raise RuntimeError(f"Selected model artifact does not verify: {source}")
            entries.append(
                {
                    "path": relative.as_posix(),
                    "bytes": source.stat().st_size,
                    "sha256": item["sha256"],
                }
            )
            if stage_files:
                destination = staging_root / "models" / approach.model_id / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
        if not entries:
            raise RuntimeError(f"No selected model files found for {approach.run_id}")
        models.append({"id": approach.model_id, "files": entries})

    payload = {"schema_version": 1, "models": models}
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_output.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest_output)
    if stage_files:
        shutil.copyfile(manifest_output, staging_root / "remote_models.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=REPOSITORY_ROOT / "artifacts" / "stage11" / "agent-runs",
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=BACKEND_ROOT / "app" / "experiments" / "snapshots" / "remote_models.json",
    )
    parser.add_argument(
        "--staging-root",
        type=Path,
        default=REPOSITORY_ROOT / "artifacts" / "stage12" / "remote-models",
    )
    parser.add_argument("--stage-files", action="store_true")
    args = parser.parse_args()
    export(
        run_root=args.run_root.resolve(),
        manifest_output=args.manifest_output.resolve(),
        staging_root=args.staging_root.resolve(),
        stage_files=args.stage_files,
    )
    print(args.manifest_output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
