from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import random
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from huggingface_hub import snapshot_download
from transformers import set_seed

from training.full_training.run import (
    _atomic_json,
    _evaluate,
    _load_and_validate_inputs,
    _sha256,
    _speaker_embedding,
    _speaker_encoder,
)


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _evidence_manifest(output: Path) -> dict[str, Any]:
    roots = (
        output / "run_provenance.json",
        output / "pip-freeze.txt",
        output / "evaluation" / "summary.json",
        output / "evaluation" / "raw_results.jsonl",
        output / "evaluation" / "raw_results.csv",
    )
    files = [path for path in roots if path.is_file()]
    verification = json.loads(
        (output / "evaluation" / "summary.json").read_text(encoding="utf-8")
    )["synthesis_verification"]
    sample = output / str(verification.get("audio", ""))
    if verification.get("status") == "passed" and sample.is_file():
        files.append(sample)
    return {
        "schema_version": 1,
        "created_utc": _utc(),
        "files": [
            {
                "path": path.relative_to(output).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in sorted(files)
        ],
    }


def _evidence_archive(output: Path, manifest: dict[str, Any]) -> Path:
    archive = output.parent / "pretrained-evidence.tar.gz"
    temporary = archive.with_suffix(archive.suffix + ".tmp")
    paths = [
        output / "EVALUATION_COMPLETE.json",
        output / "evaluation_artifact_manifest.json",
        *(output / entry["path"] for entry in manifest["files"]),
    ]
    with tarfile.open(temporary, "w:gz") as handle:
        for path in sorted(set(paths)):
            handle.add(path, arcname=path.relative_to(output.parent).as_posix())
    os.replace(temporary, archive)
    return archive


def run(config_path: Path, output: Path) -> dict[str, Any]:
    root = Path.cwd().resolve()
    config_path = config_path.resolve()
    output = output.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _, rows, lock = _load_and_validate_inputs(root, config, "v1-baseline")
    if not torch.cuda.is_available():
        raise RuntimeError("pretrained baseline evaluation requires a CUDA GPU")

    output.mkdir(parents=True, exist_ok=True)
    lock_path = root / config["dataset"]["lock"]
    test_path = root / config["variants"][0]["manifest_root"] / "test.jsonl"
    models = config["models"]
    started_utc = _utc()
    started = time.perf_counter()
    set_seed(int(config["seed"]))
    random.seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))

    cache = output / "model-cache"
    source = Path(
        snapshot_download(
            repo_id=models["tts_id"],
            revision=models["tts_revision"],
            cache_dir=cache,
        )
    )
    encoder = _speaker_encoder(config, cache)
    speaker_source = root / rows["validation"][0]["audio"]
    embedding = _speaker_embedding(encoder, speaker_source)
    del encoder
    gc.collect()
    torch.cuda.empty_cache()

    provenance = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "evaluation_id": os.environ.get(
            "OPENVOICE_EVALUATION_ID", "local-pretrained-evaluation"
        ),
        "role": "pretrained-control",
        "run_id": os.environ.get("OPENVOICE_RUN_ID"),
        "pod_id": os.environ.get("RUNPOD_POD_ID"),
        "started_utc": started_utc,
        "dataset_lock_sha256": _sha256(lock_path),
        "dataset_lock_status": lock["status"],
        "test_manifest_sha256": _sha256(test_path),
        "test_case_count": len(rows["test"]),
        "speaker_source_sample_id": rows["validation"][0]["sample_id"],
        "speaker_source_audio_sha256": rows["validation"][0]["audio_sha256"],
        "models": models,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "gpu_total_memory_mb": (
                torch.cuda.get_device_properties(0).total_memory / 1024**2
            ),
        },
    }
    _atomic_json(output / "run_provenance.json", provenance)
    with (output / "pip-freeze.txt").open("w", encoding="utf-8") as handle:
        subprocess.run(
            [sys.executable, "-m", "pip", "freeze"], check=True, stdout=handle
        )

    summary = _evaluate(config, source, rows["test"], embedding, output)
    completed_utc = _utc()
    provenance.update(
        completed_utc=completed_utc,
        elapsed_seconds=time.perf_counter() - started,
        status=(
            "completed"
            if int(summary["failure_count"]) == 0
            else "completed_with_failures"
        ),
    )
    _atomic_json(output / "run_provenance.json", provenance)
    manifest_path = output / "evaluation_artifact_manifest.json"
    manifest = _evidence_manifest(output)
    _atomic_json(manifest_path, manifest)
    marker = {
        "schema_version": 1,
        "status": provenance["status"],
        "completed_utc": completed_utc,
        "evaluation_artifact_manifest_sha256": _sha256(manifest_path),
    }
    _atomic_json(output / "EVALUATION_COMPLETE.json", marker)
    archive = _evidence_archive(output, manifest)
    print(
        json.dumps(
            {
                "provenance": provenance,
                "summary": summary,
                "evidence_archive": str(archive),
                "evidence_archive_sha256": _sha256(archive),
            },
            indent=2,
        )
    )
    return marker


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate pinned pretrained SpeechT5 on the locked Stage 11 test set."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("training/config/full_training.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/stage11/full-training/pretrained"),
    )
    args = parser.parse_args()
    run(args.config, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
