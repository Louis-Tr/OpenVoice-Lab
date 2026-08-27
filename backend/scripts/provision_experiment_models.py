"""Provision pinned public Stage 12 model dependencies with the `hf` CLI."""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

MODELS = (
    (
        "pretrained-speecht5",
        "microsoft/speecht5_tts",
        "30fcde30f19b87502b8435427b5f5068e401d5f6",
    ),
    (
        "vocoder",
        "microsoft/speecht5_hifigan",
        "bb6f429406e86a9992357a972c0698b22043307d",
    ),
    (
        "asr",
        "openai/whisper-small.en",
        "e8727524f962ee844a7319d92be39ac1bd25655a",
    ),
    (
        "speaker-encoder",
        "speechbrain/spkrec-xvect-voxceleb",
        "56895a2df401be4150a159f3a1c653f00051d477",
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def provision(root: Path) -> None:
    colocated_cli = Path(sys.executable).with_name("hf.exe" if sys.platform == "win32" else "hf")
    hf_cli = str(colocated_cli) if colocated_cli.is_file() else shutil.which("hf")
    if hf_cli is None:
        raise RuntimeError("The `hf` CLI is required. Install it from https://hf.co/cli.")
    root.mkdir(parents=True, exist_ok=True)
    for directory, repository, revision in MODELS:
        target = root / directory
        subprocess.run(
            [
                hf_cli,
                "download",
                repository,
                "--revision",
                revision,
                "--local-dir",
                str(target),
                "--max-workers",
                "4",
            ],
            check=True,
        )
        files = [
            path
            for path in target.rglob("*")
            if path.is_file()
            and ".cache" not in path.parts
            and path.name != "openvoice-model-manifest.json"
        ]
        manifest = {
            "schema_version": 1,
            "repository": repository,
            "revision": revision,
            "files": [
                {
                    "path": path.relative_to(target).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
                for path in sorted(files)
            ],
        }
        weights = next(
            (
                item
                for item in manifest["files"]
                if item["path"] in {"model.safetensors", "pytorch_model.bin"}
            ),
            None,
        )
        if weights:
            manifest["weight_sha256"] = weights["sha256"]
        (target / "openvoice-model-manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "artifacts" / "stage12" / "model-cache",
    )
    args = parser.parse_args()
    provision(args.output_root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
