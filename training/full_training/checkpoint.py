from __future__ import annotations

import hashlib
import json
from pathlib import Path

CHECKPOINT_MANIFEST = "checkpoint_manifest.json"
CHECKPOINT_MARKER = "checkpoint_complete.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint(path: Path) -> tuple[bool, str | None]:
    marker_path = path / CHECKPOINT_MARKER
    manifest_path = path / CHECKPOINT_MANIFEST
    if not marker_path.is_file() or not manifest_path.is_file():
        return False, "completion marker or manifest missing"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("manifest_sha256") != sha256(manifest_path):
        return False, "checkpoint manifest hash mismatch"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for entry in manifest.get("files", []):
        file = path / entry["path"]
        if not file.is_file():
            return False, f"missing checkpoint file: {entry['path']}"
        if file.stat().st_size != int(entry["bytes"]):
            return False, f"checkpoint size mismatch: {entry['path']}"
        if sha256(file) != entry["sha256"]:
            return False, f"checkpoint hash mismatch: {entry['path']}"
    return True, None
