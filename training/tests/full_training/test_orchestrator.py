from __future__ import annotations

import hashlib
import json
from pathlib import Path

from training.full_training.checkpoint import verify_checkpoint
from training.full_training.orchestrator import SshEndpoint, ssh_endpoint


def test_runtime_ssh_endpoint_is_discovered() -> None:
    pod = {
        "runtime": {
            "ports": [
                {
                    "ip": "203.0.113.5",
                    "privatePort": 22,
                    "publicPort": 22123,
                    "type": "tcp",
                }
            ]
        }
    }

    assert ssh_endpoint(pod) == SshEndpoint("203.0.113.5", 22123)


def test_checkpoint_verifier_requires_marker_and_file_hashes(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint-125"
    checkpoint.mkdir()
    weight = checkpoint / "model.safetensors"
    weight.write_bytes(b"test-weights")
    digest = hashlib.sha256(weight.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "checkpoint_step": 125,
        "files": [
            {"path": weight.name, "bytes": weight.stat().st_size, "sha256": digest}
        ],
    }
    manifest_path = checkpoint / "checkpoint_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    marker = {
        "checkpoint_step": 125,
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    }
    (checkpoint / "checkpoint_complete.json").write_text(
        json.dumps(marker), encoding="utf-8"
    )

    assert verify_checkpoint(checkpoint) == (True, None)
    weight.write_bytes(b"corrupt")
    valid, reason = verify_checkpoint(checkpoint)
    assert valid is False
    assert "size mismatch" in str(reason)
