from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from training.runpod_agent import core


class FakeClient:
    def __init__(self):
        self.created = []
        self.terminated = []

    def create_pod(self, body):
        self.created.append(body)
        return {
            "id": "pod-test-4090",
            "name": body["name"],
            "desiredStatus": "CREATED",
            "costPerHr": 0.74,
        }

    def get_pod(self, pod_id):
        raise RuntimeError(f"RunPod GET /pods/{pod_id} failed (404): not found")

    def terminate_pod(self, pod_id):
        self.terminated.append(pod_id)


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def test_profiles_are_configuration_driven_and_share_v1_dataset():
    root = repository_root()
    for approach in core.PROFILES:
        profile = core.load_profile(root, approach)
        assert profile["approach"] == approach
        assert profile["variant"]["manifest_root"].endswith("v1-baseline")
        assert profile["models"]["tts_revision"]
        assert profile["runtime"]["gpu_type"] == "NVIDIA GeForce RTX 4090"


def test_create_pod_writes_pod_and_run_documents(tmp_path, monkeypatch):
    root = repository_root()
    key = tmp_path / "runtime" / "id_ed25519"
    key.parent.mkdir()
    key.write_text("private-test-key", encoding="utf-8")
    key.with_suffix(".pub").write_text("ssh-ed25519 public-test-key", encoding="utf-8")
    monkeypatch.setattr(core, "runtime_key_path", lambda *_: key)
    fake = FakeClient()

    run_path = core.create_pod(
        root,
        run_id="test-v1a-run",
        approach="v1a-conservative-full",
        document_root=tmp_path / "documents",
        client=fake,
    )

    run = core.read_json(run_path)
    pod = core.read_json(tmp_path / "documents" / "pod.json")
    assert run["pod_id"] == "pod-test-4090"
    assert pod["runtime"]["gpu_type"] == "NVIDIA GeForce RTX 4090"
    assert pod["runtime"]["cloud_type"] == "SECURE"
    assert pod["runtime"]["interruptible"] is False
    assert fake.created[0]["gpuTypeIds"] == ["NVIDIA GeForce RTX 4090"]
    assert "public-test-key" in fake.created[0]["env"]["SSH_PUBLIC_KEY"]
    assert "RUNPOD" not in json.dumps(run)


def test_start_training_refuses_pod_id_mismatch(tmp_path):
    run_path = tmp_path / "run.json"
    core.atomic_json(
        run_path,
        {
            "schema_version": 1,
            "run_id": "test-run",
            "approach": "v1a-conservative-full",
            "pod_id": "expected-pod",
            "repository_root": str(repository_root()),
            "document_root": str(tmp_path),
            "ssh": {"private_key_path": str(tmp_path / "id")},
        },
    )
    with pytest.raises(ValueError, match="pod id mismatch"):
        core.start_training(
            run_path,
            pod_id="wrong-pod",
            data_root=repository_root(),
            client=FakeClient(),
        )


def test_termination_requires_verified_artifacts(tmp_path):
    run_path = tmp_path / "run.json"
    run = {
        "schema_version": 1,
        "run_id": "test-run",
        "approach": "v1a-conservative-full",
        "pod_id": "pod-test-4090",
        "repository_root": str(repository_root()),
        "document_root": str(tmp_path),
        "ssh": {"private_key_path": str(tmp_path / "id")},
    }
    core.atomic_json(run_path, run)
    fake = FakeClient()
    with pytest.raises(RuntimeError, match="refusing to terminate"):
        core.terminate_pod(run_path, client=fake)
    assert fake.terminated == []

    run["artifacts_verified"] = True
    core.atomic_json(run_path, run)
    result = core.terminate_pod(run_path, client=fake)
    assert result["pod_terminated"] is True
    assert fake.terminated == ["pod-test-4090"]


def test_collect_status_records_terminated_provider(tmp_path):
    run_path = tmp_path / "run.json"
    core.atomic_json(
        run_path,
        {
            "schema_version": 1,
            "run_id": "test-run",
            "approach": "v1a-conservative-full",
            "pod_id": "gone-pod",
            "repository_root": str(repository_root()),
            "document_root": str(tmp_path),
            "ssh": {"private_key_path": str(tmp_path / "id")},
        },
    )
    status = core.collect_status(run_path, client=FakeClient())
    assert status["provider_status"] == "TERMINATED"
    assert status["phase"] == "pod_unavailable"
    assert (tmp_path / "status.json").is_file()


def test_build_bundle_contains_training_and_locked_data_roots(tmp_path):
    repository = tmp_path / "repository"
    data = tmp_path / "data"
    (repository / "training").mkdir(parents=True)
    (repository / "training" / "entry.py").write_text("pass\n", encoding="utf-8")
    (data / "data-processing" / "manifests" / "stage11").mkdir(parents=True)
    (data / "data-processing" / "clean_audio" / "medical_16khz").mkdir(
        parents=True
    )
    (data / "data-processing" / "manifests" / "stage11" / "lock.json").write_text(
        "{}\n", encoding="utf-8"
    )
    (data / "data-processing" / "clean_audio" / "medical_16khz" / "one.wav").write_bytes(
        b"RIFF-test"
    )
    bundle = tmp_path / "bundle.tar.gz"
    metadata = core.build_bundle(repository, data, bundle)
    listing = subprocess.run(
        ["tar", "-tzf", str(bundle)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert "training/entry.py" in listing
    assert "data-processing/manifests/stage11/lock.json" in listing
    assert "data-processing/clean_audio/medical_16khz/one.wav" in listing
    assert metadata["sha256"] == core.sha256_file(bundle)


def test_checkpoint_download_is_verified_before_atomic_publish(tmp_path, monkeypatch):
    remote = tmp_path / "remote" / "checkpoint-25"
    remote.mkdir(parents=True)
    weight = remote / "model.safetensors"
    weight.write_bytes(b"verified-weights")
    manifest = {
        "schema_version": 1,
        "checkpoint_step": 25,
        "files": [
            {
                "path": weight.name,
                "bytes": weight.stat().st_size,
                "sha256": core.sha256_file(weight),
            }
        ],
    }
    core.atomic_json(remote / "checkpoint_manifest.json", manifest)
    core.atomic_json(
        remote / "checkpoint_complete.json",
        {
            "schema_version": 1,
            "checkpoint_step": 25,
            "manifest_sha256": core.sha256_file(
                remote / "checkpoint_manifest.json"
            ),
        },
    )

    class FakeTransport:
        def run(self, command, check=False):
            return subprocess.CompletedProcess([], 0, stdout="checkpoint-25\n", stderr="")

        def download_directory(self, remote_path, local_parent):
            shutil.copytree(remote, local_parent / "checkpoint-25")

    run_path = tmp_path / "run.json"
    core.atomic_json(
        run_path,
        {
            "schema_version": 1,
            "run_id": "test-run",
            "approach": "v1a-conservative-full",
            "pod_id": "pod-test",
            "repository_root": str(repository_root()),
            "document_root": str(tmp_path),
            "remote_output": "/remote/output",
            "ssh": {"private_key_path": str(tmp_path / "id")},
        },
    )
    monkeypatch.setattr(core, "_transport_from_run", lambda *_: FakeTransport())
    inventory = core.download_checkpoints(run_path)
    final = tmp_path / "checkpoints" / "checkpoint-25"
    assert inventory["checkpoints"]["checkpoint-25"]["verified"] is True
    assert (final / "model.safetensors").read_bytes() == b"verified-weights"
    assert not list((tmp_path / "checkpoints").glob(".checkpoint-25.download-*"))
