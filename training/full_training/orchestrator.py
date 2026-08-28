from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from training.full_training.checkpoint import verify_checkpoint

CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_runpod_key(env_path: Path) -> str:
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("RUNPOD_API="):
            value = line.split("=", 1)[1].strip().strip("\"'")
            if value:
                return value
    raise RuntimeError("RUNPOD_API is missing from the repository .env")


class RunPodClient:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def request(self, method: str, path: str, body: dict | None = None) -> Any:
        payload = json.dumps(body).encode("utf-8") if body is not None else None
        request = urllib.request.Request(
            f"https://rest.runpod.io/v1{path}",
            data=payload,
            method=method,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                content = response.read()
                return json.loads(content) if content else None
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"RunPod {method} {path} failed ({exc.code}): {detail}"
            ) from exc

    def list_pods(self) -> list[dict[str, Any]]:
        value = self.request("GET", "/pods?includeMachine=true")
        return value if isinstance(value, list) else []

    def get_pod(self, pod_id: str) -> dict[str, Any]:
        value = self.request("GET", f"/pods/{pod_id}?includeMachine=true")
        if not isinstance(value, dict):
            raise RuntimeError(  # noqa: TRY004 - malformed provider response is operational
                f"RunPod returned no state for Pod {pod_id}"
            )
        return value

    def create_pod(self, body: dict[str, Any]) -> dict[str, Any]:
        value = self.request("POST", "/pods", body)
        if not isinstance(value, dict) or not value.get("id"):
            raise RuntimeError("RunPod returned an invalid Pod creation response")
        return value

    def terminate_pod(self, pod_id: str) -> None:
        try:
            self.request("DELETE", f"/pods/{pod_id}")
        except RuntimeError as exc:
            if "(404)" not in str(exc):
                raise


@dataclass(frozen=True)
class SshEndpoint:
    host: str
    port: int


def ssh_endpoint(pod: dict[str, Any]) -> SshEndpoint | None:
    runtime = pod.get("runtime") or {}
    for port in runtime.get("ports") or []:
        if int(port.get("privatePort", 0)) == 22 and port.get("ip"):
            return SshEndpoint(str(port["ip"]), int(port["publicPort"]))
    machine = pod.get("machine") or {}
    mappings = pod.get("portMappings") or machine.get("portMappings") or {}
    public_ip = machine.get("publicIp") or pod.get("publicIp")
    ssh_port = mappings.get("22") or mappings.get(22)
    if public_ip and ssh_port:
        return SshEndpoint(str(public_ip), int(ssh_port))
    return None


class ExperimentOrchestrator:
    def __init__(
        self,
        repository_root: Path,
        config_path: Path,
        poll_seconds: int = 20,
        *,
        resume: bool = False,
        key_path: Path | None = None,
        attach_only: bool = False,
    ):
        self.root = repository_root.resolve()
        self.config_path = config_path.resolve()
        self.config = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        self.variants = tuple(value["id"] for value in self.config["variants"])
        if not self.variants:
            raise ValueError("orchestration requires at least one variant")
        self.variant_config = {
            value["id"]: value for value in self.config["variants"]
        }
        orchestration = self.config.get("orchestration", {})
        credential_file = Path(orchestration.get("credential_file", self.root / ".env"))
        if not credential_file.is_absolute():
            credential_file = self.root / credential_file
        self.client = RunPodClient(_read_runpod_key(credential_file.resolve()))
        self.poll_seconds = poll_seconds
        state_path = Path(
            orchestration.get(
                "state_path", "artifacts/stage11/full-training/orchestrator-state.json"
            )
        )
        self.state_path = (
            state_path if state_path.is_absolute() else self.root / state_path
        ).resolve()
        self.artifact_root = self.state_path.parent
        self.pod_name_prefix = orchestration.get("pod_name_prefix", "ovl-s11-full-")
        self.training_module = orchestration.get(
            "training_module", "training.full_training.run"
        )
        self.remote_config_path = orchestration.get(
            "config_path", "training/config/full_training.yaml"
        )
        source = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=self.root,
            check=True,
            text=True,
            capture_output=True,
        )
        self.source_commit = source.stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=self.root,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
        if dirty:
            raise RuntimeError("refusing to launch training from a dirty source worktree")
        self.resume = resume
        self.attach_only = attach_only
        if resume:
            if key_path is None or not key_path.is_file():
                raise ValueError("--resume requires the existing --key-path")
            self.key_path = key_path.resolve()
            self.temp_root = self.key_path.parent
        else:
            self.temp_root = Path(tempfile.mkdtemp(prefix="openvoice-full-training-"))
            self.key_path = self.temp_root / "id_ed25519"
        self.known_hosts = self.temp_root / "known_hosts"
        bundle_path = Path(
            orchestration.get(
                "bundle_path", "artifacts/stage11/full-training/full-training-input.tar.gz"
            )
        )
        self.bundle = (
            bundle_path if bundle_path.is_absolute() else self.root / bundle_path
        ).resolve()
        self.lock = threading.Lock()
        if resume:
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
            self.run_id = self.state["run_id"]
            self.bundle = Path(self.state["bundle"]["path"])
            self.bundle_sha256 = self.state["bundle"]["sha256"]
        else:
            self.bundle_sha256: str | None = None
            self.run_id = (
                f"{self.config['experiment_id']}-{time.strftime('%Y%m%d-%H%M%S')}"
            )
            self.state: dict[str, Any] = {
                "schema_version": 1,
                "run_id": self.run_id,
                "started_utc": _utc(),
                "status": "initializing",
                "stability_gate_decision": "user_authorized_skip",
                "source_commit": self.source_commit,
                "variants": {},
            }

    def _save_state(self) -> None:
        with self.lock:
            _atomic_json(self.state_path, self.state)

    def _variant_state(self, variant: str, **updates: Any) -> None:
        with self.lock:
            self.state["variants"].setdefault(variant, {}).update(updates)
            _atomic_json(self.state_path, self.state)

    def _create_ssh_key(self) -> None:
        subprocess.run(
            ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(self.key_path)],
            check=True,
        )

    def _build_bundle(self) -> None:
        self.bundle.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.bundle.with_name(self.bundle.name + ".tmp")
        dataset_paths = {self.config["dataset"]["lock"], "data-processing/clean_audio/medical_16khz"}
        dataset_paths.update(
            value["manifest_root"] for value in self.config["variants"]
        )
        command = [
            "tar",
            "-czf",
            str(temporary),
            "training",
            *sorted(dataset_paths),
        ]
        subprocess.run(command, cwd=self.root, check=True)
        os.replace(temporary, self.bundle)
        self.bundle_sha256 = _sha256(self.bundle)
        with self.lock:
            self.state["bundle"] = {
                "path": str(self.bundle),
                "bytes": self.bundle.stat().st_size,
                "sha256": self.bundle_sha256,
            }
            self._save_state_unlocked()

    def _save_state_unlocked(self) -> None:
        _atomic_json(self.state_path, self.state)

    def _relative_output(self, variant: str) -> Path:
        path = Path(self.variant_config[variant]["output_root"])
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"variant output_root must stay repository-relative: {path}")
        return path

    def _local_output(self, variant: str) -> Path:
        return (self.root / self._relative_output(variant)).resolve()

    def _remote_output(self, variant: str) -> str:
        return f"/workspace/OpenVoice-Lab/{self._relative_output(variant).as_posix()}"

    def _pod_body(self, variant: str, public_key: str) -> dict[str, Any]:
        runtime = self.config["runtime"]
        return {
            "name": f"{self.pod_name_prefix}{self.run_id.rsplit('-', 2)[-2]}-{self.run_id.rsplit('-', 1)[-1]}",
            "imageName": runtime["image"],
            "gpuTypeIds": [runtime["gpu_type"]],
            "gpuCount": int(runtime["gpu_count"]),
            "computeType": "GPU",
            "interruptible": bool(runtime["interruptible"]),
            "cloudType": runtime["cloud_type"],
            "containerDiskInGb": 50,
            "volumeInGb": 50,
            "volumeMountPath": "/workspace",
            "ports": ["22/tcp"],
            "supportPublicIp": True,
            "env": {"SSH_PUBLIC_KEY": public_key, "PUBLIC_KEY": public_key},
        }

    def _provision(self) -> dict[str, dict[str, Any]]:
        existing = [
            pod
            for pod in self.client.list_pods()
            if str(pod.get("name", "")).startswith(self.pod_name_prefix)
        ]
        if existing:
            names = ", ".join(str(pod.get("name")) for pod in existing)
            raise RuntimeError(f"existing OpenVoice full-training Pods found: {names}")
        public_key = (
            self.key_path.with_suffix(".pub").read_text(encoding="utf-8").strip()
        )
        created: dict[str, dict[str, Any]] = {}
        try:
            for variant in self.variants:
                last_error: Exception | None = None
                for attempt in range(1, 7):
                    try:
                        pod = self.client.create_pod(
                            self._pod_body(variant, public_key)
                        )
                        created[variant] = pod
                        self._variant_state(
                            variant,
                            status="provisioned",
                            pod_id=pod["id"],
                            pod_name=pod.get("name"),
                            cost_per_hour=pod.get("costPerHr"),
                            created_utc=_utc(),
                        )
                        break
                    except RuntimeError as exc:
                        last_error = exc
                        if (
                            "no instances currently available"
                            not in str(exc).casefold()
                            or attempt == 6
                        ):
                            raise
                        time.sleep(20)
                if variant not in created:
                    raise RuntimeError(f"failed to provision {variant}: {last_error}")
        except Exception:
            for pod in created.values():
                self.client.terminate_pod(pod["id"])
            raise
        return created

    def _wait_endpoint(self, variant: str, pod_id: str) -> SshEndpoint:
        deadline = time.monotonic() + 20 * 60
        while time.monotonic() < deadline:
            pod = self.client.get_pod(pod_id)
            endpoint = ssh_endpoint(pod)
            self._variant_state(
                variant,
                pod_status=pod.get("desiredStatus"),
                last_pod_poll_utc=_utc(),
            )
            if endpoint:
                for _ in range(30):
                    try:
                        result = self._ssh(endpoint, "true", check=False, timeout=20)
                    except subprocess.TimeoutExpired:
                        time.sleep(5)
                        continue
                    if result.returncode == 0:
                        return endpoint
                    time.sleep(5)
            time.sleep(10)
        raise TimeoutError(f"SSH endpoint was not ready for {variant}")

    def _ssh_args(self, endpoint: SshEndpoint) -> list[str]:
        return [
            "ssh",
            "-i",
            str(self.key_path),
            "-p",
            str(endpoint.port),
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            f"UserKnownHostsFile={self.known_hosts}",
            "-o",
            "ConnectTimeout=15",
            "-o",
            "ServerAliveInterval=15",
            "-o",
            "ServerAliveCountMax=4",
            f"root@{endpoint.host}",
        ]

    def _ssh(
        self,
        endpoint: SshEndpoint,
        command: str,
        *,
        check: bool = True,
        timeout: int = 120,
    ) -> subprocess.CompletedProcess[str]:
        last_timeout: subprocess.TimeoutExpired | None = None
        for attempt in range(1, 4):
            try:
                return subprocess.run(
                    [*self._ssh_args(endpoint), command],
                    check=check,
                    timeout=timeout,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    capture_output=True,
                )
            except subprocess.TimeoutExpired as exc:
                last_timeout = exc
                if attempt < 3:
                    time.sleep(3)
        assert last_timeout is not None
        raise last_timeout

    def _scp_args(self, endpoint: SshEndpoint) -> list[str]:
        return [
            "scp",
            "-O",
            "-i",
            str(self.key_path),
            "-P",
            str(endpoint.port),
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            f"UserKnownHostsFile={self.known_hosts}",
            "-o",
            "ConnectTimeout=15",
            "-o",
            "ServerAliveInterval=15",
            "-o",
            "ServerAliveCountMax=4",
        ]

    def _upload_and_launch(
        self, variant: str, pod_id: str, endpoint: SshEndpoint
    ) -> None:
        self._variant_state(
            variant, status="uploading", ssh_host=endpoint.host, ssh_port=endpoint.port
        )
        remote_hash_result = self._ssh(
            endpoint,
            "test -f /workspace/full-training-input.tar.gz && "
            "sha256sum /workspace/full-training-input.tar.gz | cut -d' ' -f1",
            check=False,
        )
        remote_hash = remote_hash_result.stdout.strip()
        if remote_hash != self.bundle_sha256:
            subprocess.run(
                [
                    *self._scp_args(endpoint),
                    str(self.bundle),
                    f"root@{endpoint.host}:/workspace/full-training-input.tar.gz",
                ],
                check=True,
            )
            remote_hash = self._ssh(
                endpoint,
                "sha256sum /workspace/full-training-input.tar.gz | cut -d' ' -f1",
            ).stdout.strip()
        if remote_hash != self.bundle_sha256:
            raise RuntimeError(f"uploaded bundle hash mismatch for {variant}")
        setup = (
            "set -e; mkdir -p /workspace/OpenVoice-Lab; "
            "tar -xzf /workspace/full-training-input.tar.gz -C /workspace/OpenVoice-Lab; "
            "cd /workspace/OpenVoice-Lab; "
            "python -m pip install --disable-pip-version-check --no-cache-dir "
            "-r training/smoke/requirements.txt; "
            f"mkdir -p {self._relative_output(variant).as_posix()}/logs; "
            "df -Pk /workspace | tail -1"
        )
        disk_line = (
            self._ssh(endpoint, setup, timeout=1800).stdout.strip().splitlines()[-1]
        )
        fields = disk_line.split()
        if len(fields) < 4 or int(fields[3]) < 25 * 1024 * 1024:
            raise RuntimeError(
                f"insufficient remote free disk before {variant}: {disk_line}"
            )
        remote_output = self._relative_output(variant).as_posix()
        existing = self._ssh(
            endpoint,
            f"pgrep -f '^python -m {self.training_module} "
            f"--variant {variant} --config {self.remote_config_path}$' "
            "| head -1",
            check=False,
        ).stdout.strip()
        if existing.isdigit():
            self._variant_state(
                variant,
                status="running",
                remote_pid=int(existing),
                controller_attached_utc=_utc(),
            )
            return
        launch = (
            "cd /workspace/OpenVoice-Lab; "
            f"RUNPOD_POD_ID={pod_id} OPENVOICE_RUN_ID={self.run_id} "
            f"OPENVOICE_SOURCE_COMMIT={self.source_commit} PYTHONUNBUFFERED=1 "
            f"nohup python -m {self.training_module} "
            f"--variant {variant} --config {self.remote_config_path} "
            f"> {remote_output}/logs/training.log 2>&1 < /dev/null & echo $!"
        )
        pid = self._ssh(endpoint, launch).stdout.strip().splitlines()[-1]
        if not pid.isdigit():
            raise RuntimeError(
                f"could not capture remote training PID for {variant}: {pid}"
            )
        self._variant_state(
            variant, status="running", remote_pid=int(pid), launched_utc=_utc()
        )

    def _remote_completed_checkpoints(
        self, endpoint: SshEndpoint, variant: str
    ) -> list[str]:
        command = (
            f"find {self._remote_output(variant)}/checkpoints "
            "-mindepth 2 -maxdepth 2 -name checkpoint_complete.json -printf '%h\\n' "
            "2>/dev/null | sed 's#.*/##' | sort -V"
        )
        result = self._ssh(endpoint, command, check=False)
        if result.returncode not in (0, 1):
            raise RuntimeError(
                result.stderr.strip() or "remote checkpoint listing failed"
            )
        return [
            line.strip()
            for line in result.stdout.splitlines()
            if CHECKPOINT_RE.fullmatch(line.strip())
        ]

    def _download_checkpoint(
        self, endpoint: SshEndpoint, variant: str, checkpoint_name: str
    ) -> dict[str, Any]:
        local_root = self._local_output(variant) / "checkpoints"
        local_root.mkdir(parents=True, exist_ok=True)
        final = local_root / checkpoint_name
        if final.exists():
            valid, reason = verify_checkpoint(final)
            if not valid:
                raise RuntimeError(
                    f"existing local {checkpoint_name} is invalid: {reason}"
                )
            return {
                "checkpoint": checkpoint_name,
                "verified": True,
                "already_present": True,
            }
        temporary = local_root / f".{checkpoint_name}.download-{uuid.uuid4().hex}"
        temporary.mkdir()
        remote = (
            f"root@{endpoint.host}:{self._remote_output(variant)}/checkpoints/"
            f"{checkpoint_name}"
        )
        try:
            subprocess.run(
                [*self._scp_args(endpoint), "-r", remote, str(temporary)], check=True
            )
            downloaded = temporary / checkpoint_name
            valid, reason = verify_checkpoint(downloaded)
            if not valid:
                raise RuntimeError(
                    f"downloaded {variant} {checkpoint_name} is invalid: {reason}"
                )
            os.replace(downloaded, final)
            shutil.rmtree(temporary)
            return {
                "checkpoint": checkpoint_name,
                "verified": True,
                "downloaded_utc": _utc(),
                "path": str(final),
                "manifest_sha256": _sha256(final / "checkpoint_manifest.json"),
            }
        except Exception:
            if temporary.parent == local_root and temporary.name.startswith(
                f".{checkpoint_name}.download-"
            ):
                shutil.rmtree(temporary, ignore_errors=True)
            raise

    def _download_final_artifacts(
        self, endpoint: SshEndpoint, variant: str
    ) -> dict[str, Any]:
        local_output = self._local_output(variant)
        local_output.mkdir(parents=True, exist_ok=True)
        remote_root = f"root@{endpoint.host}:{self._remote_output(variant)}"
        names = [
            "selected-model",
            "evaluation",
            "run_provenance.json",
            "training_metadata.json",
            "initial_parameter_inventory.json",
            "parameter_inventory.json",
            "checkpoint-probes",
            "pip-freeze.txt",
            "run_result.json",
            "run_artifact_manifest.json",
            "RUN_COMPLETE.json",
            "progress.json",
            "feature_progress.json",
            "logs",
        ]
        for name in names:
            destination = local_output / name
            if destination.exists():
                continue
            subprocess.run(
                [
                    *self._scp_args(endpoint),
                    "-r",
                    f"{remote_root}/{name}",
                    str(local_output),
                ],
                check=True,
            )
        marker = json.loads(
            (local_output / "RUN_COMPLETE.json").read_text(encoding="utf-8")
        )
        manifest_path = local_output / "run_artifact_manifest.json"
        if marker["artifact_manifest_sha256"] != _sha256(manifest_path):
            raise RuntimeError(f"final artifact manifest hash mismatch for {variant}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for entry in manifest["files"]:
            file = local_output / entry["path"]
            if not file.is_file() or file.stat().st_size != int(entry["bytes"]):
                raise RuntimeError(
                    f"missing or truncated final artifact {variant}/{entry['path']}"
                )
            if _sha256(file) != entry["sha256"]:
                raise RuntimeError(
                    f"final artifact hash mismatch {variant}/{entry['path']}"
                )
        return {
            "verified": True,
            "verified_utc": _utc(),
            "manifest_sha256": _sha256(manifest_path),
            "file_count": len(manifest["files"]),
        }

    def _capture_remote_file(
        self, endpoint: SshEndpoint, remote: str, local: Path
    ) -> bool:
        result = self._ssh(endpoint, f"test -f {remote} && cat {remote}", check=False)
        if result.returncode != 0 or not result.stdout.strip():
            return False
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_text(result.stdout, encoding="utf-8")
        return True

    def _capture_telemetry(self, endpoint: SshEndpoint, variant: str) -> dict[str, Any]:
        command = (
            "gpu=$(nvidia-smi --query-gpu=timestamp,memory.used,memory.total,"
            "utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits "
            "2>/dev/null | head -1); "
            "disk=$(df -Pk /workspace | tail -1 | awk '{print $4}'); "
            "printf '%s|%s' \"$gpu\" \"$disk\""
        )
        result = self._ssh(endpoint, command, check=False)
        fields = [value.strip() for value in result.stdout.strip().split("|")]
        payload: dict[str, Any] = {"captured_utc": _utc(), "raw": result.stdout.strip()}
        if len(fields) == 2:
            gpu = [value.strip() for value in fields[0].split(",")]
            if len(gpu) == 6:
                payload.update(
                    gpu_timestamp=gpu[0],
                    gpu_memory_used_mb=float(gpu[1]),
                    gpu_memory_total_mb=float(gpu[2]),
                    gpu_utilization_percent=float(gpu[3]),
                    gpu_temperature_c=float(gpu[4]),
                    gpu_power_w=float(gpu[5]),
                )
            if fields[1].isdigit():
                payload["workspace_disk_free_gb"] = int(fields[1]) / 1024**2
        telemetry = self._local_output(variant) / "telemetry.jsonl"
        telemetry.parent.mkdir(parents=True, exist_ok=True)
        with telemetry.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
        return payload

    def _monitor_variant(self, variant: str, pod: dict[str, Any]) -> dict[str, Any]:
        pod_id = pod["id"]
        endpoint = self._wait_endpoint(variant, pod_id)
        if self.attach_only:
            remote_pid = self._ssh(
                endpoint,
                f"pgrep -f '^python -m {self.training_module} "
                f"--variant {variant} --config {self.remote_config_path}$' "
                "| head -1",
                check=False,
            ).stdout.strip()
            if not remote_pid.isdigit():
                remote_output = (
                    self._remote_output(variant)
                )
                terminal = (
                    self._ssh(
                        endpoint,
                        f"test -f {remote_output}/RUN_COMPLETE.json -o "
                        f"-f {remote_output}/RUN_FAILED.json",
                        check=False,
                    ).returncode
                    == 0
                )
                if not terminal:
                    raise RuntimeError(
                        f"no running or terminal trainer found to attach for {variant}"
                    )
                remote_pid = "0"
            self._variant_state(
                variant,
                status="running",
                remote_pid=int(remote_pid),
                controller_attached_utc=_utc(),
            )
        else:
            self._upload_and_launch(variant, pod_id, endpoint)
        downloaded: dict[str, Any] = {}
        remote_output = self._remote_output(variant)
        while True:
            telemetry = self._capture_telemetry(endpoint, variant)
            self._variant_state(variant, latest_telemetry=telemetry)
            for checkpoint_name in self._remote_completed_checkpoints(
                endpoint, variant
            ):
                if checkpoint_name in downloaded:
                    continue
                record = self._download_checkpoint(endpoint, variant, checkpoint_name)
                downloaded[checkpoint_name] = record
                self._variant_state(
                    variant,
                    checkpoints=downloaded,
                    latest_checkpoint=checkpoint_name,
                    last_checkpoint_download_utc=_utc(),
                )
            progress_local = self._local_output(variant) / "live-progress.json"
            if self._capture_remote_file(
                endpoint, f"{remote_output}/progress.json", progress_local
            ):
                progress = json.loads(progress_local.read_text(encoding="utf-8"))
                self._variant_state(variant, progress=progress)
            complete = (
                self._ssh(
                    endpoint, f"test -f {remote_output}/RUN_COMPLETE.json", check=False
                ).returncode
                == 0
            )
            failed = (
                self._ssh(
                    endpoint, f"test -f {remote_output}/RUN_FAILED.json", check=False
                ).returncode
                == 0
            )
            if complete:
                for checkpoint_name in self._remote_completed_checkpoints(
                    endpoint, variant
                ):
                    if checkpoint_name not in downloaded:
                        record = self._download_checkpoint(
                            endpoint, variant, checkpoint_name
                        )
                        downloaded[checkpoint_name] = record
                final = self._download_final_artifacts(endpoint, variant)
                result = json.loads(
                    (self._local_output(variant) / "run_result.json").read_text(
                        encoding="utf-8"
                    )
                )
                self._variant_state(
                    variant,
                    status="completed_and_verified",
                    checkpoints=downloaded,
                    final_artifacts=final,
                    completed_utc=_utc(),
                    result_status=result["status"],
                )
                self.client.terminate_pod(pod_id)
                self._variant_state(
                    variant, pod_terminated=True, pod_terminated_utc=_utc()
                )
                return result
            if failed:
                failure_local = self._local_output(variant) / "RUN_FAILED.json"
                log_local = self._local_output(variant) / "logs" / "training.log"
                self._capture_remote_file(
                    endpoint, f"{remote_output}/RUN_FAILED.json", failure_local
                )
                self._capture_remote_file(
                    endpoint, f"{remote_output}/logs/training.log", log_local
                )
                failure = json.loads(failure_local.read_text(encoding="utf-8"))
                self._variant_state(
                    variant, status="failed", failure=failure, checkpoints=downloaded
                )
                self.client.terminate_pod(pod_id)
                self._variant_state(
                    variant, pod_terminated=True, pod_terminated_utc=_utc()
                )
                raise RuntimeError(f"{variant} training failed: {failure['error']}")
            process = self._ssh(
                endpoint,
                f"kill -0 {self.state['variants'][variant]['remote_pid']} 2>/dev/null",
                check=False,
            )
            if process.returncode != 0:
                log_local = self._local_output(variant) / "logs" / "training.log"
                self._capture_remote_file(
                    endpoint, f"{remote_output}/logs/training.log", log_local
                )
                raise RuntimeError(
                    f"{variant} training process exited without a terminal marker"
                )
            time.sleep(self.poll_seconds)

    def run(self) -> dict[str, Any]:
        if self.resume:
            pods = {
                variant: {
                    "id": self.state["variants"][variant]["pod_id"],
                    "name": self.state["variants"][variant].get("pod_name"),
                    "costPerHr": self.state["variants"][variant].get("cost_per_hour"),
                }
                for variant in self.variants
            }
            for variant in self.variants:
                self._variant_state(
                    variant,
                    status="resuming_orchestration",
                    orchestrator_error=None,
                    controller_resumed_utc=_utc(),
                )
        else:
            self._save_state()
            self._create_ssh_key()
            self._build_bundle()
            pods = self._provision()
        with self.lock:
            self.state["status"] = "running"
            self._save_state_unlocked()
        results: dict[str, Any] = {}
        failures: dict[str, str] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(self._monitor_variant, variant, pods[variant]): variant
                for variant in self.variants
            }
            for future in concurrent.futures.as_completed(futures):
                variant = futures[future]
                try:
                    results[variant] = future.result()
                except Exception as exc:  # noqa: BLE001 - preserve all run failures
                    failures[variant] = f"{type(exc).__name__}: {exc}"
                    self._variant_state(
                        variant, status="failed", orchestrator_error=failures[variant]
                    )
        for variant, pod in pods.items():
            state = self.state["variants"].get(variant, {})
            if not state.get("pod_terminated"):
                self.client.terminate_pod(pod["id"])
                self._variant_state(
                    variant, pod_terminated=True, pod_terminated_utc=_utc()
                )
        with self.lock:
            self.state.update(
                status="completed" if not failures else "failed",
                completed_utc=_utc(),
                failures=failures,
                results={
                    variant: result.get("status") for variant, result in results.items()
                },
            )
            self._save_state_unlocked()
        if failures:
            raise RuntimeError(
                "; ".join(f"{key}: {value}" for key, value in failures.items())
            )
        return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all three Stage 11 experiments on RunPod"
    )
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config", type=Path, default=Path("training/config/full_training.yaml")
    )
    parser.add_argument("--poll-seconds", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--key-path", type=Path)
    parser.add_argument("--attach-only", action="store_true")
    args = parser.parse_args()
    orchestrator = ExperimentOrchestrator(
        args.repository_root,
        args.config,
        poll_seconds=args.poll_seconds,
        resume=args.resume,
        key_path=args.key_path,
        attach_only=args.attach_only,
    )
    results = orchestrator.run()
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
