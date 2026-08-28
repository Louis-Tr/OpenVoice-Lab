from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from training.full_training.checkpoint import verify_checkpoint
from training.full_training.config import build_preflight_report, load_config
from training.v1_approaches import COMPATIBILITY_MODULES, PROFILES

RUN_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{2,80}$")
CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")
DEFAULT_IMAGE = "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04"
DEFAULT_GPU = "NVIDIA GeForce RTX 4090"


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def append_event(path: Path, event: str, **details: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"recorded_utc": utc_now(), "event": event, **details}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def read_env_value(env_path: Path, name: str) -> str:
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        if key.strip() == name:
            result = value.strip().strip("\"'")
            if result:
                return result
    raise RuntimeError(f"{name} is missing from {env_path}")


def validate_run_id(run_id: str) -> str:
    if not RUN_ID_RE.fullmatch(run_id):
        raise ValueError(
            "run id must be 3-81 lowercase letters, digits, or hyphens and start "
            "with a letter or digit"
        )
    return run_id


def repository_root_from(path: Path) -> Path:
    resolved = path.resolve()
    for candidate in (resolved, *resolved.parents):
        if (candidate / ".git").exists():
            return candidate
    raise ValueError(f"could not find repository root above {path}")


@dataclass(frozen=True)
class RunPaths:
    root: Path
    pod: Path
    run: Path
    status: Path
    events: Path
    inventory: Path
    bundle: Path

    @classmethod
    def create(cls, repository_root: Path, run_id: str, root: Path | None = None):
        document_root = (
            root
            if root is not None
            else repository_root / "artifacts" / "stage11" / "agent-runs" / run_id
        ).resolve()
        return cls(
            root=document_root,
            pod=document_root / "pod.json",
            run=document_root / "run.json",
            status=document_root / "status.json",
            events=document_root / "events.jsonl",
            inventory=document_root / "checkpoint-inventory.json",
            bundle=document_root / "training-input.tar.gz",
        )


class RunPodClient:
    def __init__(self, api_key: str, api_base: str = "https://rest.runpod.io/v1"):
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")

    def request(self, method: str, path: str, body: dict | None = None) -> Any:
        payload = json.dumps(body).encode("utf-8") if body is not None else None
        request = urllib.request.Request(
            f"{self.api_base}{path}",
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

    def create_pod(self, body: dict[str, Any]) -> dict[str, Any]:
        value = self.request("POST", "/pods", body)
        if not isinstance(value, dict) or not value.get("id"):
            raise RuntimeError("RunPod returned an invalid pod creation response")
        return value

    def get_pod(self, pod_id: str) -> dict[str, Any]:
        value = self.request("GET", f"/pods/{pod_id}?includeMachine=true")
        if not isinstance(value, dict):
            raise RuntimeError(f"RunPod returned no state for pod {pod_id}")
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


class SshTransport:
    def __init__(self, endpoint: SshEndpoint, key_path: Path, known_hosts: Path):
        self.endpoint = endpoint
        self.key_path = key_path
        self.known_hosts = known_hosts

    def _common(self) -> list[str]:
        return [
            "-i",
            str(self.key_path),
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

    def run(
        self,
        command: str,
        *,
        check: bool = True,
        timeout: int = 120,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                "ssh",
                *self._common(),
                "-p",
                str(self.endpoint.port),
                f"root@{self.endpoint.host}",
                command,
            ],
            check=check,
            timeout=timeout,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
        )

    def upload(self, source: Path, remote_path: str) -> None:
        subprocess.run(
            [
                "scp",
                "-O",
                *self._common(),
                "-P",
                str(self.endpoint.port),
                str(source),
                f"root@{self.endpoint.host}:{remote_path}",
            ],
            check=True,
        )

    def download_directory(self, remote_path: str, local_parent: Path) -> None:
        subprocess.run(
            [
                "scp",
                "-O",
                *self._common(),
                "-P",
                str(self.endpoint.port),
                "-r",
                f"root@{self.endpoint.host}:{remote_path}",
                str(local_parent),
            ],
            check=True,
        )

    def download_file(self, remote_path: str, local_path: Path) -> bool:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            [
                "scp",
                "-O",
                *self._common(),
                "-P",
                str(self.endpoint.port),
                f"root@{self.endpoint.host}:{remote_path}",
                str(local_path),
            ],
            check=False,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
        )
        return result.returncode == 0


def load_profile(repository_root: Path, approach: str) -> dict[str, Any]:
    if approach not in PROFILES:
        raise ValueError(
            f"unknown approach {approach!r}; expected one of {', '.join(PROFILES)}"
        )
    module, config_relative = PROFILES[approach]
    config_path = (repository_root / config_relative).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    variant = next(item for item in config["variants"] if item["id"] == approach)
    return {
        "approach": approach,
        "module": module,
        "config_relative": config_relative,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "variant": variant,
        "models": config["models"],
        "runtime": config["runtime"],
        "dataset": config["dataset"],
    }


def runtime_key_path(repository_root: Path, run_id: str) -> Path:
    return (
        repository_root / ".runtime" / "runpod-agent" / run_id / "id_ed25519"
    ).resolve()


def ensure_ssh_key(path: Path) -> Path:
    if path.is_file() and path.with_suffix(".pub").is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.with_suffix(".pub").exists():
        raise RuntimeError(f"incomplete SSH key pair exists at {path}")
    subprocess.run(
        ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(path)],
        check=True,
    )
    return path


def client_from_repository(repository_root: Path) -> RunPodClient:
    return RunPodClient(read_env_value(repository_root / ".env", "RUNPOD_API"))


def create_pod(
    repository_root: Path,
    *,
    run_id: str,
    approach: str,
    document_root: Path | None = None,
    client: RunPodClient | None = None,
) -> Path:
    repository_root = repository_root.resolve()
    run_id = validate_run_id(run_id)
    profile = load_profile(repository_root, approach)
    paths = RunPaths.create(repository_root, run_id, document_root)
    if paths.run.exists() or paths.pod.exists():
        raise FileExistsError(
            f"run documents already exist; use them instead of creating a duplicate: "
            f"{paths.root}"
        )
    preflight = build_preflight_report(load_config(profile["config_path"]))
    if not preflight["configuration_valid"] or not preflight["launch_ready"]:
        raise RuntimeError(
            "refusing to create a billable pod because the profile is not launch-ready"
        )
    atomic_json(paths.root / "preflight.json", preflight)
    key_path = ensure_ssh_key(runtime_key_path(repository_root, run_id))
    runtime = profile["runtime"]
    public_key = key_path.with_suffix(".pub").read_text(encoding="utf-8").strip()
    pod_body = {
        "name": f"ovl-{run_id}",
        "imageName": runtime.get("image", DEFAULT_IMAGE),
        "gpuTypeIds": [runtime.get("gpu_type", DEFAULT_GPU)],
        "gpuCount": int(runtime.get("gpu_count", 1)),
        "computeType": "GPU",
        "interruptible": bool(runtime.get("interruptible", False)),
        "cloudType": runtime.get("cloud_type", "SECURE"),
        "containerDiskInGb": 50,
        "volumeInGb": 50,
        "volumeMountPath": "/workspace",
        "ports": ["22/tcp"],
        "supportPublicIp": True,
        "env": {"SSH_PUBLIC_KEY": public_key, "PUBLIC_KEY": public_key},
    }
    initial = {
        "schema_version": 1,
        "run_id": run_id,
        "approach": approach,
        "status": "provisioning",
        "created_utc": utc_now(),
        "repository_root": str(repository_root),
        "document_root": str(paths.root),
        "profile": profile,
        "ssh": {"private_key_path": str(key_path)},
    }
    atomic_json(paths.run, initial)
    append_event(paths.events, "pod_create_requested", approach=approach)
    provider = client or client_from_repository(repository_root)
    try:
        pod = provider.create_pod(pod_body)
    except Exception as exc:
        initial.update(status="pod_create_failed", error=f"{type(exc).__name__}: {exc}")
        atomic_json(paths.run, initial)
        append_event(paths.events, "pod_create_failed", error=initial["error"])
        raise
    pod_document = {
        "schema_version": 1,
        "run_id": run_id,
        "pod_id": pod["id"],
        "pod_name": pod.get("name", pod_body["name"]),
        "status": pod.get("desiredStatus", "CREATED"),
        "created_utc": utc_now(),
        "cost_per_hour": pod.get("costPerHr"),
        "runtime": {
            "gpu_type": runtime.get("gpu_type", DEFAULT_GPU),
            "gpu_count": int(runtime.get("gpu_count", 1)),
            "cloud_type": runtime.get("cloud_type", "SECURE"),
            "interruptible": bool(runtime.get("interruptible", False)),
            "image": runtime.get("image", DEFAULT_IMAGE),
        },
        "ssh": {"private_key_path": str(key_path), "endpoint": None},
    }
    atomic_json(paths.pod, pod_document)
    initial.update(
        status="pod_created",
        pod_id=pod["id"],
        pod_document=str(paths.pod),
        updated_utc=utc_now(),
    )
    atomic_json(paths.run, initial)
    append_event(
        paths.events,
        "pod_created",
        pod_id=pod["id"],
        cost_per_hour=pod.get("costPerHr"),
    )
    return paths.run


def wait_for_endpoint(
    client: RunPodClient,
    pod_id: str,
    *,
    timeout_seconds: int = 1200,
    poll_seconds: int = 10,
) -> tuple[dict[str, Any], SshEndpoint]:
    deadline = time.monotonic() + timeout_seconds
    last_status = None
    while time.monotonic() < deadline:
        pod = client.get_pod(pod_id)
        last_status = pod.get("desiredStatus")
        endpoint = ssh_endpoint(pod)
        if endpoint is not None:
            return pod, endpoint
        time.sleep(poll_seconds)
    raise TimeoutError(
        f"pod {pod_id} did not expose SSH within {timeout_seconds}s; "
        f"last status={last_status}"
    )


def build_bundle(
    repository_root: Path,
    data_root: Path,
    destination: Path,
) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    command = [
        "tar",
        "-czf",
        str(temporary),
        "-C",
        str(repository_root),
        "training",
        "-C",
        str(data_root),
        "data-processing/manifests/stage11",
        "data-processing/clean_audio/medical_16khz",
    ]
    subprocess.run(command, check=True)
    os.replace(temporary, destination)
    return {
        "path": str(destination),
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
        "created_utc": utc_now(),
    }


def source_state(repository_root: Path) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty_lines = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {
        "commit": commit,
        "dirty": bool(dirty_lines),
        "dirty_path_count": len(dirty_lines),
    }


def _transport_from_run(
    run: dict[str, Any], endpoint: SshEndpoint | None = None
) -> SshTransport:
    key_path = Path(run["ssh"]["private_key_path"])
    if endpoint is None:
        endpoint_value = run["ssh"].get("endpoint")
        if not endpoint_value:
            raise RuntimeError("run document has no SSH endpoint")
        endpoint = SshEndpoint(endpoint_value["host"], int(endpoint_value["port"]))
    return SshTransport(
        endpoint,
        key_path,
        key_path.parent / "known_hosts",
    )


def start_training(
    run_document: Path,
    *,
    pod_id: str,
    data_root: Path,
    client: RunPodClient | None = None,
) -> dict[str, Any]:
    run_document = run_document.resolve()
    run = read_json(run_document)
    if run.get("pod_id") != pod_id:
        raise ValueError(
            f"pod id mismatch: document has {run.get('pod_id')}, command has {pod_id}"
        )
    repository_root = Path(run["repository_root"])
    paths = RunPaths.create(repository_root, run["run_id"], Path(run["document_root"]))
    profile = load_profile(repository_root, run["approach"])
    if run.get("profile", {}).get("config_sha256") != profile["config_sha256"]:
        raise RuntimeError(
            "training profile changed after pod creation; create a new run document "
            "instead of mutating an existing run"
        )
    source = source_state(repository_root)
    if source["dirty"]:
        raise RuntimeError(
            "training source is dirty; commit the implementation before launch so "
            "the run can be reproduced"
        )
    config = load_config(profile["config_path"])
    preflight = build_preflight_report(config)
    preflight_path = paths.root / "preflight.json"
    atomic_json(preflight_path, preflight)
    if not preflight["configuration_valid"] or not preflight["launch_ready"]:
        raise RuntimeError("training profile failed launch-ready preflight validation")
    provider = client or client_from_repository(repository_root)
    pod, endpoint = wait_for_endpoint(provider, pod_id)
    transport = _transport_from_run(run, endpoint)
    for _ in range(30):
        if transport.run("true", check=False, timeout=20).returncode == 0:
            break
        time.sleep(5)
    else:
        raise TimeoutError(f"SSH did not become ready for pod {pod_id}")

    bundle = build_bundle(repository_root, data_root.resolve(), paths.bundle)
    remote_bundle = f"/workspace/{run['run_id']}-training-input.tar.gz"
    remote_hash = transport.run(
        f"test -f {remote_bundle} && sha256sum {remote_bundle} | cut -d' ' -f1",
        check=False,
    ).stdout.strip()
    if remote_hash != bundle["sha256"]:
        transport.upload(paths.bundle, remote_bundle)
        observed = transport.run(
            f"sha256sum {remote_bundle} | cut -d' ' -f1"
        ).stdout.strip()
        if observed != bundle["sha256"]:
            raise RuntimeError("uploaded training bundle failed SHA-256 verification")

    output_relative = profile["variant"]["output_root"].replace("\\", "/")
    remote_repository = f"/workspace/OpenVoice-Lab-runs/{run['run_id']}"
    setup = (
        f"set -e; mkdir -p {shlex.quote(remote_repository)}; "
        f"tar -xzf {shlex.quote(remote_bundle)} -C {shlex.quote(remote_repository)}; "
        f"cd {shlex.quote(remote_repository)}; "
        "python -m pip install --disable-pip-version-check --no-cache-dir "
        "-r training/smoke/requirements.txt; "
        f"mkdir -p {shlex.quote(output_relative)}/logs; "
        "df -Pk /workspace | tail -1"
    )
    disk_line = transport.run(setup, timeout=1800).stdout.strip().splitlines()[-1]
    fields = disk_line.split()
    if len(fields) < 4 or int(fields[3]) < 25 * 1024 * 1024:
        raise RuntimeError(f"insufficient pod disk before training: {disk_line}")

    config_relative = profile["config_relative"].replace("\\", "/")
    remote_output = f"{remote_repository}/{output_relative}"
    compatibility_module = COMPATIBILITY_MODULES.get(run["approach"])
    compatibility = run.get("compatibility", {"status": "not_required"})
    if compatibility_module and compatibility.get("status") != "passed":
        remote_log = f"{remote_output}/logs/compatibility.log"
        compatibility_command = (
            f"cd {shlex.quote(remote_repository)}; "
            f"python -m {compatibility_module} --config "
            f"{shlex.quote(config_relative)} > {shlex.quote(remote_log)} 2>&1"
        )
        compatibility_result = transport.run(
            compatibility_command, check=False, timeout=1800
        )
        local_log = paths.root / "compatibility.log"
        transport.download_file(remote_log, local_log)
        compatibility = {
            "status": (
                "passed" if compatibility_result.returncode == 0 else "failed"
            ),
            "module": compatibility_module,
            "log": str(local_log),
            "completed_utc": utc_now(),
        }
        if compatibility_result.returncode != 0:
            run.update(
                status="compatibility_failed",
                updated_utc=utc_now(),
                bundle=bundle,
                compatibility=compatibility,
                ssh={
                    **run["ssh"],
                    "endpoint": {"host": endpoint.host, "port": endpoint.port},
                },
            )
            atomic_json(paths.run, run)
            append_event(
                paths.events,
                "compatibility_failed",
                module=compatibility_module,
                log=str(local_log),
            )
            raise RuntimeError(
                f"{run['approach']} compatibility check failed; evidence: {local_log}"
            )
        append_event(
            paths.events, "compatibility_passed", module=compatibility_module
        )
    training_command = (
        f"python -m {profile['module']} "
        f"--variant {shlex.quote(run['approach'])} "
        f"--config {shlex.quote(config_relative)}"
    )
    existing = transport.run(
        f"pgrep -f {shlex.quote('^' + training_command + '$')} | head -1",
        check=False,
    ).stdout.strip()
    if existing.isdigit():
        pid = int(existing)
        event = "training_reattached"
    else:
        launch = (
            f"cd {shlex.quote(remote_repository)}; "
            f"RUNPOD_POD_ID={shlex.quote(pod_id)} "
            f"OPENVOICE_RUN_ID={shlex.quote(run['run_id'])} "
            "PYTHONUNBUFFERED=1 nohup "
            f"{training_command} > {shlex.quote(remote_output)}/logs/training.log "
            "2>&1 < /dev/null & echo $!"
        )
        value = transport.run(launch).stdout.strip().splitlines()[-1]
        if not value.isdigit():
            raise RuntimeError(f"could not capture remote training PID: {value}")
        pid = int(value)
        event = "training_launched"

    endpoint_value = {"host": endpoint.host, "port": endpoint.port}
    run.update(
        status="running",
        updated_utc=utc_now(),
        pod_id=pod_id,
        remote_pid=pid,
        remote_repository=remote_repository,
        remote_output=f"{remote_repository}/{output_relative}",
        training_command=training_command,
        compatibility=compatibility,
        bundle=bundle,
        source=source,
        preflight={
            "path": str(preflight_path),
            "configuration_sha256": preflight["configuration_sha256"],
            "dataset_lock_sha256": preflight["dataset_lock"]["sha256"],
            "manifest_sha256": {
                variant: {
                    split: details["sha256"]
                    for split, details in splits.items()
                }
                for variant, splits in preflight["manifests"].items()
            },
        },
        profile=profile,
        ssh={
            **run["ssh"],
            "endpoint": endpoint_value,
        },
    )
    atomic_json(paths.run, run)
    pod_document = read_json(paths.pod)
    pod_document.update(
        status=pod.get("desiredStatus"),
        last_provider_poll_utc=utc_now(),
        ssh={**pod_document["ssh"], "endpoint": endpoint_value},
    )
    atomic_json(paths.pod, pod_document)
    append_event(paths.events, event, pod_id=pod_id, remote_pid=pid)
    return run


def _remote_json(transport: SshTransport, path: str) -> dict[str, Any] | None:
    result = transport.run(f"cat {shlex.quote(path)}", check=False)
    if result.returncode != 0 or not result.stdout.strip():
        return None
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def collect_status(
    run_document: Path,
    *,
    client: RunPodClient | None = None,
) -> dict[str, Any]:
    run_document = run_document.resolve()
    run = read_json(run_document)
    repository_root = Path(run["repository_root"])
    paths = RunPaths.create(repository_root, run["run_id"], Path(run["document_root"]))
    provider = client or client_from_repository(repository_root)
    try:
        pod = provider.get_pod(run["pod_id"])
        provider_status = pod.get("desiredStatus")
        endpoint = ssh_endpoint(pod)
    except RuntimeError as exc:
        if "(404)" not in str(exc):
            raise
        pod = None
        provider_status = "TERMINATED"
        endpoint = None

    remote: dict[str, Any] = {}
    phase = "pod_unavailable"
    if endpoint is not None:
        transport = _transport_from_run(run, endpoint)
        output = run.get("remote_output")
        if output:
            remote_pid = int(run.get("remote_pid") or 0)
            process_running = remote_pid > 0 and transport.run(
                f"kill -0 {remote_pid}", check=False
            ).returncode == 0
            complete = transport.run(
                f"test -f {shlex.quote(output + '/RUN_COMPLETE.json')}", check=False
            ).returncode == 0
            failed = transport.run(
                f"test -f {shlex.quote(output + '/RUN_FAILED.json')}", check=False
            ).returncode == 0
            progress = _remote_json(transport, output + "/progress.json")
            evaluation = _remote_json(transport, output + "/evaluation/progress.json")
            latest = _remote_json(transport, output + "/latest_checkpoint.json")
            checkpoint_result = transport.run(
                f"find {shlex.quote(output + '/checkpoints')} -mindepth 2 -maxdepth 2 "
                "-name checkpoint_complete.json -printf '%h\\n' 2>/dev/null "
                "| sed 's#.*/##' | sort -V",
                check=False,
            )
            telemetry = transport.run(
                "nvidia-smi --query-gpu=timestamp,memory.used,memory.total,"
                "utilization.gpu,temperature.gpu,power.draw "
                "--format=csv,noheader,nounits | head -1; "
                "df -Pk /workspace | tail -1",
                check=False,
            ).stdout.splitlines()
            gpu = None
            disk = None
            if telemetry:
                fields = [value.strip() for value in telemetry[0].split(",")]
                if len(fields) == 6:
                    gpu = {
                        "timestamp": fields[0],
                        "memory_used_mb": float(fields[1]),
                        "memory_total_mb": float(fields[2]),
                        "utilization_percent": float(fields[3]),
                        "temperature_c": float(fields[4]),
                        "power_w": float(fields[5]),
                    }
            if len(telemetry) > 1:
                disk_fields = telemetry[1].split()
                if len(disk_fields) >= 4:
                    disk = {
                        "filesystem": disk_fields[0],
                        "available_kb": int(disk_fields[3]),
                        "available_gb": int(disk_fields[3]) / 1024**2,
                    }
            remote = {
                "process_running": process_running,
                "progress": progress,
                "evaluation_progress": evaluation,
                "latest_checkpoint": latest,
                "completed_checkpoints": [
                    line.strip()
                    for line in checkpoint_result.stdout.splitlines()
                    if CHECKPOINT_RE.fullmatch(line.strip())
                ],
                "gpu": gpu,
                "disk": disk,
                "telemetry_raw": telemetry,
            }
            if complete:
                phase = "completed_remote"
            elif failed:
                phase = "failed_remote"
            elif process_running and evaluation:
                phase = "evaluating"
            elif process_running and progress:
                phase = "training"
            elif process_running:
                phase = "preparing"
            else:
                phase = "process_stopped"
        else:
            phase = "pod_ready_not_started"

    estimated_cost = None
    if paths.pod.exists():
        pod_document = read_json(paths.pod)
        hourly = pod_document.get("cost_per_hour")
        created = pod_document.get("created_utc")
        if hourly is not None and created:
            started = datetime.fromisoformat(created.replace("Z", "+00:00"))
            hours = max(
                0.0,
                (datetime.now(timezone.utc) - started).total_seconds() / 3600,
            )
            estimated_cost = round(hours * float(hourly), 4)
    status = {
        "schema_version": 1,
        "run_id": run["run_id"],
        "approach": run["approach"],
        "pod_id": run["pod_id"],
        "provider_status": provider_status,
        "phase": phase,
        "checked_utc": utc_now(),
        "estimated_cost_usd": estimated_cost,
        "remote": remote,
    }
    atomic_json(paths.status, status)
    if endpoint is not None:
        run["ssh"] = {
            **run["ssh"],
            "endpoint": {"host": endpoint.host, "port": endpoint.port},
        }
    run.update(status=phase, updated_utc=status["checked_utc"])
    atomic_json(paths.run, run)
    if paths.pod.exists():
        pod_document = read_json(paths.pod)
        pod_document.update(
            status=provider_status,
            last_provider_poll_utc=status["checked_utc"],
        )
        atomic_json(paths.pod, pod_document)
    append_event(paths.events, "status_checked", phase=phase)
    return status


def _completed_remote_checkpoints(
    transport: SshTransport, remote_output: str
) -> list[str]:
    result = transport.run(
        f"find {shlex.quote(remote_output + '/checkpoints')} "
        "-mindepth 2 -maxdepth 2 -name checkpoint_complete.json "
        "-printf '%h\\n' 2>/dev/null | sed 's#.*/##' | sort -V",
        check=False,
    )
    return [
        line.strip()
        for line in result.stdout.splitlines()
        if CHECKPOINT_RE.fullmatch(line.strip())
    ]


def download_checkpoints(run_document: Path) -> dict[str, Any]:
    run_document = run_document.resolve()
    run = read_json(run_document)
    repository_root = Path(run["repository_root"])
    paths = RunPaths.create(repository_root, run["run_id"], Path(run["document_root"]))
    transport = _transport_from_run(run)
    remote_output = run["remote_output"]
    local_root = paths.root / "checkpoints"
    local_root.mkdir(parents=True, exist_ok=True)
    inventory = (
        read_json(paths.inventory)
        if paths.inventory.exists()
        else {"schema_version": 1, "run_id": run["run_id"], "checkpoints": {}}
    )
    for name in _completed_remote_checkpoints(transport, remote_output):
        final = local_root / name
        if final.exists():
            valid, reason = verify_checkpoint(final)
            if not valid:
                raise RuntimeError(f"existing local {name} is invalid: {reason}")
            inventory["checkpoints"].setdefault(
                name, {"path": str(final), "verified": True, "already_present": True}
            )
            continue
        temporary = local_root / f".{name}.download-{uuid.uuid4().hex}"
        temporary.mkdir()
        try:
            transport.download_directory(
                f"{remote_output}/checkpoints/{name}", temporary
            )
            downloaded = temporary / name
            valid, reason = verify_checkpoint(downloaded)
            if not valid:
                raise RuntimeError(f"downloaded {name} is invalid: {reason}")
            os.replace(downloaded, final)
            shutil.rmtree(temporary)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        record = {
            "path": str(final),
            "verified": True,
            "downloaded_utc": utc_now(),
            "manifest_sha256": sha256_file(final / "checkpoint_manifest.json"),
        }
        inventory["checkpoints"][name] = record
        inventory["updated_utc"] = utc_now()
        atomic_json(paths.inventory, inventory)
        append_event(paths.events, "checkpoint_downloaded", checkpoint=name)
    inventory["updated_utc"] = utc_now()
    atomic_json(paths.inventory, inventory)
    run["checkpoint_inventory"] = str(paths.inventory)
    run["verified_checkpoint_count"] = len(inventory["checkpoints"])
    run["updated_utc"] = utc_now()
    atomic_json(paths.run, run)
    return inventory


FINAL_ARTIFACTS = (
    "selected-model",
    "evaluation",
    "checkpoint-probes",
    "logs",
    "run_provenance.json",
    "training_metadata.json",
    "approach_runtime.json",
    "parameter_inventory.json",
    "initial_parameter_inventory.json",
    "pip-freeze.txt",
    "run_result.json",
    "run_artifact_manifest.json",
    "RUN_COMPLETE.json",
    "RUN_FAILED.json",
)


def download_final_artifacts(run_document: Path) -> dict[str, Any]:
    run = read_json(run_document.resolve())
    repository_root = Path(run["repository_root"])
    paths = RunPaths.create(repository_root, run["run_id"], Path(run["document_root"]))
    transport = _transport_from_run(run)
    final_root = paths.root / "final"
    final_root.mkdir(parents=True, exist_ok=True)
    downloaded = []
    for name in FINAL_ARTIFACTS:
        final = final_root / name
        if final.exists():
            downloaded.append(name)
            continue
        temporary_root = final_root / f".{name}.download-{uuid.uuid4().hex}"
        temporary_root.mkdir()
        remote = f"{run['remote_output']}/{name}"
        is_directory = (
            transport.run(f"test -d {shlex.quote(remote)}", check=False).returncode == 0
        )
        exists = is_directory or (
            transport.run(f"test -f {shlex.quote(remote)}", check=False).returncode == 0
        )
        if not exists:
            shutil.rmtree(temporary_root)
            continue
        if is_directory:
            transport.download_directory(remote, temporary_root)
            source = temporary_root / name
        else:
            source = temporary_root / name
            if not transport.download_file(remote, source):
                shutil.rmtree(temporary_root)
                raise RuntimeError(f"failed to download remote artifact {name}")
        os.replace(source, final)
        shutil.rmtree(temporary_root)
        downloaded.append(name)

    complete = (final_root / "RUN_COMPLETE.json").is_file()
    required = {
        "selected-model",
        "evaluation",
        "run_result.json",
        "run_artifact_manifest.json",
        "RUN_COMPLETE.json",
    }
    missing = sorted(required - set(downloaded))
    manifest_errors: list[str] = []
    manifest_path = final_root / "run_artifact_manifest.json"
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
        for entry in manifest.get("files", []):
            path = final_root / entry["path"]
            if not path.is_file():
                manifest_errors.append(f"missing: {entry['path']}")
            elif path.stat().st_size != int(entry["bytes"]):
                manifest_errors.append(f"size mismatch: {entry['path']}")
            elif sha256_file(path) != entry["sha256"]:
                manifest_errors.append(f"sha256 mismatch: {entry['path']}")
        if complete:
            marker = read_json(final_root / "RUN_COMPLETE.json")
            expected_manifest_sha = marker.get("artifact_manifest_sha256")
            if expected_manifest_sha and expected_manifest_sha != sha256_file(
                manifest_path
            ):
                manifest_errors.append("run artifact manifest hash mismatch")
    else:
        manifest_errors.append("run artifact manifest missing")
    report = {
        "schema_version": 1,
        "run_id": run["run_id"],
        "downloaded": sorted(downloaded),
        "required_missing": missing,
        "manifest_errors": manifest_errors,
        "verified": complete and not missing and not manifest_errors,
        "verified_utc": utc_now(),
    }
    atomic_json(paths.root / "final-download.json", report)
    run["artifacts_verified"] = bool(report["verified"])
    run["status"] = "artifacts_verified" if report["verified"] else run.get("status")
    run["updated_utc"] = utc_now()
    atomic_json(paths.run, run)
    append_event(
        paths.events,
        "final_artifacts_downloaded",
        verified=report["verified"],
        missing=missing,
    )
    return report


def terminate_pod(
    run_document: Path,
    *,
    force: bool = False,
    client: RunPodClient | None = None,
) -> dict[str, Any]:
    run = read_json(run_document.resolve())
    if not force and not run.get("artifacts_verified"):
        raise RuntimeError(
            "refusing to terminate before final artifacts are verified; "
            "pass --force only for an intentionally abandoned/failed run"
        )
    repository_root = Path(run["repository_root"])
    paths = RunPaths.create(repository_root, run["run_id"], Path(run["document_root"]))
    provider = client or client_from_repository(repository_root)
    provider.terminate_pod(run["pod_id"])
    confirmed = False
    for _ in range(12):
        try:
            provider.get_pod(run["pod_id"])
        except RuntimeError as exc:
            if "(404)" in str(exc):
                confirmed = True
                break
            raise
        time.sleep(5)
    if not confirmed:
        raise RuntimeError(
            f"pod {run['pod_id']} still exists after termination request"
        )
    run.update(status="terminated", pod_terminated=True, terminated_utc=utc_now())
    atomic_json(paths.run, run)
    if paths.pod.exists():
        pod_document = read_json(paths.pod)
        pod_document.update(status="TERMINATED", terminated_utc=run["terminated_utc"])
        atomic_json(paths.pod, pod_document)
    append_event(paths.events, "pod_terminated", force=force)
    return run


def watch_run(
    run_document: Path,
    *,
    poll_seconds: int = 20,
    terminate_on_complete: bool = False,
    maximum_polls: int | None = None,
) -> dict[str, Any]:
    polls = 0
    while True:
        status = collect_status(run_document)
        if status["provider_status"] == "TERMINATED":
            return status
        if status["phase"] not in {"pod_ready_not_started", "pod_unavailable"}:
            download_checkpoints(run_document)
        if status["phase"] == "completed_remote":
            report = download_final_artifacts(run_document)
            if terminate_on_complete and report["verified"]:
                terminate_pod(run_document)
            return collect_status(run_document)
        if status["phase"] in {"failed_remote", "process_stopped"}:
            download_final_artifacts(run_document)
            return status
        polls += 1
        if maximum_polls is not None and polls >= maximum_polls:
            return status
        time.sleep(poll_seconds)
