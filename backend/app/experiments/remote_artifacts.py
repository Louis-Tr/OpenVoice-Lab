"""Atomic, checksum-verified provisioning for remotely stored experiment models."""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from threading import Lock
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen
from uuid import uuid4

from app.experiments.common import ExperimentEvidenceError, sha256_file
from app.experiments.model_registry import ExperimentModelDefinition

_SAFE_SEGMENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


@dataclass(frozen=True, slots=True)
class RemoteModelFile:
    """One immutable file in a remotely provisioned model directory."""

    path: PurePosixPath
    bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class RemoteModelArtifact:
    """Complete remote inventory for one public experiment model ID."""

    model_id: str
    files: tuple[RemoteModelFile, ...]


class RemoteModelCatalog:
    """Validated remote model inventory committed independently from large weights."""

    def __init__(self, artifacts: tuple[RemoteModelArtifact, ...]) -> None:
        self._artifacts = {artifact.model_id: artifact for artifact in artifacts}
        if len(self._artifacts) != len(artifacts):
            raise ExperimentEvidenceError("Remote experiment model IDs must be unique.")

    @classmethod
    def from_path(cls, path: Path) -> RemoteModelCatalog:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ExperimentEvidenceError(
                f"Remote experiment model manifest is unreadable: {path}"
            ) from error
        if payload.get("schema_version") != 1 or not isinstance(payload.get("models"), list):
            raise ExperimentEvidenceError("Remote experiment model manifest is unsupported.")

        artifacts: list[RemoteModelArtifact] = []
        for raw_model in payload["models"]:
            if not isinstance(raw_model, dict) or not isinstance(raw_model.get("files"), list):
                raise ExperimentEvidenceError("Remote experiment model entry is malformed.")
            model_id = str(raw_model.get("id", "")).strip()
            if not _SAFE_SEGMENT.fullmatch(model_id):
                raise ExperimentEvidenceError("Remote experiment model ID is missing.")
            files: list[RemoteModelFile] = []
            for raw_file in raw_model["files"]:
                if not isinstance(raw_file, dict):
                    raise ExperimentEvidenceError("Remote experiment file entry is malformed.")
                relative = PurePosixPath(str(raw_file.get("path", "")))
                digest = str(raw_file.get("sha256", ""))
                try:
                    size = int(raw_file.get("bytes", -1))
                except (TypeError, ValueError) as error:
                    raise ExperimentEvidenceError(
                        f"Remote experiment file metadata is invalid for {model_id}."
                    ) from error
                if (
                    relative.is_absolute()
                    or not relative.parts
                    or any(part in {"", ".", ".."} for part in relative.parts)
                    or any(not _SAFE_SEGMENT.fullmatch(part) for part in relative.parts)
                    or len(digest) != 64
                    or any(character not in "0123456789abcdef" for character in digest)
                    or size < 0
                ):
                    raise ExperimentEvidenceError(
                        f"Remote experiment file metadata is invalid for {model_id}."
                    )
                files.append(RemoteModelFile(relative, size, digest))
            if not files or len({item.path for item in files}) != len(files):
                raise ExperimentEvidenceError(
                    f"Remote experiment file inventory is invalid for {model_id}."
                )
            artifacts.append(RemoteModelArtifact(model_id, tuple(files)))
        return cls(tuple(artifacts))

    @property
    def model_ids(self) -> frozenset[str]:
        return frozenset(self._artifacts)

    def get(self, model_id: str) -> RemoteModelArtifact:
        try:
            return self._artifacts[model_id]
        except KeyError as error:
            raise ExperimentEvidenceError(
                f"No remote artifact inventory exists for {model_id}."
            ) from error


DownloadFile = Callable[[str, Path, str | None, int], None]


class ExperimentModelProvisioner:
    """Download one model into an atomic cache directory and verify every byte."""

    def __init__(
        self,
        *,
        base_url: str,
        catalog: RemoteModelCatalog,
        bearer_token: str | None = None,
        timeout_seconds: int = 300,
        download_file: DownloadFile | None = None,
    ) -> None:
        parsed_base_url = urlparse(base_url.strip())
        if parsed_base_url.scheme != "https" or not parsed_base_url.netloc:
            raise ValueError("base_url must be an absolute HTTPS URL")
        self._base_url = base_url.rstrip("/")
        self._catalog = catalog
        self._bearer_token = bearer_token
        self._timeout_seconds = timeout_seconds
        self._download_file = download_file or self._download
        self._lock = Lock()

    def ensure(self, definition: ExperimentModelDefinition) -> Path:
        """Return a verified local model directory, downloading it once if needed."""
        artifact = self._catalog.get(definition.id)
        weight = next(
            (
                item
                for item in artifact.files
                if item.path.name in {"model.safetensors", "pytorch_model.bin"}
            ),
            None,
        )
        if weight is None or weight.sha256 != definition.model_sha256:
            raise ExperimentEvidenceError(
                f"Remote model weight identity differs for {definition.id}."
            )
        with self._lock:
            if definition.local_available:
                self._verify(definition.source, artifact)
                return definition.source
            if definition.source.exists():
                raise ExperimentEvidenceError(
                    f"Refusing to replace incomplete experiment model cache: {definition.source}"
                )

            definition.source.parent.mkdir(parents=True, exist_ok=True)
            temporary = definition.source.with_name(
                f".{definition.source.name}.{uuid4().hex}.download"
            )
            temporary.mkdir()
            try:
                for item in artifact.files:
                    destination = temporary.joinpath(*item.path.parts)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    self._download_file(
                        self._url(definition.id, item.path),
                        destination,
                        self._bearer_token,
                        self._timeout_seconds,
                    )
                    self._verify_file(destination, item)
                marker = {
                    "schema_version": 1,
                    "model_id": definition.id,
                    "model_sha256": definition.model_sha256,
                    "files": [
                        {
                            "path": item.path.as_posix(),
                            "bytes": item.bytes,
                            "sha256": item.sha256,
                        }
                        for item in artifact.files
                    ],
                }
                (temporary / "openvoice-remote-manifest.json").write_text(
                    json.dumps(marker, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                temporary.replace(definition.source)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
            self._verify(definition.source, artifact)
            if not definition.local_available:
                raise ExperimentEvidenceError(
                    f"Provisioned model has no loadable weights: {definition.id}"
                )
            return definition.source

    def _verify(self, root: Path, artifact: RemoteModelArtifact) -> None:
        for item in artifact.files:
            self._verify_file(root.joinpath(*item.path.parts), item)

    @staticmethod
    def _verify_file(path: Path, expected: RemoteModelFile) -> None:
        if (
            not path.is_file()
            or path.stat().st_size != expected.bytes
            or sha256_file(path) != expected.sha256
        ):
            raise ExperimentEvidenceError(f"Experiment model file verification failed: {path}")

    def _url(self, model_id: str, relative: PurePosixPath) -> str:
        encoded = "/".join(quote(part, safe="") for part in (model_id, *relative.parts))
        return f"{self._base_url}/{encoded}"

    @staticmethod
    def _download(
        url: str,
        destination: Path,
        bearer_token: str | None,
        timeout_seconds: int,
    ) -> None:
        headers = {"User-Agent": "OpenVoice-Lab-experiment-provisioner/1.0"}
        if bearer_token:
            headers["Authorization"] = f"Bearer {bearer_token}"
        request = Request(url, headers=headers)
        with (
            urlopen(request, timeout=timeout_seconds) as response,
            destination.open("xb") as output,
        ):
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
