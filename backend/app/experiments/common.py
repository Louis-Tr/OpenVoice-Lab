"""Shared deterministic file helpers for experiment services."""

import hashlib
import json
import os
from pathlib import Path
from typing import Any


class ExperimentEvidenceError(RuntimeError):
    """Raised when locked experiment evidence is missing or inconsistent."""


class ExperimentFixtureNotFoundError(LookupError):
    """Raised when a locked fixture identifier is unknown."""


class ExperimentModelNotFoundError(LookupError):
    """Raised when an experiment model identifier is unknown."""


class ExperimentJobNotFoundError(LookupError):
    """Raised when a comparison job identifier is unknown."""


class ExperimentQueueFullError(RuntimeError):
    """Raised when the bounded CPU comparison queue has no capacity."""


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ExperimentEvidenceError(f"Required experiment evidence is missing: {path}") from error
    except json.JSONDecodeError as error:
        raise ExperimentEvidenceError(f"Experiment evidence is invalid JSON: {path}") from error
    if not isinstance(value, dict):
        raise ExperimentEvidenceError(f"Experiment evidence must be an object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
