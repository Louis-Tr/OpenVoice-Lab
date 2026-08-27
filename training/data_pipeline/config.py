from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PipelineConfig:
    repository_root: Path
    config_path: Path
    values: dict[str, Any]

    def path(self, key: str) -> Path:
        value = self.values["paths"][key]
        path = Path(value)
        return path if path.is_absolute() else (self.repository_root / path).resolve()

    def section(self, key: str) -> dict[str, Any]:
        return self.values[key]

    @property
    def seed(self) -> int:
        return int(self.values["seed"])

    @property
    def digest(self) -> str:
        canonical = json.dumps(self.values, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(canonical.encode("utf-8"))
        medical_terms = self.path("medical_terms")
        if medical_terms.is_file():
            digest.update(b"\0medical_terms\0")
            digest.update(medical_terms.read_bytes())
        return digest.hexdigest()


def discover_repository_root(config_path: Path) -> Path:
    resolved = config_path.resolve()
    for parent in (resolved.parent, *resolved.parents):
        if (parent / ".git").exists():
            return parent
    raise ValueError(f"Could not find repository root above {config_path}")


def load_config(path: str | Path) -> PipelineConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle)
    if not isinstance(values, dict) or values.get("schema_version") != 1:
        raise ValueError("dataset config must be a schema_version: 1 mapping")
    repository_root = discover_repository_root(config_path)
    config = PipelineConfig(repository_root, config_path, values)
    _validate(config)
    return config


def _validate(config: PipelineConfig) -> None:
    raw_root = config.path("raw_root")
    duplicate_root = config.path("duplicate_raw_root")
    if raw_root == duplicate_root:
        raise ValueError("canonical and duplicate raw roots must differ")
    if raw_root.name != "Medical Speech, Transcription, and Intent":
        raise ValueError(
            "canonical raw_root must select the confirmed upper-case dataset root"
        )
    ratios = config.section("splits")
    total = sum(float(ratios[name]) for name in ("train", "validation", "test"))
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"split ratios must total 1.0, got {total}")
    duration = config.section("audio")["duration"]
    if float(duration["minimum_seconds"]) >= float(duration["maximum_seconds"]):
        raise ValueError("minimum duration must be less than maximum duration")
