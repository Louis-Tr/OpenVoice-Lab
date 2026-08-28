"""Immutable SpeechT5 comparison catalog independent from the Kokoro registry."""

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from app.experiments.approach_catalog import V1_APPROACHES
from app.experiments.common import (
    ExperimentEvidenceError,
    ExperimentModelNotFoundError,
    read_json,
    sha256_file,
)
from app.schemas.experiment import ExperimentModelId, ExperimentModelSummary


@dataclass(frozen=True, slots=True)
class ExperimentModelDefinition:
    """Private model-loading metadata for one comparison target."""

    id: ExperimentModelId
    name: str
    role: Literal["pretrained", "adapted"]
    variant: str
    source: Path
    revision: str
    model_sha256: str | None

    @property
    def available(self) -> bool:
        return self.source.is_dir() and any(
            (self.source / name).is_file()
            for name in ("model.safetensors", "pytorch_model.bin")
        )


class ExperimentModelRegistry:
    """Resolve public experiment IDs without exposing local paths."""

    def __init__(self, definitions: Iterable[ExperimentModelDefinition]) -> None:
        values = tuple(definitions)
        self._definitions = {item.id: item for item in values}
        if len(self._definitions) != len(values):
            raise ValueError("Duplicate experiment model IDs are not allowed.")

    @classmethod
    def from_artifacts(
        cls,
        artifact_root: Path,
        pretrained_root: Path,
        tts_revision: str,
    ) -> "ExperimentModelRegistry":
        definitions: list[ExperimentModelDefinition] = [
            ExperimentModelDefinition(
                id="speecht5-pretrained",
                name="SpeechT5 Pretrained",
                role="pretrained",
                variant="pretrained",
                source=pretrained_root.resolve(),
                revision=tts_revision,
                model_sha256=_local_weight_hash(pretrained_root.resolve()),
            )
        ]
        for variant_id, public_id, name in (
            ("v1-baseline", "speecht5-v1-baseline", "SpeechT5 V1 Baseline"),
            (
                "v2-term-balance",
                "speecht5-v2-term-balance",
                "SpeechT5 V2 Term Balance",
            ),
            ("v3-replay", "speecht5-v3-replay", "SpeechT5 V3 Replay"),
        ):
            root = artifact_root.resolve() / variant_id
            provenance = read_json(root / "run_provenance.json")
            manifest = read_json(root / "run_artifact_manifest.json")
            selected = root / "selected-model"
            definitions.append(
                ExperimentModelDefinition(
                    id=public_id,
                    name=name,
                    role="adapted",
                    variant=variant_id,
                    source=selected,
                    revision=provenance["models"]["tts_revision"],
                    model_sha256=_manifest_weight_hash(manifest, selected),
                )
            )
        return cls(definitions)

    @classmethod
    def from_v1_approach_runs(
        cls,
        approach_run_root: Path,
        pretrained_root: Path,
        tts_revision: str,
    ) -> "ExperimentModelRegistry":
        """Build the live catalog from the four pinned, verified agent runs."""
        definitions: list[ExperimentModelDefinition] = [
            ExperimentModelDefinition(
                id="speecht5-pretrained",
                name="SpeechT5 Pretrained",
                role="pretrained",
                variant="pretrained",
                source=pretrained_root.resolve(),
                revision=tts_revision,
                model_sha256=_local_weight_hash(pretrained_root.resolve()),
            )
        ]
        for approach in V1_APPROACHES:
            final_root = approach_run_root.resolve() / approach.run_id / "final"
            provenance = read_json(final_root / "run_provenance.json")
            manifest = read_json(final_root / "run_artifact_manifest.json")
            selected = final_root / "selected-model"
            if (
                provenance.get("variant") != approach.variant_id
                or provenance.get("run_id") != approach.run_id
            ):
                raise ExperimentEvidenceError(
                    f"The selected run identity differs for {approach.variant_id}."
                )
            definitions.append(
                ExperimentModelDefinition(
                    id=approach.model_id,
                    name=approach.name,
                    role="adapted",
                    variant=approach.variant_id,
                    source=selected,
                    revision=provenance["models"]["tts_revision"],
                    model_sha256=_manifest_weight_hash(manifest, selected),
                )
            )
        return cls(definitions)

    def get(self, model_id: str) -> ExperimentModelDefinition:
        try:
            return self._definitions[model_id]  # type: ignore[index]
        except KeyError as error:
            raise ExperimentModelNotFoundError(
                f"Experiment model '{model_id}' was not found."
            ) from error

    def list(self) -> list[ExperimentModelSummary]:
        return [
            ExperimentModelSummary(
                id=item.id,
                name=item.name,
                role=item.role,
                variant=item.variant,
                revision=item.revision,
                model_sha256=item.model_sha256,
                available=item.available,
                unavailable_reason=(
                    None if item.available else "Pinned model artifacts have not been provisioned."
                ),
            )
            for item in self._definitions.values()
        ]


def _manifest_weight_hash(manifest: dict[str, object], selected: Path) -> str:
    files = manifest.get("files")
    if not isinstance(files, list):
        raise ExperimentEvidenceError("The final artifact manifest has no file inventory.")
    entry = next(
        (
            item
            for item in files
            if isinstance(item, dict)
            and item.get("path") == "selected-model/model.safetensors"
        ),
        None,
    )
    model = selected / "model.safetensors"
    if entry is None or not model.is_file() or sha256_file(model) != entry.get("sha256"):
        raise ExperimentEvidenceError(f"The selected model is incomplete: {selected}")
    return str(entry["sha256"])


def _local_weight_hash(root: Path) -> str | None:
    manifest_path = root / "openvoice-model-manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = read_json(manifest_path)
    value = manifest.get("weight_sha256")
    if not value:
        return None
    model = next(
        (root / name for name in ("model.safetensors", "pytorch_model.bin") if (root / name).is_file()),
        None,
    )
    if model is None or sha256_file(model) != value:
        raise ExperimentEvidenceError(f"The pinned pretrained model hash differs: {root}")
    return str(value)
