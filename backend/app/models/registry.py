"""Technology-neutral model catalog boundary."""

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from app.schemas.model import ModelSummary
from app.schemas.synthesis import ModelVariant


class ModelNotFoundError(LookupError):
    """Raised when a requested model configuration is not registered."""


@dataclass(frozen=True, slots=True)
class ModelDefinition:
    """Backend-only metadata needed to load a model implementation."""

    model_id: str
    display_name: str
    precision: str
    variant: ModelVariant
    model_version: str
    model_path: Path
    voices_path: Path | None
    voices: tuple[str, ...]
    runtime: str = "ONNX"
    engine: Literal["kokoro-onnx", "speecht5-transformers", "audio8-onnx"] = "kokoro-onnx"
    hosting: str = "self-hosted"
    language: str = "en-us"
    speed: float = 1.0
    additional_artifacts: tuple[Path, ...] = ()
    availability_markers: tuple[Path, ...] = ()
    enabled: bool = True
    unavailable_reason: str | None = None
    benchmark_enabled: bool = True
    description: str = "Local open-weight text-to-speech model."

    @property
    def key(self) -> str:
        return self.model_id

    @property
    def label(self) -> str:
        return self.model_id

    @property
    def artifacts_available(self) -> bool:
        return self.enabled and all(path.exists() for path in self.required_artifacts)

    @property
    def required_artifacts(self) -> tuple[Path, ...]:
        """Return every local artifact required to construct this engine."""
        paths = [self.model_path]
        if self.voices_path is not None:
            paths.append(self.voices_path)
        paths.extend(self.additional_artifacts)
        paths.extend(self.availability_markers)
        return tuple(paths)

    @property
    def resolved_unavailable_reason(self) -> str | None:
        if self.artifacts_available:
            return None
        if self.unavailable_reason:
            return self.unavailable_reason
        missing = [str(path) for path in self.required_artifacts if not path.exists()]
        return "Missing local model artifacts: " + ", ".join(missing)


class ModelRegistry:
    """Resolve model metadata without owning runtime sessions."""

    def __init__(self, models: Iterable[ModelDefinition]) -> None:
        definitions = tuple(models)
        self._models = {model.key: model for model in definitions}
        if len(self._models) != len(definitions):
            raise ValueError("Duplicate model identifiers are not allowed.")

    def get(self, model_id: str) -> ModelDefinition:
        """Resolve a deployable configuration or return a useful domain error."""
        try:
            return self._models[model_id]
        except KeyError as error:
            raise ModelNotFoundError(f"Model '{model_id}' is not registered.") from error

    def list_available(self) -> list[ModelSummary]:
        """Expose public model metadata without runtime implementation details."""
        return [
            ModelSummary(
                id=model.model_id,
                name=model.display_name,
                precision=model.precision,
                variant=model.variant,
                voices=list(model.voices),
                model_version=model.model_version,
                runtime=model.runtime,
                hosting=model.hosting,
                external_inference_apis=[],
                available=model.artifacts_available,
                unavailable_reason=model.resolved_unavailable_reason,
                description=model.description,
            )
            for model in self._models.values()
        ]

    def list_benchmark_models(self) -> list[ModelSummary]:
        """Return deployed models whose voice contract matches the benchmark corpus."""
        by_id = {summary.id: summary for summary in self.list_available()}
        return [
            by_id[model.model_id]
            for model in self._models.values()
            if model.benchmark_enabled and model.artifacts_available
        ]
