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
    precision: Literal["FP32", "INT8"]
    variant: ModelVariant
    model_version: str
    model_path: Path
    voices_path: Path
    voices: tuple[str, ...]
    runtime: str = "ONNX"
    hosting: str = "self-hosted"
    language: str = "en-us"
    speed: float = 1.0

    @property
    def key(self) -> str:
        return self.model_id

    @property
    def label(self) -> str:
        return self.model_id

    @property
    def artifacts_available(self) -> bool:
        return self.model_path.is_file() and self.voices_path.is_file()


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
            )
            for model in self._models.values()
        ]
