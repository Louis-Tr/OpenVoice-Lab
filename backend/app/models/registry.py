"""Technology-neutral model catalog boundary."""

from dataclasses import dataclass

from app.schemas.synthesis import ModelVariant


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Registry metadata needed to locate a model artifact."""

    model_id: str
    display_name: str
    artifact_path: str
    voices: tuple[str, ...]
    variant: ModelVariant


class ModelRegistry:
    """Resolve model identifiers without loading inference artifacts."""

    def get(self, _model_id: str, _variant: ModelVariant) -> ModelSpec:
        """Resolve a registered model variant."""
        raise NotImplementedError("Model registry is not configured yet.")

    def list(self) -> tuple[ModelSpec, ...]:
        """List configured model variants."""
        return ()

