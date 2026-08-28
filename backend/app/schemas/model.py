"""Model discovery contracts."""

from app.schemas.base import ApiSchema
from app.schemas.synthesis import ModelVariant


class ModelSummary(ApiSchema):
    """One selectable, deployable model configuration."""

    id: str
    name: str
    precision: str
    variant: ModelVariant
    voices: list[str]
    model_version: str
    runtime: str
    hosting: str
    external_inference_apis: list[str]
    available: bool
    unavailable_reason: str | None = None
    description: str = "Local open-weight text-to-speech model."
