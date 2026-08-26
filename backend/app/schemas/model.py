"""Model discovery contracts."""

from app.schemas.base import ApiSchema
from app.schemas.synthesis import ModelVariant


class ModelSummary(ApiSchema):
    """A selectable model with voices and available precision variants."""

    id: str
    display_name: str
    voices: list[str]
    variants: list[ModelVariant]
    model_version: str
    runtime: str
    hosting: str
    external_inference_apis: list[str]
    available: bool
