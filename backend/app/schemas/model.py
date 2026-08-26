"""Model discovery contracts."""

from app.schemas.base import ApiSchema
from app.schemas.synthesis import ModelVariant


class ModelSummary(ApiSchema):
    """A selectable model with voices and available precision variants."""

    id: str
    display_name: str
    voices: list[str]
    variants: list[ModelVariant]
