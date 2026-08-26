"""Model discovery contracts."""

from pydantic import BaseModel

from app.schemas.synthesis import ModelVariant


class ModelSummary(BaseModel):
    """A selectable model with voices and available precision variants."""

    id: str
    display_name: str
    voices: list[str]
    variants: list[ModelVariant]

