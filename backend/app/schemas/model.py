"""Model discovery contracts."""

from typing import Literal

from app.schemas.base import ApiSchema
from app.schemas.synthesis import ModelVariant


class ModelSummary(ApiSchema):
    """One selectable, deployable model configuration."""

    id: str
    name: str
    precision: Literal["FP32", "INT8"]
    variant: ModelVariant
    voices: list[str]
    model_version: str
    runtime: str
    hosting: str
    external_inference_apis: list[str]
    available: bool
