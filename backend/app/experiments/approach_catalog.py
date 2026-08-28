"""Pinned catalog for the four completed Stage 11 V1 approach runs."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class V1ApproachDefinition:
    """Stable public identity for one verified training approach."""

    variant_id: str
    run_id: str
    model_id: str
    name: str


V1_APPROACHES = (
    V1ApproachDefinition(
        variant_id="v1a-conservative-full",
        run_id="v1a-real-20260828-001",
        model_id="speecht5-v1a-conservative-full",
        name="V1A Conservative Full",
    ),
    V1ApproachDefinition(
        variant_id="v1b-lora",
        run_id="v1b-real-20260828-001",
        model_id="speecht5-v1b-lora",
        name="V1B LoRA",
    ),
    V1ApproachDefinition(
        variant_id="v1c-gradual-unfreeze",
        run_id="v1c-real-20260828-001",
        model_id="speecht5-v1c-gradual-unfreeze",
        name="V1C Gradual Unfreeze",
    ),
    V1ApproachDefinition(
        variant_id="v1d-reduction-factor-1",
        run_id="v1d-real-20260828-001",
        model_id="speecht5-v1d-reduction-factor-1",
        name="V1D Reduction Factor 1",
    ),
)

