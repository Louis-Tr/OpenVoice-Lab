"""Model loading and lifecycle boundary."""

from typing import Any

from app.models.registry import ModelSpec


class ModelLoader:
    """Load and cache runtime state independently of request controllers."""

    def load(self, _model: ModelSpec) -> Any:
        """Load model state when a concrete runtime is introduced."""
        raise NotImplementedError("Model loading is not implemented yet.")

    def unload(self, _model_id: str) -> None:
        """Release cached state when lifecycle policy is introduced."""
        raise NotImplementedError("Model unloading is not implemented yet.")

