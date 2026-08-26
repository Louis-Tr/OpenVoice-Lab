"""Technology-neutral model catalog boundary."""

from collections.abc import Iterable

from app.schemas.model import ModelSummary


class ModelRegistry:
    """Expose configured model metadata without loading artifacts."""

    def __init__(self, models: Iterable[ModelSummary]) -> None:
        self._models = tuple(models)

    def list_available(self) -> list[ModelSummary]:
        """Return defensive copies of configured API metadata."""
        return [model.model_copy(deep=True) for model in self._models]
