"""Conservative resource profiles for scheduler admission."""

from dataclasses import dataclass

from app.models.registry import ModelDefinition


@dataclass(frozen=True, slots=True)
class ResourceDemand:
    """Incremental resource commitment for one admitted unit of work."""

    memory_mb: float
    cpu_cores: float
    resource_key: str
    exclusive: bool = False
    profile_name: str = "custom"


@dataclass(frozen=True, slots=True)
class ModelResourceProfile:
    model_id: str
    resident_memory_mb: float
    peak_load_memory_mb: float
    inference_memory_mb: float
    cpu_cores: float

    def demand(self, *, loaded: bool, input_characters: int) -> ResourceDemand:
        size_factor = 1.0 + min(0.5, input_characters / 10_000)
        inference = self.inference_memory_mb * size_factor
        memory = inference if loaded else max(
            self.peak_load_memory_mb,
            self.resident_memory_mb + inference,
        )
        return ResourceDemand(
            memory_mb=memory,
            cpu_cores=self.cpu_cores,
            resource_key=self.model_id,
            profile_name=self.model_id,
        )


class ResourceProfileRegistry:
    """Resolve versioned model estimates without coupling them to API metadata."""

    def __init__(self, profiles: dict[str, ModelResourceProfile] | None = None) -> None:
        self._profiles = profiles or self._defaults()

    def for_model(
        self,
        model: ModelDefinition,
        *,
        loaded: bool,
        input_characters: int,
    ) -> ResourceDemand:
        profile = self._profiles.get(model.model_id)
        if profile is None:
            return ResourceDemand(
                memory_mb=1536.0,
                cpu_cores=2.0,
                resource_key=model.model_id,
                exclusive=True,
                profile_name=f"unknown:{model.model_id}",
            )
        return profile.demand(loaded=loaded, input_characters=input_characters)

    @staticmethod
    def benchmark() -> ResourceDemand:
        return ResourceDemand(2048.0, 2.0, "benchmark", True, "benchmark")

    @staticmethod
    def experiment(resource_key: str = "experiment") -> ResourceDemand:
        return ResourceDemand(2048.0, 2.0, resource_key, False, "experiment")

    @staticmethod
    def _defaults() -> dict[str, ModelResourceProfile]:
        return {
            "kokoro-fp32": ModelResourceProfile(
                "kokoro-fp32", 650.0, 950.0, 220.0, 2.0
            ),
            "kokoro-fp16": ModelResourceProfile(
                "kokoro-fp16", 450.0, 700.0, 180.0, 2.0
            ),
            "kokoro-q8": ModelResourceProfile(
                "kokoro-q8", 350.0, 560.0, 160.0, 1.5
            ),
            "speecht5-pretrained": ModelResourceProfile(
                "speecht5-pretrained", 1050.0, 1750.0, 420.0, 2.0
            ),
        }
