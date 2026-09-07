"""Model lifecycle coverage for memory-bounded serving."""

from pathlib import Path

import numpy as np

from app.inference.base import AudioResult, TTSInferenceEngine
from app.metrics.collector import MetricsCollector
from app.models.loader import ModelLoader
from app.models.registry import ModelDefinition
from app.models.resources import ResourceManager
from app.models.scheduler import EngineScheduler


class CloseableEngine(TTSInferenceEngine):
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.closed = False

    @property
    def voices(self) -> tuple[str, ...]:
        return ("voice",)

    def synthesize(
        self,
        text: str,
        voice: str,
        *,
        speed: float = 1.0,
        language: str = "en-us",
    ) -> AudioResult:
        del text, voice, speed, language
        return AudioResult(samples=np.ones(10, dtype=np.float32), sample_rate_hz=10)

    def close(self) -> None:
        self.closed = True


def definition(model_id: str, path: Path) -> ModelDefinition:
    path.write_bytes(model_id.encode())
    return ModelDefinition(
        model_id=model_id,
        display_name=model_id,
        precision="INT8",
        variant="quantized",
        model_version="test",
        model_path=path,
        voices_path=None,
        voices=("voice",),
    )


def test_bounded_loader_evicts_lru_engine_and_reloads_on_demand(tmp_path: Path) -> None:
    created: list[CloseableEngine] = []

    def create(model: ModelDefinition) -> CloseableEngine:
        engine = CloseableEngine(model.model_id)
        created.append(engine)
        return engine

    first = definition("first", tmp_path / "first.bin")
    second = definition("second", tmp_path / "second.bin")
    loader = ModelLoader(engine_factory=create)
    scheduler = EngineScheduler(
        loader,
        ResourceManager(
            profiles={"first": 1, "second": 1},
            memory_reader=lambda: 100_000,
            cpu_threads=1,
        ),
        MetricsCollector(),
        maximum_cached_engines=1,
    )

    with scheduler.acquire(first) as first_load:
        assert first_load.warm is False
    with scheduler.acquire(first) as cached:
        assert cached.warm is True

    with scheduler.acquire(second):
        pass
    assert created[0].closed is True
    assert loader.load_count("first") == 1
    assert loader.load_count("second") == 1

    with scheduler.acquire(first):
        pass
    assert created[1].closed is True
    assert loader.load_count("first") == 2
